import Foundation
import ArgumentParser
import MLX
import EdgeTAM

/// EdgeTAM P2 video parity gate (CPU fp32). Op-level parity vs the de-risked goldens
/// (perceiver / mem-encoder / mem-attention / track-decode + position-encoding generation +
/// memory-bank assembly pos), then the full 5-frame masklet propagation vs `vid_mask_f0..4`.
@main
struct VideoSmoke: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "edgetam-video-smoke",
        abstract: "EdgeTAM MLX-Swift P2 video memory-stack parity gate.")

    @Option(name: .long) var weights: String
    @Option(name: .long, help: "parity_video.safetensors") var parity: String
    @Option(name: .long, help: "GPU flat-footprint measurement: cycle the 5 embedded frames to N, stream through propagate, report MLX.GPU.peakMemory (0 = CPU parity gate).")
    var measureFrames: Int = 0

    func run() throws {
        if measureFrames > 0 { try measure(n: measureFrames); return }
        Device.setDefault(device: Device(.cpu))
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weights)).mapValues { $0.asType(.float32) }
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: parity)).mapValues { $0.asType(.float32) }
        let m = EdgeTAMModel(weights: w)
        var failed = false
        func report(_ name: String, _ err: Float, _ thr: Float) {
            let ok = err < thr
            failed = failed || !ok
            print(String(format: "[vid-smoke] %-22s max_abs=%.3e  thr=%.0e  %@",
                         (name as NSString).utf8String!, err, thr, ok ? "OK ✅" : "FAIL ❌"))
        }
        func maxAbs(_ x: MLXArray, _ y: MLXArray) -> Float { MLX.abs(x - y).max().item(Float.self) }

        // 1) Position-encoding generation (content-independent) vs captured goldens.
        let currPos = EdgeTAMModel.posSine(256, 64, 64).reshaped([4096, 1, 256])
        report("pos curr (256,64²)", maxAbs(currPos, fx["ma_curr_pos"]!), 1e-3)
        let pPos = EdgeTAMModel.posSine(64, 64, 64).reshaped([1, 64, 64, 64])
        report("pos perc-in (64,64²)", maxAbs(pPos, fx["perc_pos_in"]!), 1e-3)

        // 2) PerceiverResampler.
        let (pLat, pPosOut) = m.perceiver(fx["perc_in"]!, fx["perc_pos_in"]!)
        pLat.eval()
        report("perceiver", maxAbs(pLat, fx["perc_out"]!), 1e-3)

        // 3) Memory-bank assembly pos: perceiver pos + maskmem_tpos_enc[6] ; obj-ptr pos = zeros.
        let tpos6 = w["maskmem_tpos_enc"]!.asType(.float32)[6].reshaped([1, 64])
        let asmSpatial = pPosOut[0] + tpos6                               // (512,64)
        let goldSpatial = fx["ma_memory_pos"]![0 ..< 512, 0, 0...]        // (512,64)
        report("mem-pos spatial", maxAbs(asmSpatial, goldSpatial), 1e-3)
        let goldPtrPos = fx["ma_memory_pos"]![512..., 0, 0...]
        report("mem-pos obj-ptr=0", MLX.abs(goldPtrPos).max().item(Float.self), 1e-6)

        // 4) MemoryEncoder.
        let me = m.memoryEncoder(fx["me_in_feat"]!, fx["me_in_mask"]!)
        me.eval()
        report("mem-encoder", maxAbs(me, fx["me_out"]!), 1e-3)

        // 5) MemoryAttention (numSpatial=1, numObjPtr=4).
        let ma = m.memoryAttention(fx["ma_curr"]!, fx["ma_curr_pos"]!, fx["ma_memory"]!, fx["ma_memory_pos"]!,
                                   numObjPtr: 4, numSpatial: 1)           // (1,4096,256)
        ma.eval()
        report("mem-attention", maxAbs(ma, fx["ma_out"]!.transposed(1, 0, 2)), 1e-3)

        // 6) Tracking SAM head (empty prompt, multimask, obj-score gate, obj_ptr).
        let (lrBest, objPtr, objScore) = m.forwardSamHeads(fx["sh_backbone"]!, fx["sh_hrf0"]!, fx["sh_hrf1"]!,
                                                           coordsPx: nil, labels: nil)
        lrBest.eval()
        report("track mask", maxAbs(lrBest, fx["sh_low_res_best"]!), 1e-2)
        report("track obj_ptr", maxAbs(objPtr, fx["sh_obj_ptr"]!), 1e-3)
        let dScore = abs(objScore - fx["sh_obj_score"]!.item(Float.self))
        report("track obj_score", dScore, 1e-3)

        // 7) Full 5-frame masklet propagation → binary IoU vs vid_mask_f0..4.
        let frames = fx["frames"]!                                       // (5,1024,1024,3)
        let (origH, origW) = (fx["mask_f0"]!.dim(0), fx["mask_f0"]!.dim(1))
        let (masks, _) = m.propagate(frames: frames, clickFrame: 0, points: [[210, 350]], labels: [1],
                                     origH: origH, origW: origW)
        var minIoU: Float = 1
        for i in 0 ..< 5 {
            let pred = (masks[i] .> 0).asType(.float32)
            let gold = (fx["mask_f\(i)"]! .> 0).asType(.float32)
            let inter = (pred * gold).sum().item(Float.self)
            let uni = ((pred + gold) .> 0).asType(.float32).sum().item(Float.self)
            let iou = uni > 0 ? inter / uni : 1
            minIoU = min(minIoU, iou)
            print(String(format: "[vid-smoke] propagate f%d  IoU=%.4f  cov=%.2f%%", i, iou,
                         pred.mean().item(Float.self) * 100))
        }
        let ok = minIoU > 0.90
        failed = failed || !ok
        print(String(format: "[vid-smoke] %-22s min_IoU=%.4f  thr=0.90  %@",
                     ("propagate (5 frames)" as NSString).utf8String!, minIoU, ok ? "OK ✅" : "FAIL ❌"))

        // Binary-IoU helper for the v2 cases.
        func iou(_ pred: MLXArray, _ goldKey: String) -> Float {
            let p = (pred .> 0).asType(.float32)
            let gg = (fx[goldKey]! .> 0).asType(.float32)
            let inter = (p * gg).sum().item(Float.self)
            let uni = ((p + gg) .> 0).asType(.float32).sum().item(Float.self)
            return uni > 0 ? inter / uni : 1
        }

        // 8) ENHANCEMENT(v2) #1 — MULTI-OBJECT: boy + girl in ONE shared-encode pass → 2 tracks. Each must
        // match its per-object golden; object 0 (boy) must also equal the single-object track (independence).
        let objects = [
            ObjectPrompt(clickFrame: 0, points: [[210, 350]], labels: [1]),   // obj 0: boy
            ObjectPrompt(clickFrame: 0, points: [[400, 240]], labels: [1]),   // obj 1: girl
        ]
        let tracks = m.propagate(frames: frames, objects: objects, origH: origH, origW: origW)
        // Binary IoU between two Swift masks (independence check — same tensor, no golden).
        func iouAB(_ a: MLXArray, _ b: MLXArray) -> Float {
            let p = (a .> 0).asType(.float32), gg = (b .> 0).asType(.float32)
            let inter = (p * gg).sum().item(Float.self)
            let uni = ((p + gg) .> 0).asType(.float32).sum().item(Float.self)
            return uni > 0 ? inter / uni : 1
        }
        var moMin: Float = 1, indepMin: Float = 1
        for o in 0 ..< 2 {
            for i in 0 ..< 5 { moMin = min(moMin, iou(tracks[o].masks[i], "mo_obj\(o)_f\(i)")) }
            print(String(format: "[vid-smoke]   mo obj%d  cov f0=%.2f%%", o,
                         (tracks[o].masks[0] .> 0).asType(.float32).mean().item(Float.self) * 100))
        }
        // Independence: object 0 (boy) in the shared-encode multi-object pass == the single-object Swift
        // track bit-for-bit (per-object memory banks don't interact; shared encode is deterministic).
        for i in 0 ..< 5 { indepMin = min(indepMin, iouAB(tracks[0].masks[i], masks[i])) }
        let moOK = moMin > 0.90, indepOK = indepMin > 0.999
        failed = failed || !moOK || !indepOK
        print(String(format: "[vid-smoke] %-22s min_IoU=%.4f  thr=0.90  %@",
                     ("multi-object (2×5f)" as NSString).utf8String!, moMin, moOK ? "OK ✅" : "FAIL ❌"))
        print(String(format: "[vid-smoke] %-22s min_IoU=%.4f  thr=0.999 %@  (obj0==single Swift)",
                     ("  obj0 independence" as NSString).utf8String!, indepMin, indepOK ? "OK ✅" : "FAIL ❌"))

        // 9) ENHANCEMENT(v2) #2 — BOX PROMPT: box-only track (no points) vs the box-prompted golden.
        // f0 is the directly box-prompted frame (the encoding-parity test); f1–4 add propagation drift,
        // which on this ~0.4%-coverage object diverges more than the point prompt (boundary-px sensitive).
        let bx = fx["box_xyxy"]!                                          // (4,) in source px
        let box: [Float] = (0 ..< 4).map { bx[$0].item(Float.self) }
        let (bmasks, _) = m.propagate(frames: frames, clickFrame: 0, points: [], labels: [], box: box,
                                      origH: origH, origW: origW)
        var boxMin: Float = 1
        for i in 0 ..< 5 {
            let v = iou(bmasks[i], "box_f\(i)")
            boxMin = min(boxMin, v)
            print(String(format: "[vid-smoke]   box f%d  IoU=%.4f  cov=%.2f%%", i, v,
                         (bmasks[i] .> 0).asType(.float32).mean().item(Float.self) * 100))
        }
        let boxOK = boxMin > 0.85     // single-mask box-prompt frame + propagated track all parity-match
        failed = failed || !boxOK
        print(String(format: "[vid-smoke] %-22s min_IoU=%.4f  thr=0.85  %@",
                     ("box prompt (5f)" as NSString).utf8String!, boxMin, boxOK ? "OK ✅" : "FAIL ❌"))

        if failed { throw ExitCode(1) }
        print("[vid-smoke] ALL P2 PARITY GATES PASSED ✅")
    }

    /// GPU flat-footprint measurement (fp32, default Metal device — same methodology as the historical
    /// 5f=1.07GB / 30f=1.79GB stacked numbers). Cycles the 5 embedded `(1024,1024,3)` frames to `n` and
    /// drives the STREAMING `propagate(frameCount:frame:emit:)` the package surface uses: `frame(t)` slices
    /// the small 5-frame source on demand (input memory stays bounded), every emitted mask is reduced to a
    /// scalar IoU and dropped, and `clear_cache()` runs per frame inside the core. Peak then reflects the
    /// one-frame working set + the per-frame memory bank only — the flat footprint we're verifying.
    private func measure(n: Int) throws {
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weights)).mapValues { $0.asType(.float32) }
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: parity)).mapValues { $0.asType(.float32) }
        let m = EdgeTAMModel(weights: w)
        let frames = fx["frames"]!                                        // (5,1024,1024,3) on GPU
        let (origH, origW) = (fx["mask_f0"]!.dim(0), fx["mask_f0"]!.dim(1))
        func iou(_ pred: MLXArray, _ goldKey: String) -> Float {
            let p = (pred .> 0).asType(.float32), gg = (fx[goldKey]! .> 0).asType(.float32)
            let inter = (p * gg).sum().item(Float.self)
            let uni = ((p + gg) .> 0).asType(.float32).sum().item(Float.self)
            return uni > 0 ? inter / uni : 1
        }
        MLX.GPU.resetPeakMemory()
        let start = Date()
        var minIoU: Float = 1
        m.propagate(frameCount: n, clickFrame: 0, points: [[210, 350]], labels: [1],
                    origH: origH, origW: origW,
                    frame: { frames[($0 % 5) ..< ($0 % 5 + 1)] },          // cycle 5 frames, on demand
                    emit: { idx, logits, _ in
                        if idx < 5 { minIoU = min(minIoU, iou(logits, "mask_f\(idx)")) }
                    })
        let secs = Date().timeIntervalSince(start)
        let ok = minIoU > 0.90
        print(String(format: "[vid-measure] %d frames (fp32, GPU, streaming)  peak=%.3f GB  min_IoU(f0..4)=%.4f  (%.1fs)  %@",
                     n, Double(MLX.GPU.peakMemory) / 1e9, minIoU, secs, ok ? "OK ✅" : "FAIL ❌"))
        if !ok { throw ExitCode(1) }
    }
}
