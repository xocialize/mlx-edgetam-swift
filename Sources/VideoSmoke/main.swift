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

    func run() throws {
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

        if failed { throw ExitCode(1) }
        print("[vid-smoke] ALL P2 PARITY GATES PASSED ✅")
    }
}
