import Foundation
import MLX

/// EdgeTAM video memory stack (P2): PerceiverResampler + MemoryEncoder + MemoryAttention (RoPE-2D) +
/// the SAM2 memory-bank state machine + masklet propagation. Transcribed 1:1 from the parity-verified
/// `oracle/mlx_{perceiver,mem_encoder,mem_attn,track_decode}.py` and `VIDEO-ORCHESTRATION.md`.
///
/// All position encodings are **content-independent** (sinusoidal over the feature grid) and generated
/// here — validated vs the `vid_*_pos` goldens to ~6e-8: curr_pos = sine(256ch, 64×64); perceiver-input
/// pos = sine(64ch, 64×64); perceiver-2D pos = sine(64ch, 16×16); spatial mem pos = perceiver pos +
/// `maskmem_tpos_enc[6−t_pos]`; obj-ptr pos = zeros (`add_tpos_enc_to_obj_ptrs=false`).
///
/// Config (edgetam.yaml): num_maskmem=7, mem_dim=64, hidden=256, feat 64×64, image 1024, stride=1,
/// directly_add_no_mem_embed, sigmoid_scale 20 / bias −10, multimask_output_for_tracking,
/// use_multimask_token_for_obj_ptr, fixed_no_obj_ptr, only_obj_ptrs_in_the_past.

/// One tracked object's prompt for the multi-object propagate/track path: a point set and/or a box, given
/// on `clickFrame`. Coordinates are in **original-video px** (the propagate/track layer normalizes to the
/// 1024² model space). Each object carries its own independent memory bank; the per-frame RepViT+FPN
/// encode is shared across all objects (SAM2's image features don't depend on the prompt).
public struct ObjectPrompt: Sendable {
    public let clickFrame: Int      // frame the prompt is given on (forward-only propagation from here)
    public let points: [[Float]]    // [x,y] in original-video px (may be empty if `box` is set)
    public let labels: [Int]        // per-point: 1 = foreground / 0 = background
    public let box: [Float]?        // optional [x0,y0,x1,y1] in original-video px
    public init(clickFrame: Int = 0, points: [[Float]] = [], labels: [Int] = [], box: [Float]? = nil) {
        self.clickFrame = clickFrame; self.points = points; self.labels = labels; self.box = box
    }
}

extension EdgeTAMModel {

    // MARK: - Stored per-frame memory (the memory bank entries)
    public struct FrameMemory {
        public let spatial: MLXArray     // (512,64) perceiver-compressed memory (seq, mem_dim)
        public let spatialPos: MLXArray  // (512,64) perceiver pos (zeros[256] ⊕ sine2d[256]); pre-tpos
        public let objPtr: MLXArray      // (256,) object pointer
        public let frameIdx: Int
        public let isCond: Bool
    }

    private static let numMaskmem = 7
    private static let memDim = 64
    private static let hidden = 256
    private static let maxObjPtrs = 16

    // MARK: - Sinusoidal position encoding (PositionEmbeddingSine, normalize=true, temp=10000)
    /// Returns `(H*W, numPosFeats)` row-major (y outer, x inner); first half = pos_y, second = pos_x.
    public static func posSine(_ numPosFeats: Int, _ H: Int, _ W: Int, temperature: Float = 10000) -> MLXArray {
        let npf = numPosFeats / 2
        let scale = 2 * Float.pi
        let eps: Float = 1e-6
        // dim_t[j] = temperature^(2*(j//2)/npf)
        var dimT = [Float](repeating: 0, count: npf)
        for j in 0 ..< npf { dimT[j] = powf(temperature, Float(2 * (j / 2)) / Float(npf)) }
        var out = [Float](repeating: 0, count: H * W * numPosFeats)
        for y in 0 ..< H {
            let yEmbed = Float(y + 1) / (Float(H) + eps) * scale
            for x in 0 ..< W {
                let xEmbed = Float(x + 1) / (Float(W) + eps) * scale
                let base = (y * W + x) * numPosFeats
                // pos_y first npf, then pos_x; even idx = sin, odd idx = cos (shared freq per pair)
                for k in 0 ..< (npf / 2) {
                    let fy = yEmbed / dimT[2 * k]
                    out[base + 2 * k] = sin(fy)
                    out[base + 2 * k + 1] = cos(yEmbed / dimT[2 * k + 1])
                    let fx = xEmbed / dimT[2 * k]
                    out[base + npf + 2 * k] = sin(fx)
                    out[base + npf + 2 * k + 1] = cos(xEmbed / dimT[2 * k + 1])
                }
            }
        }
        return MLXArray(out, [H * W, numPosFeats])
    }

    // MARK: - 2D axial RoPE tables (real arithmetic on adjacent pairs)
    /// `compute_axial_cis(dim, ex, ey)` → cos,sin each `(ex*ey, dim/2)`.
    public static func axialCosSin(_ dim: Int, _ ex: Int, _ ey: Int, theta: Float = 10000) -> (MLXArray, MLXArray) {
        let q = dim / 4
        var fr = [Float](repeating: 0, count: q)
        for i in 0 ..< q { fr[i] = 1.0 / powf(theta, Float(4 * i) / Float(dim)) }
        let n = ex * ey, half = dim / 2
        var cosA = [Float](repeating: 0, count: n * half)
        var sinA = [Float](repeating: 0, count: n * half)
        for t in 0 ..< n {
            let tx = Float(t % ex), ty = Float(t / ex)
            for i in 0 ..< q {
                let ax = tx * fr[i], ay = ty * fr[i]
                cosA[t * half + i] = cos(ax);     sinA[t * half + i] = sin(ax)
                cosA[t * half + q + i] = cos(ay); sinA[t * half + q + i] = sin(ay)
            }
        }
        return (MLXArray(cosA, [n, half]), MLXArray(sinA, [n, half]))
    }

    /// Rotate adjacent (even,odd) pairs of `x (1,N,C)` by cos/sin `(N,C/2)`.
    private func rope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        let (B, N, C) = (x.dim(0), x.dim(1), x.dim(2))
        let xp = x.reshaped([B, N, C / 2, 2])
        let xr = xp[0..., 0..., 0..., 0], xi = xp[0..., 0..., 0..., 1]
        let outr = xr * cos - xi * sin
        let outi = xr * sin + xi * cos
        return MLX.stacked([outr, outi], axis: -1).reshaped([B, N, C])
    }

    // MARK: - PerceiverResampler (spatial_perceiver) → 512 compressed latents + pos
    private func sdpaPerc(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray { // heads=1, dim 64
        let s = MLX.softmax(MLX.matmul(q, k.transposed(0, 2, 1)) * Float(0.125), axis: -1) // 64^-0.5
        return MLX.matmul(s, v)
    }
    private func percAttn(_ lat: MLXArray, _ x: MLXArray, _ p: String, _ pos: MLXArray?) -> MLXArray {
        let l = ln(lat, p + ".norm_latents"); let xx = ln(x, p + ".norm_x")
        let q = linNB(l, p + ".to_q")
        let kv = linNB(xx, p + ".to_kv")
        var k = kv[0..., 0..., 0 ..< 64], v = kv[0..., 0..., 64 ..< 128]
        if let pos { k = k + pos; v = v + pos }
        return linNB(sdpaPerc(q, k, v), p + ".to_out")
    }
    private func percSelfAttn(_ x: MLXArray, _ p: String) -> MLXArray {
        let xx = ln(x, p + ".norm")
        let q = linNB(xx, p + ".to_q"); let kv = linNB(xx, p + ".to_kv")
        let k = kv[0..., 0..., 0 ..< 64], v = kv[0..., 0..., 64 ..< 128]
        return linNB(sdpaPerc(q, k, v), p + ".to_out")
    }
    private func percFF(_ x: MLXArray, _ p: String) -> MLXArray { // LN(.0) + Lin(.1) + GELU + Lin(.3)
        linNB(gelu(linNB(ln(x, p + ".0"), p + ".1")), p + ".3")
    }
    private func percLayer(_ lat0: MLXArray, _ x: MLXArray, _ p: String, _ pos: MLXArray?) -> MLXArray {
        var lat = percAttn(lat0, x, p + ".attn", pos) + lat0
        lat = percFF(lat, p + ".ff") + lat
        lat = percSelfAttn(lat, p + ".self_attn") + lat
        lat = percFF(lat, p + ".self_ff") + lat
        return lat
    }

    /// `x` NHWC `(1,64,64,64)`, `pos` NHWC `(1,64,64,64)` → latents `(1,512,64)`, pos `(1,512,64)`.
    public func perceiver(_ x: MLXArray, _ pos: MLXArray) -> (MLXArray, MLXArray) {
        let SP = "spatial_perceiver"
        // --- 1D global: 256 latents cross-attend all 4096 positions (pos added to k&v) ---
        var lat1 = a("\(SP).latents").reshaped([1, 256, 64])
        let xf = x.reshaped([1, 64 * 64, 64])
        let pf = pos.reshaped([1, 64 * 64, 64])
        for i in 0 ..< 2 { lat1 = percLayer(lat1, xf, "\(SP).layers.\(i)", pf) }
        lat1 = ln(lat1, "\(SP).norm")
        let pos1 = MLXArray.zeros([1, 256, 64])
        // --- 2D windowed: 256 latents, 1 per 4×4 window of the 64×64 grid ---
        var lat2 = a("\(SP).latents_2d").reshaped([256, 1, 64])
        let nwin = 16, ws = 4
        let xw = x.reshaped([1, nwin, ws, nwin, ws, 64]).transposed(0, 1, 3, 2, 4, 5).reshaped([256, ws * ws, 64])
        for i in 0 ..< 2 { lat2 = percLayer(lat2, xw, "\(SP).layers.\(i)", nil) }
        lat2 = lat2.reshaped([1, nwin, nwin, 64]).reshaped([1, 256, 64])
        lat2 = ln(lat2, "\(SP).norm")
        let pos2 = Self.posSine(64, 16, 16).reshaped([1, 256, 64])
        return (MLX.concatenated([lat1, lat2], axis: 1), MLX.concatenated([pos1, pos2], axis: 1))
    }

    // MARK: - MemoryEncoder (mask + feat → 64-ch memory feature)
    private func cxblock(_ x: MLXArray, _ p: String) -> MLXArray { // ConvNeXt, dw-k7, NHWC
        var h = conv(x, p + ".dwconv.weight", b: p + ".dwconv.bias", pad: 3, groups: x.dim(-1))
        h = ln(h, p + ".norm", eps: 1e-6)
        h = lin(h, p + ".pwconv1"); h = gelu(h); h = lin(h, p + ".pwconv2")
        h = a(p + ".gamma") * h
        return x + h
    }
    /// `feat` NHWC `(1,64,64,256)` (raw top vision feat), `maskScaled` NHWC `(1,1024,1024,1)`
    /// (caller pre-applies `sigmoid*20−10`; encoder skips its own sigmoid) → `(1,64,64,64)`.
    public func memoryEncoder(_ feat: MLXArray, _ maskScaled: MLXArray) -> MLXArray {
        let ME = "memory_encoder", enc = "memory_encoder.mask_downsampler.encoder"
        var m = maskScaled
        for (ci, ni) in [(0, 1), (3, 4), (6, 7), (9, 10)] {
            m = conv(m, "\(enc).\(ci).weight", b: "\(enc).\(ci).bias", stride: 2, pad: 1)
            m = gelu(ln(m, "\(enc).\(ni)", eps: 1e-6))
        }
        m = conv(m, "\(enc).12.weight", b: "\(enc).12.bias")            // 1×1 → 256, (1,64,64,256)
        var x = conv(feat, "\(ME).pix_feat_proj.weight", b: "\(ME).pix_feat_proj.bias") + m
        for i in 0 ..< 2 { x = cxblock(x, "\(ME).fuser.layers.\(i)") }
        return conv(x, "\(ME).out_proj.weight", b: "\(ME).out_proj.bias")  // → (1,64,64,64)
    }

    // MARK: - MemoryAttention (2 layers, RoPE-2D, heads 1, internal 256)
    private func sdpaMA(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray { // heads=1, scale 256^-0.5
        let s = MLX.softmax(MLX.matmul(q, k.transposed(0, 2, 1)) / 16.0, axis: -1)
        return MLX.matmul(s, v)
    }
    private func maSelfAttn(_ tgt2: MLXArray, _ p: String, _ cq: MLXArray, _ sq: MLXArray) -> MLXArray {
        let q = rope(lin(tgt2, p + ".q_proj"), cq, sq)
        let k = rope(lin(tgt2, p + ".k_proj"), cq, sq)
        let v = lin(tgt2, p + ".v_proj")
        return lin(sdpaMA(q, k, v), p + ".out_proj")
    }
    private func maCrossAttn(_ qIn: MLXArray, _ kIn: MLXArray, _ vIn: MLXArray, _ p: String,
                             numObjPtr: Int, numSpatial: Int,
                             _ cq: MLXArray, _ sq: MLXArray, _ ck: MLXArray, _ sk: MLXArray) -> MLXArray {
        let q = rope(lin(qIn, p + ".q_proj"), cq, sq)
        let k = lin(kIn, p + ".k_proj")                                  // (1,M,256)
        let v = lin(vIn, p + ".v_proj")
        // v2 key split per spatial frame: [256 1D no-rope, 256 2D rope-16²]; obj-ptr tokens excluded.
        var parts: [MLXArray] = []
        for f in 0 ..< numSpatial {
            let base = f * 512
            parts.append(k[0..., base ..< (base + 256), 0...])
            parts.append(rope(k[0..., (base + 256) ..< (base + 512), 0...], ck, sk))
        }
        if numObjPtr > 0 { parts.append(k[0..., (numSpatial * 512)..., 0...]) }
        let kk = MLX.concatenated(parts, axis: 1)
        return lin(sdpaMA(q, kk, v), p + ".out_proj")
    }

    /// `curr/currPos` seq-first `(4096,1,256)`, `memory/memoryPos` seq-first `(M,1,64)` → `(1,4096,256)`.
    public func memoryAttention(_ curr: MLXArray, _ currPos: MLXArray, _ memory: MLXArray, _ memoryPos: MLXArray,
                         numObjPtr: Int, numSpatial: Int) -> MLXArray {
        let (cq, sq) = Self.axialCosSin(256, 64, 64)    // self + cross-q (4096,128)
        let (ck, sk) = Self.axialCosSin(256, 16, 16)    // cross 2D keys (256,128)
        var out = (curr + 0.1 * currPos).transposed(1, 0, 2)            // (1,4096,256)
        let mem = memory.transposed(1, 0, 2)                            // (1,M,64)
        let memK = mem + memoryPos.transposed(1, 0, 2)                  // keys = memory + pos
        for i in 0 ..< 2 {
            let p = "memory_attention.layers.\(i)"
            out = out + maSelfAttn(ln(out, p + ".norm1"), p + ".self_attn", cq, sq)
            out = out + maCrossAttn(ln(out, p + ".norm2"), memK, mem, p + ".cross_attn_image",
                                    numObjPtr: numObjPtr, numSpatial: numSpatial, cq, sq, ck, sk)
            out = out + lin(relu(lin(ln(out, p + ".norm3"), p + ".linear1")), p + ".linear2")
        }
        return ln(out, "memory_attention.norm")
    }

    // MARK: - Tracking SAM head (multimask + object-score gate + obj_ptr)
    /// `backbone` NHWC `(1,64,64,256)`, `featS0 (1,256,256,32)`, `featS1 (1,128,128,64)`.
    /// All prompts nil (`coordsPx`/`labels`/`boxPx`) → empty prompt (2 not-a-point tokens), the
    /// propagated-frame path. `boxPx` is `[x0,y0,x1,y1]` in model (1024²) space. `multimask` true → pick
    /// best-by-IoU of the 3 multi-masks (SAM2's `_use_multimask` on ≤1 prompt point / tracked frames);
    /// false → the single un-ambiguous mask (token 0), used on a box / multi-point prompt frame where SAM2
    /// exceeds `multimask_max_pt_num`. → (lowResBest `(1,1,256,256)`, objPtr `(1,256)`, objScore Float).
    public func forwardSamHeads(_ backbone: MLXArray, _ featS0: MLXArray, _ featS1: MLXArray,
                         coordsPx: MLXArray?, labels: [Int]?, boxPx: [Float]? = nil,
                         multimask: Bool = true) -> (MLXArray, MLXArray, Float) {
        let D = "sam_mask_decoder"
        let (sparse, dense): (MLXArray, MLXArray) = {
            if (coordsPx != nil && labels != nil) || boxPx != nil {
                return embedPrompt(coordsPx ?? MLXArray.zeros([0, 2]), labels ?? [], box: boxPx)
            }
            return embedPrompt(MLXArray.zeros([1, 2]), [-1])           // → 2 not-a-point tokens
        }()
        let outTokens = MLX.concatenated([a(D + ".obj_score_token.weight"), a(D + ".iou_token.weight"),
                                          a(D + ".mask_tokens.weight")], axis: 0).reshaped([1, 6, 256])
        let tokens = MLX.concatenated([outTokens, sparse], axis: 1)
        let (q, keys) = transformer(backbone + dense, densePE(), tokens)
        let objScore = mlpHead(q[0..., 0, 0...], D + ".pred_obj_score_head", 3)   // (1,1)
        let iouTok = q[0..., 1, 0...]
        let maskToks = q[0..., 2 ..< 6, 0...]
        var u = convT(keys.reshaped([1, 64, 64, 256]), D + ".output_upscaling.0.weight", D + ".output_upscaling.0.bias") + featS1
        u = gelu(ln(u, D + ".output_upscaling.1", eps: 1e-6))
        u = gelu(convT(u, D + ".output_upscaling.3.weight", D + ".output_upscaling.3.bias") + featS0)  // (1,256,256,32)
        var hypers: [MLXArray] = []
        for i in 0 ..< 4 { hypers.append(mlpHead(maskToks[0..., i, 0...], "\(D).output_hypernetworks_mlps.\(i)", 3)) }
        let hyper = MLX.stacked(hypers, axis: 1)                        // (1,4,32)
        let (H, Wd) = (u.dim(1), u.dim(2))
        let masks = MLX.matmul(hyper, u.reshaped([1, H * Wd, 32]).transposed(0, 2, 1)).reshaped([1, 4, H, Wd])
        let iou = MLX.sigmoid(mlpHead(iouTok, D + ".iou_prediction_head", 3))     // (1,4)
        // Token layout: index 0 = the single (un-ambiguous) mask, 1..3 = the 3 multi-masks. multimask →
        // best-by-IoU of 1..3; else the single mask at index 0. Object-score hard-gate (NO_OBJ_SCORE −1024).
        let scoreF = objScore.item(Float.self)
        let isObj = scoreF > 0
        let sel: Int = multimask ? 1 + MLX.argMax(iou[0..., 1 ..< 4][0], axis: -1).item(Int.self) : 0
        var lowResBest = masks[0..., sel ..< (sel + 1), 0..., 0...]     // (1,1,256,256)
        if !isObj { lowResBest = MLX.zeros(like: lowResBest) - 1024.0 }
        // obj_ptr from the selected mask token → 3-MLP → fixed_no_obj_ptr gate
        let samTok = maskToks[0..., sel, 0...]                          // (1,256)
        var objPtr = mlpHead(samTok, "obj_ptr_proj", 3)
        if !isObj { objPtr = a("no_obj_ptr") }                          // lam=is_obj (hard); obj_ptr = lam·ptr + (1−lam)·no_obj_ptr
        return (lowResBest, objPtr, scoreF)
    }

    // MARK: - Memory-bank assembly (the stateful core)
    /// Build `memory/memoryPos` seq-first `(M,1,64)` for frame `F` from stored cond/non-cond memories.
    private func assembleMemory(_ F: Int, _ cond: [FrameMemory], _ noncond: [Int: FrameMemory],
                                numFrames: Int) -> (MLXArray, MLXArray, Int, Int) {
        func tpos(_ idx: Int) -> MLXArray { a("maskmem_tpos_enc")[idx].reshaped([1, 64]) }  // (1,1,1,64)[idx]→(1,1,64)→use (1,64)
        var mem: [MLXArray] = [], pos: [MLXArray] = []
        // 1) spatial — conditioning frames first (t_pos=0), then prev frames t_pos∈[1..6]
        let condSorted = cond.sorted { $0.frameIdx < $1.frameIdx }
        for cf in condSorted {
            mem.append(cf.spatial)                                       // (512,64)
            pos.append(cf.spatialPos + tpos(Self.numMaskmem - 1))        // tpos[6−0]=tpos[6]
        }
        let condIdx = Set(cond.map { $0.frameIdx })
        for tPos in 1 ..< Self.numMaskmem {
            let tRel = Self.numMaskmem - tPos
            let prev = F - tRel                                          // stride=1: unifies t_rel==1 and ≥2
            if prev < 0 || condIdx.contains(prev) { continue }          // cond frames handled above
            guard let pf = noncond[prev] else { continue }
            mem.append(pf.spatial)
            pos.append(pf.spatialPos + tpos(Self.numMaskmem - tPos - 1)) // tpos[6−t_pos]
        }
        let numSpatial = mem.count
        // 2) object pointers — cond ptrs (only_obj_ptrs_in_the_past: t≤F) first, then non-cond past
        var ptrs: [MLXArray] = []
        for cf in condSorted where cf.frameIdx <= F { ptrs.append(cf.objPtr) }
        let maxPtrs = min(numFrames, Self.maxObjPtrs)
        for tDiff in 1 ..< maxPtrs {
            let t = F - tDiff
            if t < 0 { break }
            if let pf = noncond[t] { ptrs.append(pf.objPtr) }
        }
        for ptr in ptrs {                                               // split (256)→4 tokens of (64)
            mem.append(ptr.reshaped([4, 64]))
            pos.append(MLXArray.zeros([4, 64]))                         // add_tpos_enc_to_obj_ptrs=false
        }
        let numObjPtr = ptrs.count * 4
        let memory = MLX.concatenated(mem, axis: 0).reshaped([numSpatial * 512 + numObjPtr, 1, 64])
        let memoryPos = MLX.concatenated(pos, axis: 0).reshaped([numSpatial * 512 + numObjPtr, 1, 64])
        return (memory, memoryPos, numSpatial, numObjPtr)
    }

    // MARK: - Propagate (multi-object, streaming)
    // ENHANCEMENT(v2) status: MULTI-OBJECT + LONG-CLIP STREAMING both landed here. Each object carries its
    // own memory bank (cond/noncond `FrameMemory`); the per-frame RepViT+FPN encode is done ONCE and shared
    // across all objects (SAM2's image features are prompt-independent). Frames are pulled one at a time and
    // each (object,frame) mask is emitted (then released) as it lands, so the GPU working set stays flat
    // regardless of clip length OR object count. The per-object memory banks (~0.26 MB/frame each) are the
    // only thing that grows. Box prompts wire through `forwardSamHeads`/`embedPrompt`. See the
    // EnhancementBlock at `EdgeTAMPackage.runTrack`.

    /// **Streaming multi-object masklet propagation** (flat GPU footprint). Pulls one preprocessed frame at
    /// a time via `frame(t)` (NHWC `(1,1024,1024,3)`, ImageNet-normalized), shares the RepViT+FPN encode
    /// across every object, and hands each `(objIdx, frameIdx)` output mask logits `(origH,origW)` +
    /// object-score back through `emit` the instant it lands — nothing is pre-stacked and no per-frame
    /// output is retained here. Each `ObjectPrompt` has its own click/box (original-video px) and `clickFrame`.
    ///
    /// Propagation is forward-only: an object's track is empty (`-1024` score, zero mask) on frames before
    /// its `clickFrame`. Flat-footprint discipline (cf. the scail-2 long-run cache lesson): each step
    /// `eval()`s every object's output mask **and** new memory-bank entry, hands the masks out via `emit`,
    /// then a single `MLX.GPU.clearCache()` per frame drops the transient working set. Peak ≈ one frame's
    /// shared encode + O object forwards + the tiny per-object memory banks.
    public func propagate(frameCount T: Int, objects: [ObjectPrompt], origH: Int, origW: Int,
                          frame: (Int) -> MLXArray,
                          emit: (_ objIdx: Int, _ frameIdx: Int, _ mask: MLXArray, _ score: Float) throws -> Void) rethrows {
        let O = objects.count
        let noMem = a("no_mem_embed").reshaped([1, 1, 1, 256])
        let currPos = Self.posSine(256, 64, 64).reshaped([4096, 1, 256])
        let maskmemPos = Self.posSine(64, 64, 64).reshaped([1, 64, 64, 64])
        var cond = [[FrameMemory]](repeating: [], count: O)
        var noncond = [[Int: FrameMemory]](repeating: [:], count: O)

        // Normalize each object's prompt from original-video px → 1024² model space once.
        func scale(_ x: Float, _ orig: Int) -> Float { x / Float(orig) * 1024 }
        let clicks: [MLXArray?] = objects.map { obj in
            guard !obj.points.isEmpty else { return nil }
            return MLXArray(obj.points.flatMap { [scale($0[0], origW), scale($0[1], origH)] }, [obj.points.count, 2])
        }
        let boxes: [[Float]?] = objects.map { obj in
            guard let b = obj.box else { return nil }
            return [scale(b[0], origW), scale(b[1], origH), scale(b[2], origW), scale(b[3], origH)]
        }

        for t in 0 ..< T {
            // --- Shared per-frame encode (RepViT + FPN), done ONCE, reused by every object. ---
            let input = frame(t)                                        // (1,1024,1024,3), built on demand
            let (imageEmbed, fpn0, fpn1) = encode(input)               // imageEmbed = out2 + no_mem
            let rawFeat = imageEmbed - noMem                            // (1,64,64,256) memory-encoder feat
            let featS0 = conv(fpn0, "sam_mask_decoder.conv_s0.weight", b: "sam_mask_decoder.conv_s0.bias")
            let featS1 = conv(fpn1, "sam_mask_decoder.conv_s1.weight", b: "sam_mask_decoder.conv_s1.bias")
            let currSeq = rawFeat.reshaped([1, 4096, 256]).transposed(1, 0, 2)  // (4096,1,256) shared MA query

            for o in 0 ..< O {
                let obj = objects[o]
                let isInit = (t == obj.clickFrame)
                // Forward-only: before this object's prompt frame it is absent → emit an empty mask.
                if !isInit && cond[o].isEmpty && noncond[o].isEmpty {
                    let empty = MLXArray.zeros([origH, origW])
                    empty.eval()
                    try emit(o, t, empty, -1024)
                    continue
                }
                let backbone: MLXArray
                if isInit {
                    backbone = imageEmbed                               // directly_add_no_mem_embed
                } else {
                    let (memory, memoryPos, numSpatial, numObjPtr) =
                        assembleMemory(t, cond[o], noncond[o], numFrames: T)
                    let maOut = memoryAttention(currSeq, currPos, memory, memoryPos,
                                                numObjPtr: numObjPtr, numSpatial: numSpatial)
                    backbone = maOut.reshaped([1, 64, 64, 256])
                }
                // SAM2 `_use_multimask`: a box contributes 2 (corner) points → on a box / multi-point init
                // frame num_pts exceeds multimask_max_pt_num=1 → single mask; tracked frames always multimask.
                let numPts = obj.points.count + (obj.box != nil ? 2 : 0)
                let (lowResBest, objPtr, objScore) = forwardSamHeads(
                    backbone, featS0, featS1,
                    coordsPx: isInit ? clicks[o] : nil,
                    labels: isInit ? obj.labels : nil,
                    boxPx: isInit ? boxes[o] : nil,
                    multimask: isInit ? (numPts <= 1) : true)

                // Encode new memory: high-res (1024) mask → sigmoid·20−10 → mem-encoder → perceiver compress.
                let lrNHWC = lowResBest.transposed(0, 2, 3, 1)          // (1,256,256,1)
                let hiRes = EdgeTAMImage.bilinear(lrNHWC, outH: 1024, outW: 1024)
                let maskForMem = MLX.sigmoid(hiRes) * 20.0 - 10.0
                let maskmem = memoryEncoder(rawFeat, maskForMem)        // (1,64,64,64); no_obj_embed_spatial absent
                let (spatial, spatialPos) = perceiver(maskmem, maskmemPos) // (1,512,64),(1,512,64)
                let fm = FrameMemory(spatial: spatial[0], spatialPos: spatialPos[0], objPtr: objPtr[0],
                                     frameIdx: t, isCond: isInit)
                if isInit { cond[o].append(fm) } else { noncond[o][t] = fm }

                // Output mask at original video resolution. Materialize this object's output AND its
                // persisted memory-bank entry so the lazy graph does not chain across frames/objects.
                let full = EdgeTAMImage.bilinear(lrNHWC, outH: origH, outW: origW)[0, 0..., 0..., 0]
                MLX.eval(full, fm.spatial, fm.spatialPos, fm.objPtr)
                try emit(o, t, full, objScore)
            }
            MLX.GPU.clearCache()                                        // drop the frame's transient working set
        }
    }

    /// **Streaming single-object** convenience — one `ObjectPrompt`, `emit` without the object index.
    /// Keeps the existing `(clickFrame, points, labels[, box])` call site; delegates to the multi-object core.
    public func propagate(frameCount T: Int, clickFrame: Int, points: [[Float]], labels: [Int],
                          box: [Float]? = nil, origH: Int, origW: Int,
                          frame: (Int) -> MLXArray,
                          emit: (_ frameIdx: Int, _ mask: MLXArray, _ score: Float) throws -> Void) rethrows {
        try propagate(frameCount: T,
                      objects: [ObjectPrompt(clickFrame: clickFrame, points: points, labels: labels, box: box)],
                      origH: origH, origW: origW, frame: frame,
                      emit: { _, idx, m, s in try emit(idx, m, s) })
    }

    /// Back-compat stacked multi-object entry point (parity / VideoSmoke): pre-stacked `frames` NHWC
    /// `(T,1024,1024,3)` in, one collected `(masks,scores)` track per object out. Delegates to the streaming
    /// core — same numerics — but retains every output, so it does NOT have the flat footprint.
    public func propagate(frames: MLXArray, objects: [ObjectPrompt], origH: Int, origW: Int)
        -> [(masks: [MLXArray], scores: [Float])] {
        let O = objects.count
        var masks = [[MLXArray]](repeating: [], count: O)
        var scores = [[Float]](repeating: [], count: O)
        propagate(frameCount: frames.dim(0), objects: objects, origH: origH, origW: origW,
                  frame: { frames[$0 ..< ($0 + 1)] },
                  emit: { o, _, m, s in masks[o].append(m); scores[o].append(s) })
        return (0 ..< O).map { (masks[$0], scores[$0]) }
    }

    /// Back-compat stacked single-object entry point. Delegates to the multi-object stacked form.
    public func propagate(frames: MLXArray, clickFrame: Int, points: [[Float]], labels: [Int],
                          box: [Float]? = nil, origH: Int, origW: Int) -> (masks: [MLXArray], scores: [Float]) {
        propagate(frames: frames,
                  objects: [ObjectPrompt(clickFrame: clickFrame, points: points, labels: labels, box: box)],
                  origH: origH, origW: origW)[0]
    }
}
