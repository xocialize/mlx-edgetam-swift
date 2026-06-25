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
    /// `coordsPx==nil` → empty prompt (2 not-a-point tokens). Always multimask (best-by-IoU of 3).
    /// → (lowResBest `(1,1,256,256)`, objPtr `(1,256)`, objScore Float).
    public func forwardSamHeads(_ backbone: MLXArray, _ featS0: MLXArray, _ featS1: MLXArray,
                         coordsPx: MLXArray?, labels: [Int]?) -> (MLXArray, MLXArray, Float) {
        let D = "sam_mask_decoder"
        let (sparse, dense): (MLXArray, MLXArray) = {
            if let coordsPx, let labels { return embedPrompt(coordsPx, labels) }
            return embedPrompt(MLXArray.zeros([1, 2]), [-1])            // → 2 not-a-point tokens
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
        // multimask: drop index 0 → 3 candidates; object-score hard-gate (NO_OBJ_SCORE −1024)
        let scoreF = objScore.item(Float.self)
        let isObj = scoreF > 0
        var masksM = masks[0..., 1 ..< 4, 0..., 0...]                   // (1,3,256,256)
        if !isObj { masksM = MLX.zeros(like: masksM) - 1024.0 }
        let iouM = iou[0..., 1 ..< 4]
        let best = MLX.argMax(iouM[0], axis: -1).item(Int.self)
        let lowResBest = masksM[0..., best ..< (best + 1), 0..., 0...]  // (1,1,256,256)
        // obj_ptr from best multimask token → 3-MLP → fixed_no_obj_ptr gate
        let samTok = maskToks[0..., 1 ..< 4, 0...][0..., best, 0...]    // (1,256)
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

    // MARK: - Propagate (single object, click on `clickFrame`)
    // ENHANCEMENT(v2): SINGLE OBJECT + PRE-STACKED frames. Multi-object → add a batched object axis (one
    // memory bank per object, shared per-frame encode). Long clips → process frame-by-frame instead of
    // taking the whole `(T,…)` stack + accumulating all masks (stream out + `clear_cache` per step). See
    // the EnhancementBlock at `EdgeTAMPackage.runTrack`.
    /// `frames` NHWC `(T,1024,1024,3)` (ImageNet-normalized). `points` are `[x,y]` in original-video px
    /// (foreground/background per `labels`: 1/0), prompting on `clickFrame`. Returns per-frame mask logits
    /// `(origH,origW)`.
    public func propagate(frames: MLXArray, clickFrame: Int, points: [[Float]], labels: [Int],
                          origH: Int, origW: Int) -> (masks: [MLXArray], scores: [Float]) {
        let T = frames.dim(0)
        let noMem = a("no_mem_embed").reshaped([1, 1, 1, 256])
        let currPos = Self.posSine(256, 64, 64).reshaped([4096, 1, 256])
        let maskmemPos = Self.posSine(64, 64, 64).reshaped([1, 64, 64, 64])
        var cond: [FrameMemory] = []
        var noncond: [Int: FrameMemory] = [:]
        var masks: [MLXArray] = []
        var scores: [Float] = []

        for t in 0 ..< T {
            let input = frames[t ..< (t + 1)]                           // (1,1024,1024,3)
            let (imageEmbed, fpn0, fpn1) = encode(input)               // imageEmbed = out2 + no_mem
            let rawFeat = imageEmbed - noMem                            // (1,64,64,256) memory-encoder feat
            let featS0 = conv(fpn0, "sam_mask_decoder.conv_s0.weight", b: "sam_mask_decoder.conv_s0.bias")
            let featS1 = conv(fpn1, "sam_mask_decoder.conv_s1.weight", b: "sam_mask_decoder.conv_s1.bias")

            let isInit = (t == clickFrame)
            let backbone: MLXArray
            if isInit {
                backbone = imageEmbed                                   // directly_add_no_mem_embed
            } else {
                let (memory, memoryPos, numSpatial, numObjPtr) =
                    assembleMemory(t, cond, noncond, numFrames: T)
                let currSeq = rawFeat.reshaped([1, 4096, 256]).transposed(1, 0, 2)  // (4096,1,256)
                let maOut = memoryAttention(currSeq, currPos, memory, memoryPos,
                                            numObjPtr: numObjPtr, numSpatial: numSpatial)
                backbone = maOut.reshaped([1, 64, 64, 256])
            }
            // Clicks are in original video px → normalize by (W,H) then scale to the 1024 model space.
            let scaled = points.flatMap { [$0[0] / Float(origW) * 1024, $0[1] / Float(origH) * 1024] }
            let clickPx = MLXArray(scaled, [points.count, 2])
            let (lowResBest, objPtr, objScore) = forwardSamHeads(
                backbone, featS0, featS1,
                coordsPx: isInit ? clickPx : nil,
                labels: isInit ? labels : nil)

            // Encode new memory: high-res (1024) mask → sigmoid·20−10 → mem-encoder → perceiver compress.
            let lrNHWC = lowResBest.transposed(0, 2, 3, 1)              // (1,256,256,1)
            let hiRes = EdgeTAMImage.bilinear(lrNHWC, outH: 1024, outW: 1024)
            let maskForMem = MLX.sigmoid(hiRes) * 20.0 - 10.0
            let maskmem = memoryEncoder(rawFeat, maskForMem)           // (1,64,64,64); no_obj_embed_spatial absent
            let (spatial, spatialPos) = perceiver(maskmem, maskmemPos) // (1,512,64),(1,512,64)
            let fm = FrameMemory(spatial: spatial[0], spatialPos: spatialPos[0], objPtr: objPtr[0],
                                 frameIdx: t, isCond: isInit)
            if isInit { cond.append(fm) } else { noncond[t] = fm }

            // Output mask at original video resolution.
            let full = EdgeTAMImage.bilinear(lrNHWC, outH: origH, outW: origW)[0, 0..., 0..., 0]
            full.eval()
            masks.append(full)
            scores.append(objScore)
        }
        return (masks, scores)
    }
}
