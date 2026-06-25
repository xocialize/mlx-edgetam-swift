import Foundation
import MLX

/// EdgeTAM image-mode forward (RepViT-M1 encoder + FpnNeck + SAM prompt encoder + mask decoder) over a
/// flat NHWC weights dict (from `oracle/convert.py`). Functional style, transcribed 1:1 from the
/// parity-verified `oracle/mlx_encoder.py` + `oracle/mlx_decoder.py` (image_embed 9.7e-6, masks 8.2e-5).
public final class EdgeTAMModel: @unchecked Sendable {

    let w: [String: MLXArray]
    public init(weights: [String: MLXArray]) { self.w = weights }

    func a(_ k: String) -> MLXArray { w[k]! }                          // convs already NHWC from convert
    func has(_ k: String) -> Bool { w[k] != nil }
    private static let depths = [2, 2, 14, 2]

    // MARK: primitives (shared with the P2 video extension — see EdgeTAMVideo.swift)
    func conv(_ x: MLXArray, _ k: String, b: String? = nil, stride: Int = 1, pad: Int = 0, groups: Int = 1) -> MLXArray {
        let y = MLX.conv2d(x, a(k), stride: .init(stride), padding: .init(pad), groups: groups)
        return b == nil ? y : y + a(b!)
    }
    private func bnEval(_ x: MLXArray, _ p: String) -> MLXArray {      // ConvNorm BN (.bn)
        (x - a(p + ".running_mean")) / MLX.sqrt(a(p + ".running_var") + 1e-5) * a(p + ".weight") + a(p + ".bias")
    }
    private func cn(_ x: MLXArray, _ p: String, stride: Int = 1, pad: Int = 0, groups: Int = 1) -> MLXArray {
        bnEval(conv(x, p + ".c.weight", stride: stride, pad: pad, groups: groups), p + ".bn")
    }
    func ln(_ x: MLXArray, _ p: String, eps: Float = 1e-5) -> MLXArray {
        let u = x.mean(axis: -1, keepDims: true); let d = x - u
        return d / MLX.sqrt((d * d).mean(axis: -1, keepDims: true) + eps) * a(p + ".weight") + a(p + ".bias")
    }
    func gelu(_ x: MLXArray) -> MLXArray { 0.5 * x * (1 + MLX.erf(x / 1.4142135623730951)) }
    func relu(_ x: MLXArray) -> MLXArray { MLX.maximum(x, 0) }
    func lin(_ x: MLXArray, _ p: String) -> MLXArray { MLX.matmul(x, a(p + ".weight").transposed()) + a(p + ".bias") }
    func linNB(_ x: MLXArray, _ p: String) -> MLXArray { MLX.matmul(x, a(p + ".weight").transposed()) }  // bias-free

    // MARK: RepViT encoder
    private func se(_ x: MLXArray, _ p: String) -> MLXArray {
        var g = x.mean(axes: [1, 2], keepDims: true)
        g = relu(conv(g, p + ".fc1.weight", b: p + ".fc1.bias"))
        g = conv(g, p + ".fc2.weight", b: p + ".fc2.bias")
        return x * MLX.sigmoid(g)
    }
    private func mlpConv(_ x: MLXArray, _ p: String) -> MLXArray { cn(gelu(cn(x, p + ".conv1")), p + ".conv2") }
    private func repvggDw(_ x: MLXArray, _ p: String) -> MLXArray {
        let c = x.dim(-1)
        return cn(x, p + ".conv", pad: 1, groups: c) + cn(x, p + ".conv1", groups: c) + x
    }
    private func block(_ x0: MLXArray, _ p: String) -> MLXArray {
        var x = repvggDw(x0, p + ".token_mixer")
        if w[p + ".se.fc1.weight"] != nil { x = se(x, p + ".se") }
        return x + mlpConv(x, p + ".channel_mixer")
    }
    private func downsample(_ x0: MLXArray, _ p: String) -> MLXArray {
        var x = block(x0, p + ".pre_block")
        x = cn(x, p + ".spatial_downsample", stride: 2, pad: 1, groups: x.dim(-1))
        x = cn(x, p + ".channel_downsample")
        return x + mlpConv(x, p + ".ffn")
    }
    private func trunk(_ input: MLXArray) -> [MLXArray] {
        let t = "image_encoder.trunk.body"
        var x = cn(input, "\(t).stem.conv1", stride: 2, pad: 1)
        x = gelu(x)
        x = cn(x, "\(t).stem.conv2", stride: 2, pad: 1)
        var feats: [MLXArray] = []
        for s in 0 ..< 4 {
            let sp = "\(t).stages_\(s)"
            if w["\(sp).downsample.spatial_downsample.c.weight"] != nil { x = downsample(x, "\(sp).downsample") }
            for b in 0 ..< Self.depths[s] { x = block(x, "\(sp).blocks.\(b)") }
            feats.append(x)
        }
        return feats
    }
    private func fpn(_ xs: [MLXArray]) -> [MLXArray] {
        let n = "image_encoder.neck.convs"
        var out = [MLXArray?](repeating: nil, count: 4)
        var prev: MLXArray? = nil
        for i in stride(from: 3, through: 0, by: -1) {
            var lat = conv(xs[i], "\(n).\(3 - i).conv.weight", b: "\(n).\(3 - i).conv.bias")
            if (i == 2 || i == 3), let p = prev {
                let (B, H, W, C) = (lat.dim(0), lat.dim(1), lat.dim(2), lat.dim(3))
                let td = MLX.broadcast(p.reshaped([B, p.dim(1), 1, p.dim(2), 1, C]),
                                       to: [B, p.dim(1), 2, p.dim(2), 2, C]).reshaped([B, H, W, C])
                lat = lat + td
            }
            prev = lat; out[i] = lat
        }
        return out.map { $0! }
    }

    /// Encoder → image_embed `(1,64,64,256)` (FPN out[2] + no_mem_embed).
    public func encode(_ input: MLXArray) -> (imageEmbed: MLXArray, fpn0: MLXArray, fpn1: MLXArray) {
        let out = fpn(trunk(input))
        let emb = out[2] + a("no_mem_embed").reshaped([1, 1, 1, 256])
        return (emb, out[0], out[1])
    }

    // MARK: SAM positional encoding (random Fourier)
    private func peEncoding(_ coords: MLXArray) -> MLXArray {          // coords (...,2) in [0,1]
        var c = MLX.matmul(2 * coords - 1, a("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"))
        c = 2 * Float.pi * c
        return MLX.concatenated([MLX.sin(c), MLX.cos(c)], axis: -1)
    }
    func densePE(_ h: Int = 64, _ wd: Int = 64) -> MLXArray {          // (1,h,w,256)
        var coords = [Float](repeating: 0, count: h * wd * 2)
        for y in 0 ..< h { for x in 0 ..< wd {
            coords[(y * wd + x) * 2 + 0] = (Float(x) + 0.5) / Float(wd)
            coords[(y * wd + x) * 2 + 1] = (Float(y) + 0.5) / Float(h)
        } }
        return peEncoding(MLXArray(coords, [h * wd, 2])).reshaped([1, h, wd, 256])
    }

    // MARK: prompt encoder
    func embedPrompt(_ coordsPx: MLXArray, _ labels: [Int]) -> (sparse: MLXArray, dense: MLXArray) {
        let pts = MLX.concatenated([coordsPx + 0.5, MLXArray([0.0, 0.0] as [Float], [1, 2])], axis: 0)
        let labelsF = labels + [-1]
        let pe = peEncoding(pts / 1024.0)                             // (N+1,256)
        var rows: [MLXArray] = []
        for (i, lb) in labelsF.enumerated() {
            if lb == -1 { rows.append(a("sam_prompt_encoder.not_a_point_embed.weight")[0]) }
            else { rows.append(pe[i] + a("sam_prompt_encoder.point_embeddings.\(lb).weight")[0]) }
        }
        let sparse = MLX.stacked(rows, axis: 0).reshaped([1, rows.count, 256])
        let dense = MLX.broadcast(a("sam_prompt_encoder.no_mask_embed.weight").reshaped([1, 1, 1, 256]), to: [1, 64, 64, 256])
        return (sparse, dense)
    }

    // MARK: two-way transformer
    private func attn(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ p: String, heads: Int = 8) -> MLXArray {
        let Q = lin(q, p + ".q_proj"), K = lin(k, p + ".k_proj"), V = lin(v, p + ".v_proj")
        let (B, Nq, C) = (Q.dim(0), Q.dim(1), Q.dim(2)); let Nk = K.dim(1); let hd = C / heads
        let qh = Q.reshaped([B, Nq, heads, hd]).transposed(0, 2, 1, 3)
        let kh = K.reshaped([B, Nk, heads, hd]).transposed(0, 2, 1, 3)
        let vh = V.reshaped([B, Nk, heads, hd]).transposed(0, 2, 1, 3)
        let s = MLX.softmax(MLX.matmul(qh, kh.transposed(0, 1, 3, 2)) / Float(hd).squareRoot(), axis: -1)
        let o = MLX.matmul(s, vh).transposed(0, 2, 1, 3).reshaped([B, Nq, C])
        return lin(o, p + ".out_proj")
    }
    private func twoWay(_ q0: MLXArray, _ k0: MLXArray, _ qpe: MLXArray, _ kpe: MLXArray, _ p: String, skipPe: Bool) -> (MLXArray, MLXArray) {
        var q = q0, k = k0
        if skipPe { q = attn(q, q, q, p + ".self_attn") }              // layer 0 REPLACES (no residual)
        else { let qq = q + qpe; q = q + attn(qq, qq, q, p + ".self_attn") }
        q = ln(q, p + ".norm1")
        q = ln(q + attn(q + qpe, k + kpe, k, p + ".cross_attn_token_to_image"), p + ".norm2")
        q = ln(q + lin(relu(lin(q, p + ".mlp.layers.0")), p + ".mlp.layers.1"), p + ".norm3")
        k = ln(k + attn(k + kpe, q + qpe, q, p + ".cross_attn_image_to_token"), p + ".norm4")
        return (q, k)
    }
    func transformer(_ imageEmbed: MLXArray, _ imagePe: MLXArray, _ tokens: MLXArray) -> (MLXArray, MLXArray) {
        var keys = imageEmbed.reshaped([1, 64 * 64, 256])
        let kpe = imagePe.reshaped([1, 64 * 64, 256])
        var q = tokens
        let T = "sam_mask_decoder.transformer"
        for i in 0 ..< 2 { (q, keys) = twoWay(q, keys, tokens, kpe, "\(T).layers.\(i)", skipPe: i == 0) }
        q = ln(q + attn(q + tokens, keys + kpe, keys, "\(T).final_attn_token_to_image"), "\(T).norm_final_attn")
        return (q, keys)
    }

    // MARK: mask decoder
    func mlpHead(_ x0: MLXArray, _ p: String, _ n: Int) -> MLXArray {
        var x = x0
        for i in 0 ..< n { x = lin(x, "\(p).layers.\(i)"); if i < n - 1 { x = relu(x) } }
        return x
    }
    func convT(_ x: MLXArray, _ k: String, _ b: String) -> MLXArray {
        MLX.convTransposed2d(x, a(k), stride: 2, padding: 0) + a(b)
    }

    /// Full image-mode forward → (masks `(3,256,256)` raw logits, iou `(3,)`).
    public func segment(input: MLXArray, coordsPx: MLXArray, labels: [Int]) -> (masks: MLXArray, iou: MLXArray) {
        let (emb, fpn0, fpn1) = encode(input)
        let featS0 = conv(fpn0, "sam_mask_decoder.conv_s0.weight", b: "sam_mask_decoder.conv_s0.bias")
        let featS1 = conv(fpn1, "sam_mask_decoder.conv_s1.weight", b: "sam_mask_decoder.conv_s1.bias")
        let (sparse, dense) = embedPrompt(coordsPx, labels)
        let D = "sam_mask_decoder"
        let outTokens = MLX.concatenated([a(D + ".obj_score_token.weight"), a(D + ".iou_token.weight"),
                                          a(D + ".mask_tokens.weight")], axis: 0).reshaped([1, 6, 256])
        let tokens = MLX.concatenated([outTokens, sparse], axis: 1)
        var (q, keys) = transformer(emb + dense, densePE(), tokens)
        let iouTok = q[0..., 1, 0...]
        let maskToks = q[0..., 2 ..< 6, 0...]
        var u = convT(keys.reshaped([1, 64, 64, 256]), D + ".output_upscaling.0.weight", D + ".output_upscaling.0.bias") + featS1
        u = gelu(ln(u, D + ".output_upscaling.1", eps: 1e-6))
        u = gelu(convT(u, D + ".output_upscaling.3.weight", D + ".output_upscaling.3.bias") + featS0)  // (1,256,256,32)
        var hypers: [MLXArray] = []
        for i in 0 ..< 4 { hypers.append(mlpHead(maskToks[0..., i, 0...], "\(D).output_hypernetworks_mlps.\(i)", 3)) }
        let hyper = MLX.stacked(hypers, axis: 1)                       // (1,4,32)
        let (H, Wd) = (u.dim(1), u.dim(2))
        let masks = MLX.matmul(hyper, u.reshaped([1, H * Wd, 32]).transposed(0, 2, 1)).reshaped([1, 4, H, Wd])
        let iou = MLX.sigmoid(mlpHead(iouTok, D + ".iou_prediction_head", 3))
        _ = q
        return (masks[0, 1 ..< 4, 0..., 0...], iou[0, 1 ..< 4])        // multimask: drop index 0 → 3
    }
}
