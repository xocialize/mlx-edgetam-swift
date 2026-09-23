import Foundation
import MLX
import MLXFast

/// SAM 2.1 image backbone: the Hiera trunk (hieradet.py), transcribed for the shared SAM2-family core
/// (AB-T-0173). `EdgeTAMModel` runs it instead of RepViT when the weights carry
/// `image_encoder.trunk.patch_embed.proj.weight`; the FpnNeck (same keys/code as EdgeTAM's, scalp 1),
/// prompt encoder, mask decoder and predictor are shared unchanged. Image mode only — SAM 2.1's video memory
/// stack (RoPE memory attention, no perceiver) is not ported.
public struct HieraSpec: Sendable, Equatable {
    public let name: String
    public let embedDim: Int, numHeads: Int
    public let stages: [Int], globalAttBlocks: [Int], windowSpec: [Int]

    /// The four SAM 2.1 configs (sam2/configs/sam2.1/*.yaml + Hiera defaults), keyed by (embed dim, depth).
    public static let all: [HieraSpec] = [
        HieraSpec(name: "tiny", embedDim: 96, numHeads: 1, stages: [1, 2, 7, 2], globalAttBlocks: [5, 7, 9],
                  windowSpec: [8, 4, 14, 7]),
        HieraSpec(name: "small", embedDim: 96, numHeads: 1, stages: [1, 2, 11, 2], globalAttBlocks: [7, 10, 13],
                  windowSpec: [8, 4, 14, 7]),
        HieraSpec(name: "base_plus", embedDim: 112, numHeads: 2, stages: [2, 3, 16, 3],
                  globalAttBlocks: [12, 16, 20], windowSpec: [8, 4, 14, 7]),
        HieraSpec(name: "large", embedDim: 144, numHeads: 2, stages: [2, 6, 36, 4],
                  globalAttBlocks: [23, 33, 43], windowSpec: [8, 4, 16, 8]),
    ]

    /// Identify the variant from the weights (patch-embed width + block count); nil = not a Hiera checkpoint.
    public static func detect(_ w: [String: MLXArray]) -> HieraSpec? {
        guard let pe = w["image_encoder.trunk.patch_embed.proj.weight"] else { return nil }
        var depth = 0
        while w["image_encoder.trunk.blocks.\(depth).norm1.weight"] != nil { depth += 1 }
        return all.first { $0.embedDim == pe.dim(0) && $0.stages.reduce(0, +) == depth }
    }
}

extension EdgeTAMModel {
    private static let hp = "image_encoder.trunk"

    /// Hiera forward → the four stage-end features, strides 4/8/16/32, NHWC (same contract as RepViT `trunk`).
    func hieraTrunk(_ input: MLXArray, _ spec: HieraSpec) -> [MLXArray] {
        let p = Self.hp
        var x = conv(input, "\(p).patch_embed.proj.weight", b: "\(p).patch_embed.proj.bias", stride: 4, pad: 3)
        x = x + hieraPosEmbed(x.dim(1), x.dim(2))
        let depth = spec.stages.reduce(0, +)
        let stageEnds = (1 ... spec.stages.count).map { spec.stages[..<$0].reduce(0, +) - 1 }
        let qPoolBlocks = Set(stageEnds.dropLast().map { $0 + 1 })              // q_pool = 3 stages
        var dim = spec.embedDim, heads = spec.numHeads, stage = 1
        var outputs: [MLXArray] = []
        for i in 0 ..< depth {
            var dimOut = dim
            var window = spec.windowSpec[stage - 1]                              // the PREVIOUS stage's window
            if spec.globalAttBlocks.contains(i) { window = 0 }
            if stageEnds.contains(i - 1) { dimOut = dim * 2; heads *= 2; stage += 1 }
            x = hieraBlock(x, "\(p).blocks.\(i)", dim: dim, dimOut: dimOut, heads: heads,
                           qPool: qPoolBlocks.contains(i), window: window)
            dim = dimOut
            if stageEnds.contains(i) { outputs.append(x) }
        }
        return outputs
    }

    /// `_get_pos_embed`: bicubic(pos_embed → h×w) + pos_embed_window tiled. Constant for a fixed input size;
    /// built once per (h,w) and cached.
    func hieraPosEmbed(_ h: Int, _ wd: Int) -> MLXArray {
        let key = "\(h)x\(wd)"
        if let c = cache.value(key) { return c }
        let p = Self.hp
        let bg = a("\(p).pos_embed")                                              // (1,bh,bw,C) NHWC
        let win = a("\(p).pos_embed_window")                                      // (1,ws,ws,C)
        let (bh, bw, C) = (bg.dim(1), bg.dim(2), bg.dim(3))
        let wh = Self.bicubicWeights(inSize: bh, outSize: h), ww = Self.bicubicWeights(inSize: bw, outSize: wd)
        var y = MLX.matmul(wh, bg.asType(.float32).reshaped([bh, bw * C]))       // (h, bw·C)
        y = y.reshaped([h, bw, C]).transposed(0, 2, 1)                             // (h, C, bw)
        y = MLX.matmul(y, ww.transposed()).transposed(0, 2, 1)                     // (h, wd, C)
        let ws = win.dim(1)
        let tiled = MLX.broadcast(win.asType(.float32).reshaped([1, 1, ws, 1, ws, C]),
                                  to: [1, h / ws, ws, wd / ws, ws, C]).reshaped([1, h, wd, C])
        let pe = (y.reshaped([1, h, wd, C]) + tiled).asType(bg.dtype)
        pe.eval()
        cache.set(key, pe)
        return pe
    }

    /// torch `F.interpolate(mode="bicubic", align_corners=False)` as a dense `(out,in)` matrix: cubic convolution
    /// A = −0.75, source index `(i+0.5)·in/out − 0.5` (not clamped), tap indices clamped to the border.
    static func bicubicWeights(inSize: Int, outSize: Int) -> MLXArray {
        let A = -0.75
        func c1(_ x: Double) -> Double { ((A + 2) * x - (A + 3)) * x * x + 1 }
        func c2(_ x: Double) -> Double { ((A * x - 5 * A) * x + 8 * A) * x - 4 * A }
        let scale = Double(inSize) / Double(outSize)
        var m = [Float](repeating: 0, count: outSize * inSize)
        for i in 0 ..< outSize {
            let real = scale * (Double(i) + 0.5) - 0.5
            let i0 = Int(real.rounded(.down)), t = real - Double(i0)
            let ws = [c2(t + 1), c1(t), c1(1 - t), c2(2 - t)]
            for (k, wk) in ws.enumerated() {
                let j = min(max(i0 - 1 + k, 0), inSize - 1)
                m[i * inSize + j] += Float(wk)
            }
        }
        return MLXArray(m, [outSize, inSize])
    }

    /// MultiScaleBlock: norm1 → (proj+pool shortcut on a dim change) → windowed / global MultiScaleAttention with
    /// optional q max-pool → unpartition → residual → MLP(GELU) residual. LayerNorm eps 1e-6.
    private func hieraBlock(_ x0: MLXArray, _ p: String, dim: Int, dimOut: Int, heads: Int,
                            qPool: Bool, window: Int) -> MLXArray {
        var shortcut = x0
        let xn = ln(x0, "\(p).norm1", eps: 1e-6)
        if dim != dimOut {
            let pr = lin(xn, "\(p).proj")
            shortcut = qPool ? Self.maxPool2(pr) : pr
        }
        var (H, W) = (xn.dim(1), xn.dim(2))
        var y = xn
        var padHW = (H, W)
        if window > 0 { (y, padHW) = Self.windowPartition(xn, window) }
        y = hieraAttention(y, "\(p).attn", dimOut: dimOut, heads: heads, qPool: qPool)
        var unWindow = window
        if qPool && window > 0 {
            unWindow = window / 2
            (H, W) = (shortcut.dim(1), shortcut.dim(2))
            padHW = (H + (unWindow - H % unWindow) % unWindow, W + (unWindow - W % unWindow) % unWindow)
        }
        if window > 0 { y = Self.windowUnpartition(y, unWindow, padHW, (H, W)) }
        let x = shortcut + y
        let m = lin(gelu(lin(ln(x, "\(p).norm2", eps: 1e-6), "\(p).mlp.layers.0")), "\(p).mlp.layers.1")
        return x + m
    }

    private func hieraAttention(_ x: MLXArray, _ p: String, dimOut: Int, heads: Int, qPool: Bool) -> MLXArray {
        let (B, H, W) = (x.dim(0), x.dim(1), x.dim(2))
        let hd = dimOut / heads
        let qkv = lin(x, "\(p).qkv").reshaped([B, H * W, 3, heads, hd])
        var q = qkv[0..., 0..., 0], k = qkv[0..., 0..., 1], v = qkv[0..., 0..., 2]   // (B,N,heads,hd)
        var (h, w) = (H, W)
        if qPool {
            q = Self.maxPool2(q.reshaped([B, H, W, dimOut]))
            (h, w) = (q.dim(1), q.dim(2))
            q = q.reshaped([B, h * w, heads, hd])
        }
        let o = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
            scale: 1 / Float(hd).squareRoot(), mask: nil)
        return lin(o.transposed(0, 2, 1, 3).reshaped([B, h, w, dimOut]), "\(p).proj")
    }

    /// MaxPool2d(kernel 2, stride 2) on NHWC (even H, W — every Hiera pooling site).
    static func maxPool2(_ x: MLXArray) -> MLXArray {
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped([B, H / 2, 2, W / 2, 2, C]).max(axes: [2, 4])
    }

    /// Zero-pad to a multiple of `ws` (bottom/right, like F.pad) and split into `(B·nW, ws, ws, C)` windows.
    static func windowPartition(_ x0: MLXArray, _ ws: Int) -> (MLXArray, (Int, Int)) {
        var x = x0
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let ph = (ws - H % ws) % ws, pw = (ws - W % ws) % ws
        if ph > 0 || pw > 0 { x = MLX.padded(x, widths: [0, .init((0, ph)), .init((0, pw)), 0]) }
        let (Hp, Wp) = (H + ph, W + pw)
        let win = x.reshaped([B, Hp / ws, ws, Wp / ws, ws, C]).transposed(0, 1, 3, 2, 4, 5)
            .reshaped([-1, ws, ws, C])
        return (win, (Hp, Wp))
    }

    static func windowUnpartition(_ win: MLXArray, _ ws: Int, _ pad: (Int, Int), _ hw: (Int, Int)) -> MLXArray {
        let (Hp, Wp) = pad, (H, W) = hw
        let C = win.dim(3)
        let B = win.dim(0) / (Hp * Wp / ws / ws)
        var x = win.reshaped([B, Hp / ws, Wp / ws, ws, ws, C]).transposed(0, 1, 3, 2, 4, 5).reshaped([B, Hp, Wp, C])
        if Hp > H || Wp > W { x = x[0..., 0 ..< H, 0 ..< W, 0...] }
        return x
    }
}

/// Tiny thread-safe cache for per-model constants (the Hiera positional embedding).
final class ConstantCache: @unchecked Sendable {
    private var d: [String: MLXArray] = [:]
    private let lock = NSLock()
    func value(_ k: String) -> MLXArray? { lock.lock(); defer { lock.unlock() }; return d[k] }
    func set(_ k: String, _ v: MLXArray) { lock.lock(); d[k] = v; lock.unlock() }
}
