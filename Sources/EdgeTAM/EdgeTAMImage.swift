import Foundation
import CoreGraphics
import MLX

/// Image I/O + resize for the EdgeTAM predictor. RGB `[0,1]` NHWC; bilinear matches torch
/// `F.interpolate(mode="bilinear", align_corners=False)` (the SAM2 pre/post resize).
public enum EdgeTAMImage {

    /// Raw RGB bytes → `[0,1]`, like PIL `convert("RGB")`: drawn in the image's OWN RGB colour space so no
    /// colour matching changes the values (sRGB when the image has none or isn't RGB). Transparent pixels
    /// composite onto black (premultiplied draw).
    public static func rgb(from cg: CGImage, width W: Int, height H: Int) -> MLXArray {
        let cs = (cg.colorSpace?.model == .rgb ? cg.colorSpace : nil) ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var buf = [UInt8](repeating: 0, count: W * H * 4)
        buf.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                                bytesPerRow: W * 4, space: cs,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
        }
        var rgb = [Float](repeating: 0, count: W * H * 3)
        for p in 0 ..< W * H { for c in 0 ..< 3 { rgb[p * 3 + c] = Float(buf[p * 4 + c]) / 255 } }
        return MLXArray(rgb, [1, H, W, 3])
    }

    /// Separable resize of NHWC `x`. `antialias: true` = torch `F.interpolate(bilinear, align_corners=False,
    /// antialias=True)` — what torchvision's tensor `Resize` (SAM2Transforms, antialias default True) runs, a
    /// triangle filter widened by the scale factor when DOWNsampling; `false` = plain bilinear (`bilinear`).
    public static func resize(_ x: MLXArray, outH: Int, outW: Int, antialias: Bool) -> MLXArray {
        guard antialias else { return bilinear(x, outH: outH, outW: outW) }
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        precondition(B == 1, "resize: batch 1")
        var y = x
        if H != outH {                                           // (outH,H) @ (H, W·C)
            y = MLX.matmul(aaWeights(inSize: H, outSize: outH), y.reshaped([H, W * C])).reshaped([1, outH, W, C])
        }
        if W != outW {                                           // (outH·C, W) @ (W, outW)
            let t = y.transposed(0, 1, 3, 2).reshaped([outH * C, W])
            y = MLX.matmul(t, aaWeights(inSize: W, outSize: outW).transposed())
                .reshaped([1, outH, C, outW]).transposed(0, 1, 3, 2)
        }
        return y
    }

    /// PIL `Image.resize((1024,1024))` on an 8-bit RGB image (default BICUBIC, antialiased) — what upstream's
    /// VIDEO predictor runs on every frame (`sam2.utils.misc._load_img_as_tensor`). Horizontal pass first, each
    /// pass rounded + clipped to the uint8 grid, coefficients in PIL's 22-bit fixed point. `x` NHWC in [0,1]
    /// (exact /255 values); returns [0,1].
    public static func resizePILBicubic(_ x: MLXArray, outH: Int, outW: Int) -> MLXArray {
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        precondition(B == 1, "resizePILBicubic: batch 1")
        func grid(_ v: MLXArray) -> MLXArray { MLX.clip(MLX.floor(v + 0.5), min: 0, max: 255) }
        var y = MLX.round(x * 255)
        if W != outW {                                           // (H·C, W) @ (W, outW)
            let t = y.transposed(0, 1, 3, 2).reshaped([H * C, W])
            y = grid(MLX.matmul(t, kernelWeights(inSize: W, outSize: outW, filter: .bicubicPIL).transposed()))
                .reshaped([1, H, C, outW]).transposed(0, 1, 3, 2)
        }
        if H != outH {                                           // (outH,H) @ (H, outW·C)
            y = grid(MLX.matmul(kernelWeights(inSize: H, outSize: outH, filter: .bicubicPIL),
                                y.reshaped([H, outW * C]))).reshaped([1, outH, outW, C])
        }
        return y / 255
    }

    enum Filter { case bilinear, bicubicPIL }

    /// Dense `(out,in)` antialiased resampling weights — torch `_compute_indices_weights_aa` for `.bilinear`,
    /// PIL `precompute_coeffs` (+ its 22-bit fixed-point quantisation) for `.bicubicPIL`. Same scheme: the
    /// filter support widens by the scale factor when downsampling; rows renormalised.
    static func kernelWeights(inSize: Int, outSize: Int, filter: Filter) -> MLXArray {
        let (fsupport, f): (Double, (Double) -> Double) = {
            switch filter {
            case .bilinear: return (1, { max(0, 1 - abs($0)) })
            case .bicubicPIL: return (2, { v in                   // a = −0.5
                let a = -0.5, x = abs(v)
                if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
                return 0 })
            }
        }()
        let scale = Double(inSize) / Double(outSize)
        let fscale = max(scale, 1)
        let support = fsupport * fscale
        var m = [Float](repeating: 0, count: outSize * inSize)
        for i in 0 ..< outSize {
            let center = scale * (Double(i) + 0.5)
            let xmin = max(Int(center - support + 0.5), 0)          // Int() truncates toward zero, like C
            let xmax = min(Int(center + support + 0.5), inSize)
            var ws = [Double](), total = 0.0
            for j in xmin ..< max(xmin, xmax) { let w = f((Double(j) - center + 0.5) / fscale); ws.append(w); total += w }
            for (k, w) in ws.enumerated() where total != 0 {
                var v = w / total
                if filter == .bicubicPIL { v = (v * 4_194_304).rounded(.toNearestOrAwayFromZero) / 4_194_304 }  // 1<<22
                m[i * inSize + xmin + k] = Float(v)
            }
        }
        return MLXArray(m, [outSize, inSize])
    }
    static func aaWeights(inSize: Int, outSize: Int) -> MLXArray { kernelWeights(inSize: inSize, outSize: outSize, filter: .bilinear) }

    /// torch bilinear (align_corners=False), separable H then W. `x` NHWC.
    public static func bilinear(_ x: MLXArray, outH: Int, outW: Int) -> MLXArray {
        func resizeAxis(_ x: MLXArray, axis: Int, inSize: Int, outSize: Int) -> MLXArray {
            if inSize == outSize { return x }
            let scale = Float(inSize) / Float(outSize)
            var i0 = [Int32](repeating: 0, count: outSize), i1 = [Int32](repeating: 0, count: outSize)
            var w0 = [Float](repeating: 0, count: outSize), w1 = [Float](repeating: 0, count: outSize)
            for i in 0 ..< outSize {
                let src = max(0, (Float(i) + 0.5) * scale - 0.5)
                let s0 = Int(src.rounded(.down))
                let s0c = min(s0, inSize - 1), s1c = min(s0 + 1, inSize - 1)
                let f = src - Float(s0)
                i0[i] = Int32(s0c); i1[i] = Int32(s1c); w0[i] = 1 - f; w1[i] = f
            }
            let g0 = MLX.take(x, MLXArray(i0), axis: axis)
            let g1 = MLX.take(x, MLXArray(i1), axis: axis)
            let shape = axis == 1 ? [1, outSize, 1, 1] : [1, 1, outSize, 1]
            return g0 * MLXArray(w0, shape) + g1 * MLXArray(w1, shape)
        }
        let h = resizeAxis(x, axis: 1, inSize: x.dim(1), outSize: outH)
        return resizeAxis(h, axis: 2, inSize: x.dim(2), outSize: outW)
    }

    /// Soft matte `(H,W)` in [0,1] → opaque grey RGB CGImage (value·255, rounded).
    public static func matteCGImage(_ matte: MLXArray) -> CGImage {
        let H = matte.dim(0), W = matte.dim(1)
        let m = MLX.clip(matte.asType(.float32) * 255 + 0.5, min: 0, max: 255).asType(.uint8).asArray(UInt8.self)
        var buf = [UInt8](repeating: 255, count: W * H * 4)
        for p in 0 ..< W * H { buf[p * 4] = m[p]; buf[p * 4 + 1] = m[p]; buf[p * 4 + 2] = m[p] }
        let ctx = buf.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        }
        return ctx.makeImage()!
    }

    /// Boolean mask `(H,W)` → opaque RGB CGImage overlay (white = mask) for visual checks.
    public static func maskCGImage(_ mask: MLXArray) -> CGImage {
        let H = mask.dim(0), W = mask.dim(1)
        let m = mask.asArray(Float.self)
        var buf = [UInt8](repeating: 255, count: W * H * 4)
        for p in 0 ..< W * H { let v: UInt8 = m[p] > 0 ? 255 : 0; buf[p * 4] = v; buf[p * 4 + 1] = v; buf[p * 4 + 2] = v }
        let ctx = buf.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        }
        return ctx.makeImage()!
    }
}
