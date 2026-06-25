import Foundation
import CoreGraphics
import MLX

/// Image I/O + resize for the EdgeTAM predictor. RGB `[0,1]` NHWC; bilinear matches torch
/// `F.interpolate(mode="bilinear", align_corners=False)` (the SAM2 pre/post resize).
public enum EdgeTAMImage {

    public static func rgb(from cg: CGImage, width W: Int, height H: Int) -> MLXArray {
        let cs = CGColorSpaceCreateDeviceRGB()
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
