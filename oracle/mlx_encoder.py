"""EdgeTAM image encoder (RepViT-M1 + FpnNeck) in MLX-Python (NHWC) — parity vs golden image_embed.

RepViT-M1: stem (2× ConvNorm /4) → 4 stages [2,2,14,2] of RepViTBlock (legacy RepVggDw token_mixer
= conv3x3dw + conv1x1dw + identity; optional SE; RepVitMlp channel_mixer w/ residual); downsample
between stages. FpnNeck: 4 lateral 1x1 convs, top-down (nearest) fusion at levels [2,3]; image_embed
= the 64x64 level (after scalp drops the 32x32 level).
"""
import os
import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(__file__)
DIMS = [48, 96, 192, 384]
DEPTHS = [2, 2, 14, 2]

_SD = torch.load("/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM/checkpoints/edgetam.pt",
                 map_location="cpu", weights_only=False)["model"]
W = {k: v.float().numpy() for k, v in _SD.items()
     if k.startswith("image_encoder") or k == "no_mem_embed"}


def has(k):  # weight key present?
    return k in W


def cw(k):   # conv weight (O,I,kH,kW) -> NHWC (O,kH,kW,I)
    return mx.array(np.transpose(W[k], (0, 2, 3, 1)))


def a(k):
    return mx.array(W[k])


def gelu(x):
    return 0.5 * x * (1 + mx.erf(x / 1.4142135623730951))


def conv(x, wk, bk=None, stride=1, pad=0, groups=1):
    y = mx.conv2d(x, cw(wk), stride=stride, padding=pad, groups=groups)
    return y + a(bk) if bk is not None else y


def cn(x, p, stride=1, pad=0, groups=1):   # ConvNorm: conv(.c) + BN(.bn) eval
    y = mx.conv2d(x, cw(p + ".c.weight"), stride=stride, padding=pad, groups=groups)
    rm, rv = a(p + ".bn.running_mean"), a(p + ".bn.running_var")
    return (y - rm) / mx.sqrt(rv + 1e-5) * a(p + ".bn.weight") + a(p + ".bn.bias")


def se(x, p):   # SqueezeExcite: x * sigmoid(fc2(relu(fc1(gap))))
    g = x.mean(axis=(1, 2), keepdims=True)
    g = mx.maximum(conv(g, p + ".fc1.weight", p + ".fc1.bias"), 0)
    g = conv(g, p + ".fc2.weight", p + ".fc2.bias")
    return x * mx.sigmoid(g)


def mlp(x, p):   # RepVitMlp: conv2(gelu(conv1))
    return cn(gelu(cn(x, p + ".conv1")), p + ".conv2")


def repvgg_dw(x, p):   # legacy: conv(3x3 dw) + conv1(1x1 dw) + identity
    c = x.shape[-1]
    return cn(x, p + ".conv", pad=1, groups=c) + cn(x, p + ".conv1", groups=c) + x


def block(x, p):   # RepViTBlock
    x = repvgg_dw(x, p + ".token_mixer")
    if has(p + ".se.fc1.weight"):
        x = se(x, p + ".se")
    return x + mlp(x, p + ".channel_mixer")


def downsample(x, p):   # RepVitDownsample
    x = block(x, p + ".pre_block")
    x = cn(x, p + ".spatial_downsample", stride=2, pad=1, groups=x.shape[-1])
    x = cn(x, p + ".channel_downsample")
    return x + mlp(x, p + ".ffn")


def trunk(x):   # -> [stage0..3 features]
    t = "image_encoder.trunk.body"
    x = cn(x, f"{t}.stem.conv1", stride=2, pad=1)
    x = gelu(x)
    x = cn(x, f"{t}.stem.conv2", stride=2, pad=1)
    feats = []
    for s in range(4):
        sp = f"{t}.stages_{s}"
        if has(sp + ".downsample.spatial_downsample.c.weight"):
            x = downsample(x, sp + ".downsample")
        for b in range(DEPTHS[s]):
            x = block(x, f"{sp}.blocks.{b}")
        feats.append(x)
    return feats   # [48@256, 96@128, 192@64, 384@32]


def fpn(xs):   # FpnNeck top-down; returns out[0..3] (256ch); image_embed = out[2]
    n = "image_encoder.neck.convs"
    out = [None] * 4
    prev = None
    for i in range(3, -1, -1):
        lat = conv(xs[i], f"{n}.{3 - i}.conv.weight", f"{n}.{3 - i}.conv.bias")
        if i in (2, 3) and prev is not None:
            B, H, Wd, C = lat.shape
            td = mx.broadcast_to(prev.reshape(B, prev.shape[1], 1, prev.shape[2], 1, C),
                                 (B, prev.shape[1], 2, prev.shape[2], 2, C)).reshape(B, H, Wd, C)  # nearest 2x
            lat = lat + td
        prev = lat
        out[i] = lat
    return out


def main():
    x_nchw = np.load(f"{HERE}/goldens/enc_input.npy")
    x = mx.array(np.transpose(x_nchw, (0, 2, 3, 1)))      # NHWC
    xs = trunk(x)
    out = fpn(xs)
    # SAM2 adds the learned "no memory" embedding to the image features in image mode (no video memory).
    image_embed = out[2] + a("no_mem_embed").reshape(1, 1, 1, 256)   # 64x64 @256
    gold = np.transpose(np.load(f"{HERE}/goldens/image_embed.npy"), (0, 2, 3, 1))
    d = np.max(np.abs(np.array(image_embed) - gold))
    print(f"trunk feats: {[tuple(f.shape) for f in xs]}")
    print(f"image_embed mlx{tuple(image_embed.shape)} gold{gold.shape}  max_abs={d:.3e}  {'OK ✅' if d < 1e-3 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
