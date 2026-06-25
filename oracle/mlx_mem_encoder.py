"""De-risk the EdgeTAM MemoryEncoder in MLX-Python vs golden (skip_mask_sigmoid=True path).
MaskDownSampler (4× conv-s2 + LN2d + GELU, then 1x1) + add pix_feat_proj + Fuser(2× CXBlock dw-k7) + out_proj."""
import os
import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(__file__)
W = {k[len("memory_encoder."):]: v.float().numpy()
     for k, v in torch.load("/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM/checkpoints/edgetam.pt",
                            map_location="cpu", weights_only=False)["model"].items() if k.startswith("memory_encoder.")}


def cw(k): return mx.array(np.transpose(W[k], (0, 2, 3, 1)))     # conv NHWC
def a(k): return mx.array(W[k])
def conv(x, k, stride=1, pad=0, groups=1):
    return mx.conv2d(x, cw(k + ".weight"), stride=stride, padding=pad, groups=groups) + a(k + ".bias")
def ln2d(x, k, eps=1e-6):                                        # channels-last LN (eps 1e-6)
    u = x.mean(-1, keepdims=True); d = x - u
    return d / mx.sqrt((d * d).mean(-1, keepdims=True) + eps) * a(k + ".weight") + a(k + ".bias")
def gelu(x): return 0.5 * x * (1 + mx.erf(x / 1.4142135623730951))


def mask_downsampler(m):                                         # m (1,1024,1024,1) NHWC
    e = "mask_downsampler.encoder"
    for ci, ni in [(0, 1), (3, 4), (6, 7), (9, 10)]:
        m = conv(m, f"{e}.{ci}", stride=2, pad=1)
        m = gelu(ln2d(m, f"{e}.{ni}"))
    return conv(m, f"{e}.12")                                    # final 1x1 -> 256


def cxblock(x, p):                                              # ConvNeXt block (NHWC)
    h = conv(x, p + ".dwconv", pad=3, groups=x.shape[-1])       # dw k7
    h = ln2d(h, p + ".norm")
    h = mx.matmul(h, a(p + ".pwconv1.weight").T) + a(p + ".pwconv1.bias")
    h = gelu(h)
    h = mx.matmul(h, a(p + ".pwconv2.weight").T) + a(p + ".pwconv2.bias")
    h = a(p + ".gamma") * h
    return x + h


def main():
    feat = mx.array(np.transpose(np.load(f"{HERE}/goldens/vid_me_in_feat.npy"), (0, 2, 3, 1)))   # (1,64,64,256)
    mask = mx.array(np.transpose(np.load(f"{HERE}/goldens/vid_me_in_mask.npy"), (0, 2, 3, 1)))   # (1,1024,1024,1)
    md = mask_downsampler(mask)                                 # (1,64,64,256)
    x = conv(feat, "pix_feat_proj") + md
    for i in range(2): x = cxblock(x, f"fuser.layers.{i}")
    x = conv(x, "out_proj")                                     # (1,64,64,64)
    gold = np.transpose(np.load(f"{HERE}/goldens/vid_me_out.npy"), (0, 2, 3, 1))
    d = np.max(np.abs(np.array(x) - gold))
    print(f"mem-encoder out mlx{tuple(x.shape)} gold{gold.shape}  max_abs={d:.3e}  {'OK ✅' if d < 1e-3 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
