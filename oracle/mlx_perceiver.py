"""De-risk the EdgeTAM 2D-spatial PerceiverResampler (memory compression) in MLX-Python vs golden.
Input memory feats (1,64,64,64) + pos → 512 latents (256 1D-global + 256 2D-windowed)."""
import math, os
import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(__file__)
W = {k[len("spatial_perceiver."):]: v.float().numpy()
     for k, v in torch.load("/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM/checkpoints/edgetam.pt",
                            map_location="cpu", weights_only=False)["model"].items() if k.startswith("spatial_perceiver.")}
DIM, HEADS = 64, 1
SCALE = (DIM // HEADS) ** -0.5


def a(k): return mx.array(W[k])
def lin(x, k): return mx.matmul(x, a(k + ".weight").T)           # bias-free Linear
def ln(x, p):
    u = x.mean(-1, keepdims=True); d = x - u
    return d / mx.sqrt((d * d).mean(-1, keepdims=True) + 1e-5) * a(p + ".weight") + a(p + ".bias")
def gelu(x): return 0.5 * x * (1 + mx.erf(x / 1.4142135623730951))


def sdpa(q, k, v):                                              # heads=1: (B,N,C)
    s = mx.softmax(mx.matmul(q, k.transpose(0, 2, 1)) * SCALE, axis=-1)
    return mx.matmul(s, v)


def perceiver_attn(latents, x, p, pos=None):                   # cross-attn (concat_kv_latents=false)
    lat = ln(latents, p + ".norm_latents"); xx = ln(x, p + ".norm_x")
    q = lin(lat, p + ".to_q")
    kv = lin(xx, p + ".to_kv"); k, v = kv[..., :DIM], kv[..., DIM:]
    if pos is not None: k = k + pos; v = v + pos
    return lin(sdpa(q, k, v), p + ".to_out")


def self_attn(x, p):
    xx = ln(x, p + ".norm")
    q = lin(xx, p + ".to_q"); kv = lin(xx, p + ".to_kv"); k, v = kv[..., :DIM], kv[..., DIM:]
    return lin(sdpa(q, k, v), p + ".to_out")


def ff(x, p):                                                  # LN + Linear(4x) + GELU + Linear
    return lin(gelu(lin(ln(x, p + ".0"), p + ".1")), p + ".3")


def layer(latents, x, p, pos=None):
    latents = perceiver_attn(latents, x, p + ".attn", pos) + latents
    latents = ff(latents, p + ".ff") + latents
    latents = self_attn(latents, p + ".self_attn") + latents
    latents = ff(latents, p + ".self_ff") + latents
    return latents


def forward_1d(x, pos):                                        # x (1,64,64,64) NCHW, pos same
    lat = a("latents").reshape(1, 256, DIM)
    xf = mx.array(np.transpose(np.array(x), (0, 2, 3, 1))).reshape(1, 64 * 64, DIM)
    pf = mx.array(np.transpose(np.array(pos), (0, 2, 3, 1))).reshape(1, 64 * 64, DIM)
    for i in range(2): lat = layer(lat, xf, f"layers.{i}", pf)
    return ln(lat, "norm")


def forward_2d(x):                                             # x (1,64,64,64)
    B, C, H, Wd = 1, DIM, 64, 64
    lat = a("latents_2d").reshape(256, 1, DIM)                 # (B*256,1,C), B=1
    nwin = int(math.sqrt(256)); ws = H // nwin                # 16, 4
    xp = mx.array(np.transpose(np.array(x), (0, 2, 3, 1)))     # (1,64,64,64) NHWC
    xw = xp.reshape(B, nwin, ws, nwin, ws, C).transpose(0, 1, 3, 2, 4, 5).reshape(256, ws * ws, C)
    for i in range(2): lat = layer(lat, xw, f"layers.{i}")     # no pos
    lat = lat.reshape(B, nwin, nwin, C).reshape(B, nwin * nwin, C)
    return ln(lat, "norm")


def main():
    x = mx.array(np.load(f"{HERE}/goldens/vid_perc_in.npy"))
    pos = mx.array(np.load(f"{HERE}/goldens/vid_perc_pos_in.npy"))
    l1 = forward_1d(x, pos); l2 = forward_2d(x)
    out = mx.concatenate([l1, l2], axis=1)
    gold = np.load(f"{HERE}/goldens/vid_perc_out.npy")
    d = np.max(np.abs(np.array(out) - gold))
    print(f"perceiver out mlx{tuple(out.shape)} gold{gold.shape}  max_abs={d:.3e}  {'OK ✅' if d < 1e-3 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
