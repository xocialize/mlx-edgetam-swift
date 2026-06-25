"""De-risk the EdgeTAM MemoryAttention (RoPE-2D, 2 layers, heads 1) in MLX-Python vs golden.
Self-attn (RoPEAttention, rope q&k 64²) + cross-attn (RoPEAttentionv2: q rope 64²; keys = 256 1D no-rope +
256 2D rope-16² + N obj-ptr no-rope) + MLP. pos_enc_at_input: output = curr + 0.1*curr_pos."""
import os
import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(__file__)
W = {k[len("memory_attention."):]: v.float().numpy()
     for k, v in torch.load("/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM/checkpoints/edgetam.pt",
                            map_location="cpu", weights_only=False)["model"].items() if k.startswith("memory_attention.")}
HD = 256  # internal_dim / heads (heads=1)


def a(k): return mx.array(W[k])
def lin(x, p): return mx.matmul(x, a(p + ".weight").T) + a(p + ".bias")
def ln(x, p):
    u = x.mean(-1, keepdims=True); d = x - u
    return d / mx.sqrt((d * d).mean(-1, keepdims=True) + 1e-5) * a(p + ".weight") + a(p + ".bias")
def relu(x): return mx.maximum(x, 0)


def axial_cossin(dim, ex, ey, theta=10000.0):                    # -> cos,sin (ex*ey, dim/2) numpy
    fr = 1.0 / (theta ** (np.arange(0, dim, 4)[: dim // 4] / dim))   # (dim//4,)
    t = np.arange(ex * ey); tx = (t % ex).astype(np.float32); ty = (t // ex).astype(np.float32)
    ang = np.concatenate([np.outer(tx, fr), np.outer(ty, fr)], -1)   # (N, dim/2)
    return mx.array(np.cos(ang).astype(np.float32)), mx.array(np.sin(ang).astype(np.float32))


def rope(x, cos, sin):                                           # x (B,N,256); cos/sin (N,128)
    B, N, C = x.shape
    xp = x.reshape(B, N, C // 2, 2)
    xr, xi = xp[..., 0], xp[..., 1]                              # (B,N,128)
    outr = xr * cos - xi * sin
    outi = xr * sin + xi * cos
    return mx.stack([outr, outi], axis=-1).reshape(B, N, C)


def sdpa(q, k, v):                                              # heads=1 -> (B,Nq,C)
    s = mx.softmax(mx.matmul(q, k.transpose(0, 2, 1)) / (HD ** 0.5), axis=-1)
    return mx.matmul(s, v)


# precompute rope tables
COS_Q, SIN_Q = axial_cossin(256, 64, 64)    # self + cross q (4096,128)
COS_K, SIN_K = axial_cossin(256, 16, 16)    # cross 2D keys (256,128)


def self_attn(tgt2, p):                                         # q=k=v=tgt2 (B,4096,256)
    q = rope(lin(tgt2, p + ".q_proj"), COS_Q, SIN_Q)
    k = rope(lin(tgt2, p + ".k_proj"), COS_Q, SIN_Q)
    v = lin(tgt2, p + ".v_proj")
    return lin(sdpa(q, k, v), p + ".out_proj")


def cross_attn(q_in, k_in, v_in, p, num_obj_ptr):              # q (B,4096,256); k/v (B,516,64)
    q = rope(lin(q_in, p + ".q_proj"), COS_Q, SIN_Q)
    k = lin(k_in, p + ".k_proj"); v = lin(v_in, p + ".v_proj")  # (B,516,256)
    num_k_rope = k.shape[1] - num_obj_ptr                       # 512 spatial
    # v2 split of the 512 spatial keys: first 256 (1D) no-rope, last 256 (2D) rope-16²
    k_spatial = k[:, :num_k_rope]                               # (B,512,256)
    k_1d = k_spatial[:, :256]                                   # no rope
    k_2d = rope(k_spatial[:, 256:512], COS_K, SIN_K)            # rope 16²
    k_rest = k[:, num_k_rope:]                                  # obj ptrs, no rope
    k = mx.concatenate([k_1d, k_2d, k_rest], axis=1)
    return lin(sdpa(q, k, v), p + ".out_proj")


def main():
    curr = np.load(f"{HERE}/goldens/vid_ma_curr.npy")           # (4096,1,256) seq-first
    memory = np.load(f"{HERE}/goldens/vid_ma_memory.npy")       # (516,1,64)
    curr_pos = np.load(f"{HERE}/goldens/vid_ma_curr_pos.npy")
    memory_pos = np.load(f"{HERE}/goldens/vid_ma_memory_pos.npy")
    num_obj_ptr = 4
    out = mx.array(np.transpose(curr, (1, 0, 2))) + 0.1 * mx.array(np.transpose(curr_pos, (1, 0, 2)))  # (1,4096,256)
    mem = mx.array(np.transpose(memory, (1, 0, 2)))             # (1,516,64)
    mem_k = mem + mx.array(np.transpose(memory_pos, (1, 0, 2))) # keys = memory + pos
    for i in range(2):
        p = f"layers.{i}"
        out = out + self_attn(ln(out, p + ".norm1"), p + ".self_attn")
        out = out + cross_attn(ln(out, p + ".norm2"), mem_k, mem, p + ".cross_attn_image", num_obj_ptr)
        out = out + lin(relu(lin(ln(out, p + ".norm3"), p + ".linear1")), p + ".linear2")
    out = ln(out, "norm")
    gold = np.transpose(np.load(f"{HERE}/goldens/vid_ma_out.npy"), (1, 0, 2))   # (1,4096,256)
    d = np.max(np.abs(np.array(out) - gold))
    print(f"mem-attn out mlx{tuple(out.shape)} gold{gold.shape}  max_abs={d:.3e}  {'OK ✅' if d < 1e-3 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
