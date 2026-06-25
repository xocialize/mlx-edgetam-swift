"""EdgeTAM P1b: SAM prompt encoder + mask decoder in MLX-Python — parity vs golden low-res masks + IoU.
Reuses mlx_encoder for image_embed + FPN feats. Modules kept SAM-generic (future segment_anything reuse).
"""
import os
import math
import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
import mlx_encoder as E   # provides trunk/fpn + the full checkpoint via E._SD

HERE = os.path.dirname(__file__)
SD = E._SD
W = {k: v.float().numpy() for k, v in SD.items()}


def a(k): return mx.array(W[k])
def cw(k): return mx.array(np.transpose(W[k], (0, 2, 3, 1)))            # conv (O,I,kH,kW)->NHWC
def ctw(k): return mx.array(np.transpose(W[k], (1, 2, 3, 0)))           # convT (I,O,kH,kW)->(O,kH,kW,I)


def lin(x, p):                                                          # nn.Linear: x@W.T + b
    return mx.matmul(x, a(p + ".weight").T) + a(p + ".bias")


def ln(x, p, eps=1e-5):                                                 # LayerNorm over last axis
    u = x.mean(-1, keepdims=True); v = ((x - u) ** 2).mean(-1, keepdims=True)
    return (x - u) / mx.sqrt(v + eps) * a(p + ".weight") + a(p + ".bias")


def gelu(x): return 0.5 * x * (1 + mx.erf(x / 1.4142135623730951))
def relu(x): return mx.maximum(x, 0)


# ---------- PositionEmbeddingRandom ----------
GAUSS = a("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix")                                # (2,128)


def pe_encoding(coords):                                                # coords (...,2) in [0,1]
    c = 2 * coords - 1
    c = mx.matmul(c, GAUSS)                                             # (...,128)
    c = 2 * math.pi * c
    return mx.concatenate([mx.sin(c), mx.cos(c)], axis=-1)              # (...,256)


def pe_with_coords(coords, size=1024.0):                               # coords in pixels
    return pe_encoding(coords / size)


def dense_pe(h=64, w=64):                                               # get_dense_pe -> (1,h,w,256) NHWC
    ys = (mx.arange(h, dtype=mx.float32) + 0.5) / h
    xs = (mx.arange(w, dtype=mx.float32) + 0.5) / w
    yg = mx.broadcast_to(ys.reshape(h, 1), (h, w))
    xg = mx.broadcast_to(xs.reshape(1, w), (h, w))
    return pe_encoding(mx.stack([xg, yg], axis=-1)).reshape(1, h, w, 256)


# ---------- prompt encoder ----------
def embed_prompt(coords_px, labels):                                    # coords_px (N,2), labels (N,)
    pts = coords_px + 0.5                                               # pixel center
    pad = mx.array([[0.0, 0.0]]); pts = mx.concatenate([pts, pad], axis=0)
    labels = list(labels) + [-1]
    pe = pe_with_coords(pts)                                            # (N+1,256)
    rows = []
    for i, lb in enumerate(labels):
        v = pe[i]
        if lb == -1:
            v = mx.zeros_like(v) + a("sam_prompt_encoder.not_a_point_embed.weight")[0]
        else:
            v = v + a(f"sam_prompt_encoder.point_embeddings.{int(lb)}.weight")[0]
        rows.append(v)
    sparse = mx.stack(rows, axis=0).reshape(1, len(rows), 256)          # (1,N+1,256)
    dense = mx.broadcast_to(a("sam_prompt_encoder.no_mask_embed.weight").reshape(1, 1, 1, 256), (1, 64, 64, 256))
    return sparse, dense


# ---------- attention ----------
def attn(q, k, v, p, heads=8):                                          # q,k,v (B,Nx,256)
    Q, K, V = lin(q, p + ".q_proj"), lin(k, p + ".k_proj"), lin(v, p + ".v_proj")
    B, Nq, C = Q.shape; Nk = K.shape[1]; hd = C // heads
    Q = Q.reshape(B, Nq, heads, hd).transpose(0, 2, 1, 3)
    K = K.reshape(B, Nk, heads, hd).transpose(0, 2, 1, 3)
    V = V.reshape(B, Nk, heads, hd).transpose(0, 2, 1, 3)
    s = mx.softmax(mx.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(hd), axis=-1)
    o = mx.matmul(s, V).transpose(0, 2, 1, 3).reshape(B, Nq, C)
    return lin(o, p + ".out_proj")


def two_way_block(q, k, qpe, kpe, p, skip_pe):
    if skip_pe:
        q = attn(q, q, q, p + ".self_attn")          # reference REPLACES queries (no residual) on layer 0
    else:
        qq = q + qpe; q = q + attn(qq, qq, q, p + ".self_attn")
    q = ln(q, p + ".norm1")
    aout = attn(q + qpe, k + kpe, k, p + ".cross_attn_token_to_image"); q = ln(q + aout, p + ".norm2")
    m = lin(relu(lin(q, p + ".mlp.layers.0")), p + ".mlp.layers.1"); q = ln(q + m, p + ".norm3")
    aout = attn(k + kpe, q + qpe, q, p + ".cross_attn_image_to_token"); k = ln(k + aout, p + ".norm4")
    return q, k


def transformer(image_embed, image_pe, tokens):                        # image_embed/pe NHWC (1,64,64,256)
    keys = image_embed.reshape(1, 64 * 64, 256)
    kpe = image_pe.reshape(1, 64 * 64, 256)
    q = tokens
    T = "sam_mask_decoder.transformer"
    for i in range(2):
        q, keys = two_way_block(q, keys, tokens, kpe, f"{T}.layers.{i}", skip_pe=(i == 0))
    aout = attn(q + tokens, keys + kpe, keys, f"{T}.final_attn_token_to_image")
    q = ln(q + aout, f"{T}.norm_final_attn")
    return q, keys


# ---------- mask decoder ----------
def mlp_head(x, p, n, act_last=False):
    for i in range(n):
        x = lin(x, f"{p}.layers.{i}")
        if i < n - 1: x = relu(x)
    return x


def convT(x, k, b):                                                     # ConvTranspose2d s2
    return mx.conv_transpose2d(x, ctw(k), stride=2, padding=0) + a(b)


def decode(image_embed, feat_s0, feat_s1, sparse, dense):
    D = "sam_mask_decoder"
    out_tokens = mx.concatenate([a(D + ".obj_score_token.weight"), a(D + ".iou_token.weight"), a(D + ".mask_tokens.weight")], axis=0)
    tokens = mx.concatenate([out_tokens.reshape(1, 6, 256), sparse], axis=1)   # (1,6+N,256)
    src = image_embed + dense                                          # (1,64,64,256)
    q, keys = transformer(src, dense_pe(), tokens)
    iou_tok = q[:, 1, :]; mask_toks = q[:, 2:6, :]                      # s=1
    src = keys.reshape(1, 64, 64, 256)                                 # back to NHWC
    # upscaling: act1(ln1(dc1(src)+feat_s1)); act2(dc2(.)+feat_s0)
    u = convT(src, D + ".output_upscaling.0.weight", D + ".output_upscaling.0.bias") + feat_s1
    u = ln(u, D + ".output_upscaling.1", eps=1e-6)                     # LayerNorm2d (last-axis NHWC, eps 1e-6)
    u = gelu(u)
    u = convT(u, D + ".output_upscaling.3.weight", D + ".output_upscaling.3.bias") + feat_s0
    u = gelu(u)                                                        # (1,256,256,32)
    hyper = mx.stack([mlp_head(mask_toks[:, i, :], f"{D}.output_hypernetworks_mlps.{i}", 3) for i in range(4)], axis=1)  # (1,4,32)
    H, Wd = u.shape[1], u.shape[2]
    masks = mx.matmul(hyper, u.reshape(1, H * Wd, 32).transpose(0, 2, 1)).reshape(1, 4, H, Wd)
    iou = mx.sigmoid(mlp_head(iou_tok, D + ".iou_prediction_head", 3))  # (1,4), sigmoid output
    return masks[:, 1:], iou[:, 1:]                                    # multimask: drop index 0 → 3


def main():
    x = mx.array(np.transpose(np.load(f"{HERE}/goldens/enc_input.npy"), (0, 2, 3, 1)))
    out = E.fpn(E.trunk(x))
    image_embed = out[2] + a("no_mem_embed").reshape(1, 1, 1, 256)
    feat_s0 = mx.conv2d(out[0], cw("sam_mask_decoder.conv_s0.weight"), stride=1) + a("sam_mask_decoder.conv_s0.bias")
    feat_s1 = mx.conv2d(out[1], cw("sam_mask_decoder.conv_s1.weight"), stride=1) + a("sam_mask_decoder.conv_s1.bias")
    coords = mx.array(np.load(f"{HERE}/goldens/unnorm_coords.npy")[0])  # (1,2)
    labels = np.load(f"{HERE}/goldens/labels.npy")[0]
    sparse, dense = embed_prompt(coords, labels)
    masks, iou = decode(image_embed, feat_s0, feat_s1, sparse, dense)

    gmask = np.load(f"{HERE}/goldens/dec_masks_raw.npy")[0]             # (3,256,256) RAW decoder output
    gscore = np.load(f"{HERE}/goldens/scores.npy")                      # (predict postprocesses low_res; raw is the port target)
    dm = np.max(np.abs(np.array(masks)[0] - gmask))
    ds = np.max(np.abs(np.array(iou)[0] - gscore))
    print(f"low_res masks max_abs={dm:.3e}  iou max_abs={ds:.3e}  {'OK ✅' if dm < 1e-2 and ds < 1e-3 else 'FAIL ❌'}")
    print(f"  mlx iou={np.array(iou)[0].round(3)}  gold={gscore.round(3)}")


if __name__ == "__main__":
    main()
