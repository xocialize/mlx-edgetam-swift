"""De-risk the EdgeTAM tracking-decoder path (propagated frame, no point prompt, multimask + best-by-IoU +
object-score gating + obj_ptr extraction) in MLX-Python vs frame-1 goldens. Reuses mlx_decoder primitives.
Closes the video chain: memory_attention out (= backbone, verified bit-identical) → tracking decode → mask."""
import os
import numpy as np
import mlx.core as mx
import mlx_decoder as D   # reuses transformer/embed_prompt/dense_pe/mlp_head/convT/lin/ln/gelu/relu + weights

HERE = os.path.dirname(__file__)
a = D.a


def nhwc(x): return mx.array(np.transpose(x, (0, 2, 3, 1)))


def track_decode(backbone, feat_s0, feat_s1):
    Dn = "sam_mask_decoder"
    # empty prompt: point_inputs=None -> one (0,0) point label -1; prompt encoder pads one more (-1).
    sparse, dense = D.embed_prompt(mx.zeros((1, 2)), [-1])      # embed_prompt appends a pad -> 2 not-a-point tokens
    out_tokens = mx.concatenate([a(Dn + ".obj_score_token.weight"), a(Dn + ".iou_token.weight"),
                                 a(Dn + ".mask_tokens.weight")], axis=0)
    tokens = mx.concatenate([out_tokens.reshape(1, 6, 256), sparse], axis=1)
    src = backbone + dense
    q, keys = D.transformer(src, D.dense_pe(), tokens)
    obj_score = D.mlp_head(q[:, 0, :], Dn + ".pred_obj_score_head", 3)          # (1,1)
    iou_tok = q[:, 1, :]; mask_toks = q[:, 2:6, :]
    src = keys.reshape(1, 64, 64, 256)
    u = D.convT(src, Dn + ".output_upscaling.0.weight", Dn + ".output_upscaling.0.bias") + feat_s1
    u = D.gelu(D.ln(u, Dn + ".output_upscaling.1", eps=1e-6))
    u = D.gelu(D.convT(u, Dn + ".output_upscaling.3.weight", Dn + ".output_upscaling.3.bias") + feat_s0)
    hyper = mx.stack([D.mlp_head(mask_toks[:, i, :], f"{Dn}.output_hypernetworks_mlps.{i}", 3) for i in range(4)], axis=1)
    H, Wd = u.shape[1], u.shape[2]
    masks = mx.matmul(hyper, u.reshape(1, H * Wd, 32).transpose(0, 2, 1)).reshape(1, 4, H, Wd)
    iou = mx.sigmoid(D.mlp_head(iou_tok, Dn + ".iou_prediction_head", 3))       # (1,4)
    # multimask: drop index 0 -> 3 candidates; object-score hard-gate
    masks_m, iou_m = masks[:, 1:], iou[:, 1:]
    is_obj = (obj_score > 0).reshape(1, 1, 1, 1)
    masks_m = mx.where(is_obj, masks_m, mx.array(-1024.0))
    best = int(mx.argmax(iou_m[0]).item())
    low_res_best = masks_m[:, best:best + 1]
    # obj_ptr: best of the 3 multimask tokens (use_multimask_token_for_obj_ptr) -> proj -> gate
    sam_tok = mask_toks[:, 1:][:, best]                                         # (1,256)
    obj_ptr = D.mlp_head(sam_tok, "obj_ptr_proj", 3)
    lam = (obj_score > 0).astype(mx.float32)                                    # fixed_no_obj_ptr, hard
    obj_ptr = lam * obj_ptr + (1 - lam) * a("no_obj_ptr")
    return low_res_best, obj_ptr, obj_score, best, iou_m


def main():
    backbone = nhwc(np.load(f"{HERE}/goldens/vid_sh_backbone.npy"))             # (1,64,64,256)
    feat_s0 = nhwc(np.load(f"{HERE}/goldens/vid_sh_hrf0.npy"))                  # (1,256,256,32)
    feat_s1 = nhwc(np.load(f"{HERE}/goldens/vid_sh_hrf1.npy"))                  # (1,128,128,64)
    low_res_best, obj_ptr, obj_score, best, iou_m = track_decode(backbone, feat_s0, feat_s1)

    g_best = np.load(f"{HERE}/goldens/vid_sh_low_res_best.npy")                 # (1,1,256,256)
    g_ptr = np.load(f"{HERE}/goldens/vid_sh_obj_ptr.npy")                       # (1,256)
    g_score = np.load(f"{HERE}/goldens/vid_sh_obj_score.npy").ravel()
    dm = np.max(np.abs(np.array(low_res_best) - g_best))
    dp = np.max(np.abs(np.array(obj_ptr) - g_ptr))
    ds = abs(float(np.array(obj_score).ravel()[0]) - float(g_score[0]))
    ok = dm < 1e-2 and dp < 1e-3 and ds < 1e-3
    print(f"track-decode  best_idx={best} iou={np.array(iou_m)[0].round(3)}  obj_score mlx={float(np.array(obj_score).ravel()[0]):.4f} gold={g_score[0]:.4f}")
    print(f"  low_res_best max_abs={dm:.3e}  obj_ptr max_abs={dp:.3e}  obj_score d={ds:.3e}  {'OK ✅' if ok else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
