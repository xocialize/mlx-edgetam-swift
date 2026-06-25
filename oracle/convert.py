"""EdgeTAM image+video weights → canonical MLX safetensors (NHWC) + Swift parity fixtures.

Conv (O,I,kH,kW)→(O,kH,kW,I); ConvTranspose (output_upscaling.0/.3) (I,O,kH,kW)→(O,kH,kW,I);
Linear/2D kept; drop num_batches_tracked. Flat torch keys (Swift WeightLoading maps them).

Image-mode keys: image_encoder, sam_prompt_encoder, sam_mask_decoder, no_mem_embed.
Video (P2) keys (147): memory_encoder, memory_attention, spatial_perceiver, obj_ptr_proj,
no_obj_ptr, no_mem_pos_enc, no_obj_embed_spatial, maskmem_tpos_enc. Of these only 9 are real
convs needing NHWC: memory_encoder.mask_downsampler.encoder.{0,3,6,9,12}.weight,
memory_encoder.pix_feat_proj.weight, memory_encoder.fuser.layers.{0,1}.dwconv.weight,
memory_encoder.out_proj.weight. maskmem_tpos_enc (7,1,1,64) is a learned param — kept RAW
(in RAW_4D). Everything else (memory_attention 54 / spatial_perceiver 44 / obj_ptr_proj 6 / …)
is linear/norm/1D — kept raw.
"""
import argparse
import os
import numpy as np
import torch
import mlx.core as mx

HERE = os.path.dirname(__file__)
CKPT = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM/checkpoints/edgetam.pt"
CONV_T = {"sam_mask_decoder.output_upscaling.0.weight", "sam_mask_decoder.output_upscaling.3.weight"}
RAW_4D = {"maskmem_tpos_enc"}  # learned param (7,1,1,64), NOT a conv → no NHWC transpose
KEEP = ("image_encoder", "sam_prompt_encoder", "sam_mask_decoder", "no_mem_embed",
        "memory_encoder", "memory_attention", "spatial_perceiver", "obj_ptr_proj",
        "no_obj_ptr", "no_mem_pos_enc", "no_obj_embed_spatial", "maskmem_tpos_enc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--out", default=f"{HERE}/weights/edgetam_fp32.safetensors")
    args = ap.parse_args()
    dt = mx.float16 if args.dtype == "float16" else mx.float32
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]

    out = {}
    for k, v in sd.items():
        if not k.startswith(KEEP) or k.endswith("num_batches_tracked"):
            continue
        a = v.float().numpy()
        if a.ndim == 4 and k not in RAW_4D:
            a = np.transpose(a, (1, 2, 3, 0)) if k in CONV_T else np.transpose(a, (0, 2, 3, 1))
        out[k] = mx.array(a.astype(np.float32)).astype(dt)
    mx.eval(list(out.values()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mx.save_safetensors(args.out, out)
    print(f"[convert] {len(out)} tensors ({args.dtype}) -> {args.out}")

    # Swift parity fixture (NHWC inputs + goldens)
    g = f"{HERE}/goldens"
    fx = {
        "enc_input": mx.array(np.transpose(np.load(f"{g}/enc_input.npy"), (0, 2, 3, 1)).astype(np.float32)),
        "image_embed": mx.array(np.transpose(np.load(f"{g}/image_embed.npy"), (0, 2, 3, 1)).astype(np.float32)),
        "unnorm_coords": mx.array(np.load(f"{g}/unnorm_coords.npy")[0].astype(np.float32)),  # (1,2)
        "labels": mx.array(np.load(f"{g}/labels.npy")[0].astype(np.float32)),                # (1,)
        "masks_raw": mx.array(np.load(f"{g}/dec_masks_raw.npy")[0].astype(np.float32)),       # (3,256,256)
        "scores": mx.array(np.load(f"{g}/scores.npy").astype(np.float32)),                    # (3,)
    }
    mx.save_safetensors(f"{HERE}/weights/parity.safetensors", fx)
    print(f"[fixture] {[ (k,tuple(v.shape)) for k,v in fx.items() ]}")

    video_fixture(g)


def video_fixture(g):
    """Bundle P2 video goldens (op I/O + pos + full-propagate frames/masks) for the Swift smoke.
    4D feature/mask tensors → NHWC; seq-first mem-attn tensors kept (N,B,C); masks → (H,W)."""
    def nhwc(p): return mx.array(np.transpose(np.load(p), (0, 2, 3, 1)).astype(np.float32))
    def raw(p): return mx.array(np.load(p).astype(np.float32))
    vf = {
        # PerceiverResampler: in (1,64,64,64) NHWC + pos NHWC → out (1,512,64)
        "perc_in": nhwc(f"{g}/vid_perc_in.npy"),
        "perc_pos_in": nhwc(f"{g}/vid_perc_pos_in.npy"),
        "perc_out": raw(f"{g}/vid_perc_out.npy"),
        # MemoryEncoder: feat (1,64,64,256) + mask (1,1024,1024,1) → out (1,64,64,64)
        "me_in_feat": nhwc(f"{g}/vid_me_in_feat.npy"),
        "me_in_mask": nhwc(f"{g}/vid_me_in_mask.npy"),
        "me_out": nhwc(f"{g}/vid_me_out.npy"),
        # MemoryAttention: seq-first (N,B,C)
        "ma_curr": raw(f"{g}/vid_ma_curr.npy"),
        "ma_curr_pos": raw(f"{g}/vid_ma_curr_pos.npy"),
        "ma_memory": raw(f"{g}/vid_ma_memory.npy"),
        "ma_memory_pos": raw(f"{g}/vid_ma_memory_pos.npy"),
        "ma_out": raw(f"{g}/vid_ma_out.npy"),
        # TrackDecode: backbone (1,64,64,256) + hrf NHWC → low_res_best (1,1,256,256), obj_ptr (1,256), obj_score (1,1)
        "sh_backbone": nhwc(f"{g}/vid_sh_backbone.npy"),
        "sh_hrf0": nhwc(f"{g}/vid_sh_hrf0.npy"),
        "sh_hrf1": nhwc(f"{g}/vid_sh_hrf1.npy"),
        "sh_low_res_best": raw(f"{g}/vid_sh_low_res_best.npy"),
        "sh_obj_ptr": raw(f"{g}/vid_sh_obj_ptr.npy"),
        "sh_obj_score": raw(f"{g}/vid_sh_obj_score.npy"),
        # Full propagate: frames (T,1024,1024,3) already NHWC + per-frame masks (H,W)
        "frames": raw(f"{g}/vid_frames.npy"),
    }
    for i in range(5):
        vf[f"mask_f{i}"] = mx.array(np.load(f"{g}/vid_mask_f{i}.npy")[0, 0].astype(np.float32))  # (540,960)
    # ENHANCEMENT(v2) goldens: per-object multi-object tracks + box-prompt track (all (H,W)).
    import os
    for i in range(5):
        for o in range(2):
            p = f"{g}/vid_mo_obj{o}_f{i}.npy"
            if os.path.exists(p):
                vf[f"mo_obj{o}_f{i}"] = mx.array(np.load(p)[0, 0].astype(np.float32))
        bp = f"{g}/vid_box_f{i}.npy"
        if os.path.exists(bp):
            vf[f"box_f{i}"] = mx.array(np.load(bp)[0, 0].astype(np.float32))
    if os.path.exists(f"{g}/vid_box_xyxy.npy"):
        vf["box_xyxy"] = mx.array(np.load(f"{g}/vid_box_xyxy.npy").astype(np.float32))  # (4,)
    mx.save_safetensors(f"{HERE}/weights/parity_video.safetensors", vf)
    print(f"[vid-fixture] {[(k, tuple(v.shape)) for k, v in vf.items()]}")


if __name__ == "__main__":
    main()
