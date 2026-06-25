"""EdgeTAM video oracle: run the SAM2 video predictor on a few bedroom frames, click frame 0,
propagate, and capture goldens for the memory stack — perceiver I/O, memory-attention I/O, memory-encoder
output, propagated masks. De-risk targets for P2 (video memory).
"""
import os, sys, shutil, glob
import numpy as np
from PIL import Image
import torch

REPO = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM"
sys.path.insert(0, REPO)
os.chdir(REPO)
import sam2  # noqa: E402  hydra config
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

HERE = "/Users/dustinnielson/Development/MLXEngine/mlx-edgetam-swift/oracle"


def main():
    torch.set_grad_enabled(False)
    # small frame subset (predictor loads a whole dir)
    tmp = "/tmp/bedroom5"; os.makedirs(tmp, exist_ok=True)
    for f in sorted(glob.glob(f"{REPO}/notebooks/videos/bedroom/*.jpg"))[:5]:
        shutil.copy(f, tmp)

    pred = build_sam2_video_predictor("configs/edgetam.yaml", "checkpoints/edgetam.pt", device="cpu")
    cap = {}
    # hook the spatial perceiver: forward(x, pos) -> (latents, pos)
    sp = pred.spatial_perceiver
    orig = sp.forward
    def wrap(x, pos=None):
        r = orig(x, pos)
        if "perc_in" not in cap:   # capture the first invocation (frame-0 memory encode)
            cap["perc_in"] = x.detach().numpy()
            cap["perc_pos_in"] = pos.detach().numpy() if pos is not None else None
            cap["perc_out"] = r[0].detach().numpy()
        return r
    sp.forward = wrap

    # hook MemoryEncoder.forward(pix_feat, masks, skip_mask_sigmoid) -> {"vision_features",...}
    me = pred.memory_encoder; me_orig = me.forward
    def me_wrap(pix_feat, masks, skip_mask_sigmoid=False):
        r = me_orig(pix_feat, masks, skip_mask_sigmoid)
        if "me_in_feat" not in cap:
            cap["me_in_feat"] = pix_feat.detach().numpy()
            cap["me_in_mask"] = masks.detach().numpy()
            cap["me_skip_sig"] = bool(skip_mask_sigmoid)
            cap["me_out"] = r["vision_features"].detach().numpy()
        return r
    me.forward = me_wrap

    # hook MemoryAttention.forward (first call = frame 1, conditioning on frame-0 memory)
    ma = pred.memory_attention; ma_orig = ma.forward
    def ma_wrap(curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0, num_spatial_mem=-1):
        r = ma_orig(curr, memory, curr_pos=curr_pos, memory_pos=memory_pos,
                    num_obj_ptr_tokens=num_obj_ptr_tokens, num_spatial_mem=num_spatial_mem)
        if "ma_curr" not in cap:
            c = curr[0] if isinstance(curr, list) else curr
            cp = curr_pos[0] if isinstance(curr_pos, list) else curr_pos
            cap["ma_curr"] = c.detach().numpy(); cap["ma_memory"] = memory.detach().numpy()
            cap["ma_curr_pos"] = cp.detach().numpy(); cap["ma_memory_pos"] = memory_pos.detach().numpy()
            cap["ma_num_obj_ptr"] = int(num_obj_ptr_tokens); cap["ma_num_spatial"] = int(num_spatial_mem)
            cap["ma_out"] = r.detach().numpy()
        return r
    ma.forward = ma_wrap

    # hook _forward_sam_heads to capture frame-1 (2nd call) tracking-decoder I/O:
    # backbone_features (= memory-conditioned pix_feat), high_res_features, raw masks, obj scores.
    sh_orig = pred._forward_sam_heads
    def sh_wrap(backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        r = sh_orig(backbone_features, point_inputs=point_inputs, mask_inputs=mask_inputs,
                    high_res_features=high_res_features, multimask_output=multimask_output)
        cap.setdefault("_sh_n", 0)
        n = cap["_sh_n"]; cap["_sh_n"] = n + 1
        if n == 1:  # frame 1 (frame 0 = call 0)
            cap["sh_backbone"] = backbone_features.detach().numpy()
            if high_res_features is not None:
                cap["sh_hrf0"] = high_res_features[0].detach().numpy()
                cap["sh_hrf1"] = high_res_features[1].detach().numpy()
            cap["sh_multimask"] = bool(multimask_output)
            cap["sh_has_point"] = point_inputs is not None
            cap["sh_low_res_multi"] = r[0].detach().numpy()   # low_res_multimasks (gated)
            cap["sh_low_res_best"] = r[3].detach().numpy()     # low_res_masks (best)
            cap["sh_obj_ptr"] = r[5].detach().numpy()
            cap["sh_obj_score"] = r[6].detach().numpy()
        return r
    pred._forward_sam_heads = sh_wrap

    state = pred.init_state(video_path=tmp)
    # Dump the preprocessed frames (NHWC) so the Swift full-propagate smoke runs on identical pixels.
    g = f"{HERE}/goldens"; os.makedirs(g, exist_ok=True)
    np.save(f"{g}/vid_frames.npy", np.transpose(state["images"].cpu().numpy(), (0, 2, 3, 1)))  # (T,1024,1024,3)
    pred.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                               points=np.array([[210, 350]], np.float32), labels=np.array([1], np.int32))
    masks = {}
    for fidx, obj_ids, mask_logits in pred.propagate_in_video(state):
        masks[fidx] = mask_logits.detach().cpu().numpy()  # (num_obj,1,H,W) logits

    np.save(f"{g}/vid_perc_in.npy", cap["perc_in"])
    np.save(f"{g}/vid_perc_out.npy", cap["perc_out"])
    if cap.get("perc_pos_in") is not None:
        np.save(f"{g}/vid_perc_pos_in.npy", cap["perc_pos_in"])
    # memory encoder goldens
    if "me_out" in cap:
        np.save(f"{g}/vid_me_in_feat.npy", cap["me_in_feat"]); np.save(f"{g}/vid_me_in_mask.npy", cap["me_in_mask"])
        np.save(f"{g}/vid_me_out.npy", cap["me_out"]); print(f"[vid-oracle] mem-enc feat{cap['me_in_feat'].shape} mask{cap['me_in_mask'].shape} skip_sig={cap['me_skip_sig']} -> {cap['me_out'].shape}")
    # memory attention goldens
    if "ma_out" in cap:
        for k in ["ma_curr", "ma_memory", "ma_curr_pos", "ma_memory_pos", "ma_out"]:
            np.save(f"{g}/vid_{k}.npy", cap[k])
        print(f"[vid-oracle] mem-attn curr{cap['ma_curr'].shape} memory{cap['ma_memory'].shape} num_obj_ptr={cap['ma_num_obj_ptr']} num_spatial={cap['ma_num_spatial']} -> {cap['ma_out'].shape}")
    # frame-1 tracking-decoder goldens
    if "sh_backbone" in cap:
        for k in ["sh_backbone", "sh_hrf0", "sh_hrf1", "sh_low_res_multi", "sh_low_res_best", "sh_obj_ptr", "sh_obj_score"]:
            if k in cap:
                np.save(f"{g}/vid_{k}.npy", cap[k])
        print(f"[vid-oracle] f1 sam-head: backbone{cap['sh_backbone'].shape} hrf0{cap.get('sh_hrf0', np.zeros(0)).shape} "
              f"multimask={cap['sh_multimask']} has_point={cap['sh_has_point']} "
              f"low_res_best{cap['sh_low_res_best'].shape} obj_score={cap['sh_obj_score'].ravel()}")
    for fidx, m in masks.items():
        np.save(f"{g}/vid_mask_f{fidx}.npy", m)
    print(f"[vid-oracle] perceiver in {cap['perc_in'].shape} -> out {cap['perc_out'].shape}")
    print(f"[vid-oracle] propagated frames: {sorted(masks)}  mask shape {masks[0].shape}")
    # quick mask coverage per frame (sanity: object tracked across frames)
    for fidx in sorted(masks):
        cov = (masks[fidx] > 0).mean() * 100
        print(f"  frame {fidx}: coverage {cov:.2f}%")


if __name__ == "__main__":
    main()
