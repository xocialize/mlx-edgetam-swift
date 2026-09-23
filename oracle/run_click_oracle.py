"""EdgeTAM click oracle THROUGH set_image (AB-T-0171): upstream SAM2ImagePredictor on arbitrary images (non-square,
flat/synthetic) at a list of clicks → per-stage dumps for the Swift `edgetam-smoke stages` comparison.

Per image  (<out>/<name>.torch.npz): enc_input (1,3,1024,1024), image_embed (1,256,64,64),
                                     hrf0 (1,32,256,256), hrf1 (1,64,128,128).
Per click  (same npz, key prefix c<i>_): low_res (3,256,256) logits, scores (3,), area (3,) = fraction of the
                                        ORIGINAL-res image each multimask output selects (logit > 0).
Per box    (key prefix b<i>_): the SINGLE-mask output (multimask_output=False, SAM2's box recommendation, with
                               the dynamic-stability fallback): low_res (1,256,256), scores (1,), area (1,).

    oracle/.venv/bin/python oracle/run_click_oracle.py --repo oracle/upstream/EdgeTAM \
        --image fox.png --clicks 471,635 300,700 --out oracle/goldens/click
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "upstream/EdgeTAM"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--clicks", nargs="*", default=[], help="x,y in source px")
    ap.add_argument("--boxes", nargs="*", default=[], help="x0,y0,x1,y1 in source px (box-only prompt)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    image = os.path.abspath(args.image); out = os.path.abspath(args.out)
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo); os.chdir(repo)            # hydra config resolves from the repo root
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_grad_enabled(False)
    model = build_sam2("configs/edgetam.yaml", "checkpoints/edgetam.pt", device="cpu")
    pred = SAM2ImagePredictor(model)
    img = np.array(Image.open(image).convert("RGB"))
    enc_input = pred._transforms(img)[None]             # exactly what set_image feeds the backbone
    pred.set_image(img)
    f = pred._features
    dump = {"enc_input": enc_input.numpy(), "image_embed": f["image_embed"].numpy(),
            "hrf0": f["high_res_feats"][0].numpy(), "hrf1": f["high_res_feats"][1].numpy()}
    name = os.path.splitext(os.path.basename(image))[0]
    print(f"[oracle] {name} {img.shape[1]}x{img.shape[0]}")
    for i, c in enumerate(args.clicks):
        x, y = (float(v) for v in c.split(","))
        masks, scores, low = pred.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]),
                                          multimask_output=True, return_logits=True)
        area = (masks > 0).reshape(3, -1).mean(1)
        dump[f"c{i}_low_res"] = low; dump[f"c{i}_scores"] = scores; dump[f"c{i}_area"] = area
        dump[f"c{i}_click"] = np.array([x, y], np.float32)
        best = int(np.argmax(scores))
        print(f"  click {x:g},{y:g}  " + "  ".join(f"{s:.3f}→{a:.4f}" for s, a in zip(scores, area)) + f"  best={best}")
        np.save(f"{out}/{name}.c{i}.best_mask.npy", masks[best] > 0)
    for i, b in enumerate(args.boxes):
        box = np.array([float(v) for v in b.split(",")], np.float32)
        masks, scores, low = pred.predict(box=box, multimask_output=False, return_logits=True)
        area = (masks > 0).reshape(1, -1).mean(1)
        dump[f"b{i}_low_res"] = low; dump[f"b{i}_scores"] = scores; dump[f"b{i}_area"] = area
        print(f"  box {b}  single {scores[0]:.3f}→{area[0]:.4f}")
        np.save(f"{out}/{name}.b{i}.best_mask.npy", masks[0] > 0)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/{name}.torch.npz", **dump)


if __name__ == "__main__":
    main()
