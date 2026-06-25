"""EdgeTAM image-mode oracle: load edgetam.pt, run the image predictor on a point click → golden mask
+ intermediates (image embedding, mask logits). De-risk gate before any Swift.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch

REPO = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM"
sys.path.insert(0, REPO)
os.chdir(REPO)  # hydra config + relative paths resolve from the repo root

import sam2  # noqa: E402  registers the hydra config module
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

HERE = "/Users/dustinnielson/Development/MLXEngine/mlx-edgetam-swift/oracle"


def main():
    torch.set_grad_enabled(False)
    model = build_sam2("configs/edgetam.yaml", "checkpoints/edgetam.pt", device="cpu")
    predictor = SAM2ImagePredictor(model)

    img = np.array(Image.open("notebooks/images/truck.jpg").convert("RGB"))
    predictor.set_image(img)

    pt = np.array([[500, 375]]); lbl = np.array([1])      # click on the truck
    masks, scores, low_res = predictor.predict(
        point_coords=pt, point_labels=lbl, multimask_output=True, return_logits=True)

    best = int(np.argmax(scores))
    feats = predictor._features  # image_embed + high_res_feats
    g = HERE + "/goldens"; os.makedirs(g, exist_ok=True)
    np.save(f"{g}/image_embed.npy", feats["image_embed"].cpu().numpy())
    for i, hf in enumerate(feats.get("high_res_feats", [])):
        np.save(f"{g}/high_res_feat{i}.npy", hf.cpu().numpy())
    np.save(f"{g}/masks_logits.npy", masks)          # (3, H, W) logits (return_logits)
    np.save(f"{g}/scores.npy", scores)
    np.save(f"{g}/low_res.npy", low_res)
    np.save(f"{g}/input_image.npy", img)

    print(f"[oracle] image {img.shape}  embed {tuple(feats['image_embed'].shape)}")
    print(f"[oracle] high_res_feats: {[tuple(h.shape) for h in feats.get('high_res_feats', [])]}")
    print(f"[oracle] masks {masks.shape}  scores {scores.round(3)}  best={best}")

    # overlay the best mask for a visual gate
    mask = masks[best] > 0
    ov = img.copy(); ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([30, 144, 255])).astype(np.uint8)
    ov[..., :][np.array([[500 - 4 <= x <= 500 + 4 and 375 - 4 <= y <= 375 + 4
                          for x in range(img.shape[1])] for y in range(img.shape[0])])] = [255, 0, 0]
    Image.fromarray(ov).save(f"{HERE}/edgetam_mask.png")
    print(f"[oracle] mask coverage {mask.mean()*100:.1f}%  → goldens + edgetam_mask.png")


if __name__ == "__main__":
    main()
