"""Golden for the committed NON-SQUARE end-to-end parity test (Tests/EdgeTAMParityTests, AB-T-0171).

Runs upstream SAM2ImagePredictor THROUGH set_image on
  • gate  — the Forge Canvas Lab pointer-gate page, 1200×800 grey 128 + five 12-px colour squares, regenerated
            here and in the Swift test from the same recipe (no image file needed);
  • truck — upstream notebooks/images/truck.jpg (1800×1200; the test uses oracle/goldens/truck.png, a lossless
            copy of PIL's decode, when present),
at single clicks (multimask_output=True, best by IoU) and box prompts (multimask_output=False — SAM2's
recommendation, with the dynamic-stability fallback), and writes scores + the chosen mask (row-major RLE,
runs alternate starting with background) to Tests/EdgeTAMParityTests/nonsquare_golden.json.

    oracle/.venv/bin/python oracle/make_parity_golden.py
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "upstream/EdgeTAM")
OUT = os.path.join(HERE, "..", "Tests/EdgeTAMParityTests/nonsquare_golden.json")
MARKS = [((37, 41), (255, 0, 255)), ((1149, 58), (0, 255, 255)), ((601, 397), (255, 255, 0)),
         ((71, 757), (255, 0, 0)), ((1003, 731), (0, 0, 255))]


def gate_page():
    img = np.full((800, 1200, 3), 128, np.uint8)
    for (x, y), rgb in MARKS:
        img[y:y + 12, x:x + 12] = rgb
    return img


def rle(mask):
    flat = mask.reshape(-1).astype(np.uint8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    return runs if flat[0] == 0 else [0] + runs


def main():
    sys.path.insert(0, REPO); os.chdir(REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_grad_enabled(False)
    pred = SAM2ImagePredictor(build_sam2("configs/edgetam.yaml", "checkpoints/edgetam.pt", device="cpu"))

    cases = {
        "gate": (gate_page(), [[43, 47], [1155, 64], [607, 403], [77, 763], [1009, 737]],
                 [[30, 34, 56, 60], [590, 386, 626, 422]]),
        "truck": (np.array(Image.open("notebooks/images/truck.jpg").convert("RGB")),
                  [[500, 375], [1375, 550]], [[425, 600, 700, 875], [75, 275, 1725, 850]]),
    }
    golden = {"source": "facebookresearch/EdgeTAM SAM2ImagePredictor, CPU fp32, checkpoints/edgetam.pt", "images": {}}
    for name, (img, clicks, boxes) in cases.items():
        pred.set_image(img)
        entries = []
        for c in clicks:
            m, s, _ = pred.predict(point_coords=np.array([c], np.float32), point_labels=np.array([1]),
                                   multimask_output=True)
            b = int(np.argmax(s))
            entries.append({"click": c, "scores": s.tolist(), "best": b, "score": float(s[b]), "mask": rle(m[b] > 0)})
        for bx in boxes:
            m, s, _ = pred.predict(box=np.array(bx, np.float32), multimask_output=False)
            entries.append({"box": bx, "score": float(s[0]), "mask": rle(m[0] > 0)})
        golden["images"][name] = {"width": img.shape[1], "height": img.shape[0], "prompts": entries}
        print(f"[golden] {name} {img.shape[1]}x{img.shape[0]}: "
              + "  ".join(f"{e.get('click', e.get('box'))}→{e['score']:.3f}" for e in entries))
    with open(OUT, "w") as f:
        json.dump(golden, f, separators=(",", ":"))
    print(f"[golden] → {os.path.normpath(OUT)} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
