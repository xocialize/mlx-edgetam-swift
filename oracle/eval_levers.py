"""AB-T-0171 quality levers, measured on the upstream PyTorch predictor (the Swift port matches it to mask IoU
1.0000, so these numbers ARE the port's). Prompts on the Lab fixtures:

  fox_caption.png (1024², flat anime + the Lab caption) — reference = BiRefNet's whole-fox subject mask
      (the Lab's Select Subject output, fox_subject_birefnet.png): "IoU·fox" = how well the prompt selects the fox.
  gate.png (1200×800 grey + five 12-px squares) — which squares each click's mask contains.

Levers: single click (SAM2 multimask best-by-IoU) · best-of-3 with hindsight (does ANY of the three masks
match?) · 2/3 positive clicks (single-mask + stability fallback) · click→refine with the previous low-res
logits as mask_input (the SAM2 demo loop) · box prompt. Models: EdgeTAM and, when their checkpoints exist,
SAM 2.1 hiera-small / base+ (Apache-2.0) as the "bigger model" reference.

    oracle/.venv/bin/python oracle/eval_levers.py [--sam21 /Volumes/Satechi/Models/sam2.1-oracle]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "goldens/click")
FOX_CLICKS = [(471, 635), (620, 560), (620, 400), (300, 700), (100, 780)]
MARKS = {"A": (37, 41), "B": (1149, 58), "C": (601, 397), "D": (71, 757), "E": (1003, 731)}


def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sam21", default="/Volumes/Satechi/Models/sam2.1-oracle")
    args = ap.parse_args()
    fox = np.array(Image.open(f"{G}/fox_caption.png").convert("RGB"))
    gt = np.array(Image.open(f"{G}/fox_subject_birefnet.png").convert("L")) > 127
    gate = np.array(Image.open(f"{G}/gate.png").convert("RGB"))
    repo = os.path.join(HERE, "upstream/EdgeTAM"); sys.path.insert(0, repo); os.chdir(repo)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_grad_enabled(False)
    models = [("EdgeTAM", "configs/edgetam.yaml", "checkpoints/edgetam.pt")]
    for tag, cfg, f in (("SAM2.1-S", "configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
                        ("SAM2.1-B+", "configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt")):
        if os.path.exists(os.path.join(args.sam21, f)): models.append((tag, cfg, os.path.join(args.sam21, f)))

    for tag, cfg, ckpt in models:
        pred = SAM2ImagePredictor(build_sam2(cfg, ckpt, device="cpu"))
        nparam = sum(p.numel() for p in pred.model.parameters()) / 1e6
        t0 = time.time(); pred.set_image(fox); enc = time.time() - t0
        print(f"\n### {tag}  ({nparam:.1f} M params, CPU set_image {enc:.2f} s)")
        print("| fox prompt | pred-IoU | area | IoU·fox | best-of-3 IoU·fox |\n|---|---|---|---|---|")
        for c in FOX_CLICKS:
            m, s, low = pred.predict(point_coords=np.array([c]), point_labels=np.array([1]), multimask_output=True)
            b = int(np.argmax(s)); fits = [iou(mi > 0, gt) for mi in m]
            print(f"| click {c[0]},{c[1]} | {s[b]:.3f} | {(m[b] > 0).mean():.3f} | {fits[b]:.3f} | {max(fits):.3f} |")
        for pts in (FOX_CLICKS[:1] + FOX_CLICKS[3:4], [FOX_CLICKS[0], FOX_CLICKS[3], FOX_CLICKS[2]]):
            m, s, _ = pred.predict(point_coords=np.array(pts), point_labels=np.ones(len(pts)), multimask_output=False)
            print(f"| {len(pts)} clicks {' + '.join(f'{x},{y}' for x, y in pts)} | {s[0]:.3f} | {(m[0] > 0).mean():.3f} | {iou(m[0] > 0, gt):.3f} | – |")
        # SAM2 demo loop: click 1 (multimask) → add click 2 with the previous best low-res logits as mask_input
        m, s, low = pred.predict(point_coords=np.array([FOX_CLICKS[0]]), point_labels=np.array([1]), multimask_output=True)
        prev = low[int(np.argmax(s))][None]
        pts = [FOX_CLICKS[0], FOX_CLICKS[3]]
        m, s, _ = pred.predict(point_coords=np.array(pts), point_labels=np.ones(2), mask_input=prev, multimask_output=False)
        print(f"| 2 clicks + mask_input (refine loop) | {s[0]:.3f} | {(m[0] > 0).mean():.3f} | {iou(m[0] > 0, gt):.3f} | – |")
        m, s, _ = pred.predict(box=np.array([40, 245, 745, 910]), multimask_output=False)
        print(f"| box 40,245,745,910 | {s[0]:.3f} | {(m[0] > 0).mean():.3f} | {iou(m[0] > 0, gt):.3f} | – |")

        pred.set_image(gate)
        print("\n| gate click | pred-IoU | squares in mask (≥50 % covered) |\n|---|---|---|")
        for n, (x, y) in MARKS.items():
            m, s, _ = pred.predict(point_coords=np.array([[x + 6, y + 6]]), point_labels=np.array([1]), multimask_output=True)
            mb = m[int(np.argmax(s))] > 0
            inside = [k for k, (mx, my) in MARKS.items() if mb[my:my + 12, mx:mx + 12].mean() >= 0.5]
            print(f"| {n} ({x + 6},{y + 6}) | {s.max():.3f} | {''.join(inside)} |")


if __name__ == "__main__":
    main()
