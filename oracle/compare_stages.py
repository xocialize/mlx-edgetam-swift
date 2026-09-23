"""Swift-vs-PyTorch stage comparison through setImage (AB-T-0171). Reads <name>.torch.npz (run_click_oracle.py)
and <name>.swift.safetensors (edgetam-smoke --stages-out) and prints per-stage max-abs / rel error, per-prompt
IoU-score deltas, and the IoU between the two BEST masks at source resolution. Exit 1 on a gate failure.

    oracle/.venv/bin/python oracle/compare_stages.py oracle/goldens/click/gate [--markdown]
"""
import sys

import numpy as np
from safetensors.numpy import load_file

GATE = {"enc_input": 1e-3, "image_embed": 1e-3, "hrf0": 1e-3, "hrf1": 1e-3}   # rel to max |torch|


def nchw(a): return np.transpose(a, (0, 3, 1, 2))


def main():
    stem = sys.argv[1]; md = "--markdown" in sys.argv
    t = dict(np.load(f"{stem}.torch.npz")); s = load_file(f"{stem}.swift.safetensors")
    ok = True
    for k, thr in GATE.items():
        a, b = t[k], nchw(s[k])
        err = np.abs(a - b).max(); rel = err / np.abs(a).max()
        ok &= rel < thr
        print(f"  {k:12s} {str(a.shape):22s} max_abs {err:.2e}  rel {rel:.2e}  {'OK' if rel < thr else 'FAIL'}")
    rows = []
    for tag in sorted({k.split('_')[0] for k in t if k[0] in "cb" and k[1].isdigit()}):
        box = tag[0] == "b"
        low_t = t[f"{tag}_low_res"]; sc_t = t[f"{tag}_scores"]; area_t = t[f"{tag}_area"]
        low_s = np.clip(s[f"{tag}_low_res"], -32, 32)   # SAM2ImagePredictor._predict clamps its returned low-res
        iou_s = s[f"{tag}_iou"]
        tok = int(s[f"{tag}_token"][0])
        if box:   # torch = single-mask output (after dynamic stability); swift = the chosen token
            dl = np.abs(low_t[0] - low_s[tok]).max(); ds = abs(sc_t[0] - iou_s[tok]); sw_sc = [iou_s[tok]]
        else:
            dl = np.abs(low_t - low_s[1:4]).max(); ds = np.abs(sc_t - iou_s[1:4]).max(); sw_sc = iou_s[1:4]
        mt = np.load(f"{stem}.{tag}.best_mask.npy"); ms = s[f"{tag}_best_mask"] > 0.5
        miou = (mt & ms).sum() / max((mt | ms).sum(), 1)
        ok &= (dl < 5e-2) and (ds < 5e-3) and (miou > 0.99)
        where = "box" if box else "click " + ",".join(f"{v:g}" for v in t[f"{tag}_click"])
        rows.append((where, sc_t, area_t, sw_sc, dl, ds, miou))
        print(f"  {tag} {where:16s} torch " + " ".join(f"{a:.3f}→{b:.4f}" for a, b in zip(sc_t, area_t))
              + f" | swift iou " + " ".join(f"{v:.3f}" for v in sw_sc)
              + f" | low-res max_abs {dl:.2e}  iou Δ {ds:.1e}  best-mask IoU {miou:.4f}")
    if md:
        print("\n| prompt | PyTorch pred-IoU→area (3 multimask) | Swift pred-IoU | low-res Δ | best-mask IoU |\n|---|---|---|---|---|")
        for where, sc, ar, sw, dl, ds, miou in rows:
            print(f"| {where} | " + ", ".join(f"{a:.3f}→{b:.4f}" for a, b in zip(sc, ar)) + " | "
                  + ", ".join(f"{v:.3f}" for v in sw) + f" | {dl:.1e} | {miou:.4f} |")
    print("PARITY", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
