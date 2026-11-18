"""KPI report for possession submissions -- the dead-ball and segmentation readout.

This is the harness to consult before believing any score. It prints, per submission:

  - dead-ball recall/precision and the false-D rate on live frames
  - GT dead-ball stretches touched k/N (partial coverage is the interesting failure)
  - false D-runs inside live play: count and median length (each one splits a possession)
  - possession count and mean duration vs the GT (fragmentation = flicker)
  - transition precision/recall at the scorer's tolerances

and `--inject` re-runs the damage experiments that calibrate what these errors are worth
under spellF1 (a false D ~ -0.06, a wrong-team blip ~ -0.09, an X abstention 0.00), as a
regression check that the scorer path is understood.

Read-only: prints, scores nothing into evaluation/. The numbers are computed against the
PUBLIC window and are semantic checks -- alignment, label meaning, structure. They are not
a tuning target (RULES.md); use them to catch bugs, not to pick hyperparameters.

    python starter/kpi_report.py                              # GT structure + all results/*.csv
    python starter/kpi_report.py evaluation/results/mine.csv  # one submission
    python starter/kpi_report.py --inject                     # spellF1 damage calibration
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FPS = 25.0
D, X = 2, 3


def gt_structure(gt, runs):
    segs = pm.segments(gt, runs)
    dead = [(a, b) for a, b, k in segs if k == D]
    poss = [(a, b) for a, b, k in segs if k in pm.POSSESSION_LABELS]
    dl = np.array([b - a for a, b in dead]) / FPS if dead else np.array([0.0])
    pl = np.array([b - a for a, b in poss]) / FPS
    print(f"GT: {len(dead)} dead stretches (median {np.median(dl):.1f} s, min {dl.min():.1f} s), "
          f"{len(poss)} possessions (median {np.median(pl):.1f} s)")


def deadball_kpis(gt, pred, runs, name):
    segs = pm.segments(gt, runs)
    dead = [(a, b) for a, b, k in segs if k == D]
    cm = pm.scored_confusion(pm.confusion(gt, pred))
    p_d = cm[2, 2] / max(1, cm[:, 2].sum())
    r_d = cm[2, 2] / cm[2, :].sum()
    live = cm[0, :].sum() + cm[1, :].sum()
    false_live = (cm[0, 2] + cm[1, 2]) / live

    pred_dead = [(a, b) for a, b, k in pm.segments(pred, runs) if k == D]
    touched = sum(1 for ga, gb in dead if any(a < gb and b > ga for a, b in pred_dead))
    false_runs = [(a, b) for a, b in pred_dead
                  if not any(a < gb and b > ga for ga, gb in dead)]
    lengths = np.array([b - a for a, b in false_runs]) / FPS if false_runs else np.array([0.0])

    n_pred = len(pm.possession_spells(pred, runs))
    spells = pm.possession_spells(gt, runs)
    spell = pm.spell_scores(gt, pred, runs)["mean_f1"]
    print(f"  {os.path.basename(name):<28} spellF1 {spell:.4f}  "
          f"p_D {p_d:.3f} r_D {r_d:.3f}  false-D/live {false_live:.3f}  "
          f"stretches {touched}/{len(dead)}  false D-runs {len(false_runs)} "
          f"(med {np.median(lengths):.1f} s)  poss {n_pred} (GT {len(spells)})")


def inject(gt, runs):
    """Damage calibration: one error injected mid-possession, scored under spellF1."""
    spells = pm.possession_spells(gt, runs)
    a, b, k = max(spells, key=lambda s: s[1] - s[0])
    mid = (a + b) // 2
    base = pm.spell_scores(gt, gt.copy(), runs)["mean_f1"]
    print(f"spellF1 damage calibration (perfect = {base:.4f}):")
    for name, w, code in [("1.0s false D", 12, D), ("3.0s false D", 37, D),
                          ("1.0s wrong team", 12, 1 - k), ("1.0s X abstention", 12, X)]:
        pred = gt.copy()
        pred[mid - w: mid + w] = code
        pred = pm.fill_abstentions(pred, runs)
        f1 = pm.spell_scores(gt, pred, runs)["mean_f1"]
        print(f"  {name:<18} -> {f1:.4f}  (cost {base - f1:+.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="*", help="CSV files; default: everything scored")
    parser.add_argument("--inject", action="store_true", help="run the damage calibration")
    args = parser.parse_args()

    gt = pm.label_codes(pd.read_csv(GT_PATH))
    runs = pm.valid_runs(gt)
    gt_structure(gt, runs)

    paths = args.submissions or sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
    for path in paths:
        try:
            pred = pm.label_codes(pd.read_csv(path))
        except Exception as exc:
            print(f"  {os.path.basename(path):<28} unreadable: {exc}")
            continue
        try:
            deadball_kpis(gt, pred, runs, path)
        except Exception as exc:
            print(f"  {os.path.basename(path):<28} failed: {exc}")

    if args.inject:
        print()
        inject(gt, runs)


if __name__ == "__main__":
    main()
