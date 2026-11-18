"""End-to-end possession submission: tracking -> control runs -> holder -> gaps -> dead ball.

The whole pipeline, deterministic, no manual steps:

    python starter/build_submission.py                     # writes evaluation/results/control_holder_v1.csv
    python starter/build_submission.py --param t_confirm=30 --param r_pz=0.8
    python starter/build_submission.py --name my_variant   # different output name

then validate and score:

    python evaluation/validate_submission.py evaluation/results/control_holder_v1.csv
    python evaluation/run_evaluation.py                    # public 5-minute smoke test only

The five hyperparameters (r_pz, r_dz, eps_speed, eps_dir_deg, t_confirm) carry
Vidal-Codina defaults and are tuned ONLY on Metrica's open matches (metrica_harness.py) --
never against this match's public window, per RULES.md.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks
from load_smart_tagging import load as load_smart_tagging
from load_tracking import load_tracking
from control import DEFAULTS as CONTROL_DEFAULTS, control_runs
from holder import DEFAULTS as HOLDER_DEFAULTS, hold_possession, fill_unknown
from gaps import DEFAULTS as GAPS_DEFAULTS, refine_transitions, arbitrate_with_tagging
from deadball import DEFAULTS as DEADBALL_DEFAULTS, dead_mask

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
LABELS = np.array(["H", "A", "D", "X"])

ALL_DEFAULTS = {**CONTROL_DEFAULTS, **HOLDER_DEFAULTS, **GAPS_DEFAULTS, **DEADBALL_DEFAULTS}


def build(params=None, game=GAME, data_dir=None, verbose=True):
    """labels int8[n_frames] over the four classes H/A/D/X, plus diagnostics."""
    params = {**ALL_DEFAULTS, **(params or {})}
    blocks = half_video_blocks(game)
    tracking = load_tracking(game, data_dir)
    n = tracking["n_frames"]
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    runs, evidence = control_runs(tracking, params)
    if verbose:
        print("control runs: {} (home {}, away {})".format(
            len(runs),
            sum(1 for r in runs if r["team"] == 0),
            sum(1 for r in runs if r["team"] == 1)))

    codes = hold_possession(runs, n, params)
    codes = refine_transitions(codes, evidence, tracking, params)
    events = load_smart_tagging(game, data_dir)
    codes = arbitrate_with_tagging(codes, events, tracking["team_names"])
    codes = fill_unknown(codes, blocks)

    dead = dead_mask(codes, evidence, tracking, events, blocks, params)

    labels = np.full(n, 3, dtype=np.int8)
    alive = graded & ~dead
    labels[alive & (codes == 0)] = 0
    labels[alive & (codes == 1)] = 1
    labels[graded & dead] = 2
    for _, first, last in blocks:
        window = labels[first:last + 1]
        unknown = (window == 3) & graded[first:last + 1]
        if unknown.any():
            known = window != 3
            if known.any():
                carry = np.maximum.accumulate(np.where(known, np.arange(len(window)), -1))
                carry[carry < 0] = int(np.flatnonzero(known)[0])
                window[unknown] = window[carry]

    if verbose:
        counts = {name: int(np.sum(labels == i)) for i, name in enumerate(LABELS)}
        print("labels: " + "  ".join("{} {:>6}".format(k, v) for k, v in counts.items()))
    return labels, {"runs": runs, "codes": codes, "dead": dead, "graded": graded}


def write_submission(path, labels):
    """A frame,label CSV -- the whole submission format, in four lines."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, LABELS[label])
                          for frame, label in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="control_holder_v1", help="output file stem")
    parser.add_argument("--param", action="append", default=[],
                        help="hyperparameter override, key=value, repeatable")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    params = dict(ALL_DEFAULTS)
    for override in args.param:
        key, value = override.split("=", 1)
        if key not in params:
            raise SystemExit("unknown parameter {!r}; have {}".format(key, sorted(params)))
        params[key] = type(params[key])(value)

    labels, _ = build(params)
    path = write_submission(os.path.join(args.out, "{}.csv".format(args.name)), labels)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    print("validate:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
