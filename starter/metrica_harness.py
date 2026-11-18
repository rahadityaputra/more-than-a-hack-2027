"""Tune the control pipeline on Metrica's open sample matches.

RULES.md bans hand labels on the provided game; Metrica's sample games are different
matches with official tracking + event CSVs, so tuning on them is legal. This harness is
the whole loop:

  1. caches the sample-game CSVs from GitHub under a scratch dir (default
     /tmp/opencode/metrica, override with MTAH_METRICA_CACHE)
  2. loads tracking via kloppy from the cached files and converts it to the dense dict
     control.py consumes (metres, per-period blocks, NaN where missing)
  3. derives a possession PSEUDO-GT from the event CSV: consecutive possessive events
     (PASS, RECOVERY, SHOT, SET PIECE, FAULT RECEIVED) of one team form one spell;
     CHALLENGE / BALL LOST / BALL OUT are neutral. Metrica's tracking has no owning-team
     column, so events are the only ground truth there is
  4. runs control -> holder -> gaps -> kinematic deadball (the exact production stack,
     fed synthetic Smart-Tagging-shaped events built from SET PIECE / goal events)
  5. reports boundary precision/recall/F1 at +/-1 s against the pseudo-GT transitions,
     team accuracy on pseudo-GT possession frames, and dead-mask frame P/R against
     BALL OUT -> next SET PIECE stretches

    uv run python starter/metrica_harness.py --match 1          # single run, full report
    uv run python starter/metrica_harness.py --match 1 --grid   # r_pz x t_confirm sweep
    uv run python starter/metrica_harness.py --match 1 --limit 20000   # quick smoke

Never points at this match's public window; nothing here reads evaluation/.
"""

import argparse
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from control import DEFAULTS as CONTROL_DEFAULTS, control_runs
from deadball import DEFAULTS as DEADBALL_DEFAULTS, dead_mask
from gaps import DEFAULTS as GAPS_DEFAULTS, refine_transitions
from holder import DEFAULTS as HOLDER_DEFAULTS, hold_possession, fill_unknown

CACHE = os.environ.get("MTAH_METRICA_CACHE", "/tmp/opencode/metrica")
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FPS = 25
POSSESSIVE = ("PASS", "RECOVERY", "SHOT", "SET PIECE", "FAULT RECEIVED")
ALL_DEFAULTS = {**CONTROL_DEFAULTS, **HOLDER_DEFAULTS, **GAPS_DEFAULTS, **DEADBALL_DEFAULTS}

TEAM_CODE = {"HOME": 0, "AWAY": 1}


def ensure_cache(match):
    """Fetch the three CSVs once; later calls are no-ops."""
    os.makedirs(CACHE, exist_ok=True)
    names = tuple("Sample_Game_{}_{}".format(match, part) for part in
                  ("RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv",
                   "RawEventsData.csv"))
    paths = []
    for name in names:
        path = os.path.join(CACHE, name)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            url = "{}/Sample_Game_{}/{}".format(BASE, match, name)
            print("downloading {} ...".format(name))
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def load_tracking(home_csv, away_csv):
    """kloppy Metrica CSVs -> the dense tracking dict the pipeline consumes."""
    from kloppy import metrica as km
    ds = km.load_tracking_csv(home_csv, away_csv)
    df = ds.to_df()
    pitch = ds.metadata.pitch_dimensions
    length, width = float(pitch.pitch_length), float(pitch.pitch_width)

    n = len(df)
    slots = []
    for side, team in (("home", 0), ("away", 1)):
        ids = sorted(int(re.match(r"{}_(\d+)_x$".format(side), col).group(1))
                     for col in df.columns
                     if col.startswith(side + "_") and col.endswith("_x"))
        slots.extend(("{}/{}".format(side, i), team) for i in ids)
    n_slots = len(slots)

    player_xy = np.full((n, n_slots, 2), np.nan, dtype=np.float32)
    for slot, (prefix, _) in enumerate(slots):
        side, pid = prefix.split("/")
        player_xy[:, slot, 0] = df["{}_{}_x".format(side, pid)] * length
        player_xy[:, slot, 1] = df["{}_{}_y".format(side, pid)] * width

    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    ball_xy[:, 0] = df["ball_x"] * length
    ball_xy[:, 1] = df["ball_y"] * width

    periods = df["period_id"].to_numpy()
    blocks = []
    for period in (1, 2):
        rows = np.flatnonzero(periods == period)
        if len(rows):
            blocks.append((period, int(rows[0]), int(rows[-1])))
    if not blocks:
        blocks = [(1, 0, n - 1)]

    print("loaded metrica match {}: {} frames, blocks {}, ball observed {:.1f}%".format(
        os.path.basename(home_csv).split("_")[2], n, blocks,
        100.0 * np.sum(~np.isnan(ball_xy[:, 0])) / n))
    return {
        "n_frames": n, "fps": FPS, "pitch": (length, width),
        "ball_xy": ball_xy, "player_xy": player_xy,
        "slot_team": np.array([team for _, team in slots], dtype=np.int8),
        "team_names": ("Home", "Away"),
    }, blocks


def load_events(events_csv):
    """Events CSV with normalised upper-case Type/Subtype/Team."""
    ev = pd.read_csv(events_csv)
    for col in ("Type", "Subtype", "Team"):
        ev[col] = ev[col].astype(str).str.strip().str.upper()
        ev.loc[ev[col].isin(("NAN", "")), col] = None
    return ev


def tagging_events(ev, fps):
    """SET PIECE / goal events recast as Smart-Tagging-shaped dicts for dead_mask.

    Metrica timestamps the restart itself, so the anchor interval is the ~8 s before
    it -- an approximation of the tagged dead window, documented here and nowhere else.
    """
    out = []
    for rec in ev.to_dict("records"):
        try:
            start = int(rec["Start Frame"])
        except (TypeError, ValueError):
            continue
        if start <= 0:
            continue
        subtype = rec["Subtype"] or ""
        if rec["Type"] == "SET PIECE":
            out.append({"code": "SET PIECES", "start_frame": max(0, start - 8 * fps),
                        "end_frame": start, "tags": {"Type": subtype}})
        elif rec["Type"] == "SHOT" and subtype.startswith("GOAL"):
            out.append({"code": "GOALS", "start_frame": start,
                        "end_frame": start + 90 * fps, "tags": {"Type": "goal"}})
    return out


def pseudo_spells(ev, n_frames):
    """[(start, end, team)] possession spells from the event chain.

    Possessive events only; a spell ends where the next possessive event of the OTHER
    team starts (events extend to the next boundary, so spells tile the match).
    """
    recs = sorted((r for r in ev.to_dict("records")
                   if r["Type"] in POSSESSIVE and r["Team"] in TEAM_CODE
                   and r["Start Frame"] is not None),
                  key=lambda r: int(r["Start Frame"]))
    if not recs:
        return []
    starts = np.array([int(r["Start Frame"]) for r in recs])
    teams = np.array([TEAM_CODE[r["Team"]] for r in recs])
    change = np.r_[True, teams[1:] != teams[:-1]]
    b_start, b_team = starts[change], teams[change]
    ends = np.r_[b_start[1:], n_frames]
    return [(int(s), int(min(e, n_frames)), int(t)) for s, e, t in zip(b_start, ends, b_team)
            if s < n_frames and min(e, n_frames) - s >= 10]


def pseudo_dead(ev, n_frames, fps):
    """Dead stretches: BALL OUT -> next SET PIECE (capped at 120 s)."""
    outs = sorted(int(r["Start Frame"]) for r in ev.to_dict("records")
                  if r["Type"] == "BALL OUT" and r["Start Frame"] is not None)
    pieces = sorted(int(r["Start Frame"]) for r in ev.to_dict("records")
                    if r["Type"] == "SET PIECE" and r["Start Frame"] is not None)
    dead = np.zeros(n_frames, dtype=bool)
    for out in outs:
        after = [p for p in pieces if p > out]
        end = min(after[0] if after else out + 120 * fps, out + 120 * fps, n_frames)
        dead[out:end] = True
    return dead


def run_pipeline(tracking, tagevents, blocks, params):
    """The production stack, exactly as build_submission.build drives it."""
    n = tracking["n_frames"]
    runs, evidence = control_runs(tracking, params)
    codes = hold_possession(runs, n, params)
    codes = refine_transitions(codes, evidence, tracking, params)
    codes = fill_unknown(codes, blocks)
    dead = dead_mask(codes, evidence, tracking, tagevents, blocks, params)
    return codes, dead, runs, evidence


def transitions(codes, blocks):
    """Frames where possession changes team inside each block."""
    out = []
    for _, first, last in blocks:
        seg = codes[first:last + 1]
        ch = np.flatnonzero((seg[1:] != seg[:-1])
                            & np.isin(seg[1:], (0, 1)) & np.isin(seg[:-1], (0, 1)))
        out.extend((ch + first + 1).tolist())
    return sorted(out)


def boundary_scores(pred, gt, tol):
    """Greedy nearest matching within +-tol frames: precision, recall, f1."""
    used = set()
    matched = 0
    for g in gt:
        cands = [p for p in pred if p not in used and abs(p - g) <= tol]
        if cands:
            used.add(min(cands, key=lambda p: abs(p - g)))
            matched += 1
    precision = matched / max(1, len(pred))
    recall = matched / max(1, len(gt))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return precision, recall, f1


def spell_stats(spells):
    lengths = np.array([e - s for s, e, _ in spells]) / FPS
    return len(spells), float(np.median(lengths)) if len(lengths) else 0.0


def report(codes, dead, blocks, tracking, gt_labels, gt_spells, gt_dead, tol):
    """The structural readout that decides hyperparameters."""
    n = tracking["n_frames"]
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    pred_tr = transitions(codes, blocks)
    gt_tr = [s for s, _, _ in gt_spells[1:]] if gt_spells else []
    precision, recall, f1 = boundary_scores(pred_tr, gt_tr, tol)

    on_poss = graded & (gt_labels >= 0)
    agree = float((codes[on_poss] == gt_labels[on_poss]).mean()) if on_poss.any() else 0.0
    coverage = float(((codes[on_poss] >= 0) & ~dead[on_poss]).mean()) if on_poss.any() else 0.0

    d_pred, d_gt = dead & graded, gt_dead & graded
    tp = int((d_pred & d_gt).sum())
    dead_p = tp / max(1, int(d_pred.sum()))
    dead_r = tp / max(1, int(d_gt.sum()))

    n_pred, med_pred = spell_stats(_pred_spells(codes, blocks))
    n_gt, med_gt = spell_stats(gt_spells)

    return {
        "f1": f1, "precision": precision, "recall": recall,
        "team_acc": agree, "poss_coverage": coverage,
        "dead_p": dead_p, "dead_r": dead_r,
        "dead_frac": float(d_pred.sum()) / max(1, int(graded.sum())),
        "n_pred": n_pred, "med_pred": med_pred, "n_gt": n_gt, "med_gt": med_gt,
        "n_transitions": len(pred_tr), "n_gt_transitions": len(gt_tr),
    }


def _pred_spells(codes, blocks):
    """Spells from the predicted codes: maximal same-team stretches per block."""
    out = []
    for _, first, last in blocks:
        seg = codes[first:last + 1]
        edges = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
        for a, b in zip(edges[:-1], edges[1:]):
            team = int(seg[a])
            if team in (0, 1) and b - a >= 10:
                out.append((int(a + first), int(b + first), team))
    return out


def print_report(tag, m, elapsed):
    print("{}  boundary P {:.3f} R {:.3f} F1 {:.3f} | team acc {:.3f} | "
          "poss cov {:.3f} | dead P {:.3f} R {:.3f} frac {:.3f} | spells {} "
          "(med {:.1f}s) vs GT {} (med {:.1f}s) | {:.1f}s".format(
              tag, m["precision"], m["recall"], m["f1"], m["team_acc"],
              m["poss_coverage"], m["dead_p"], m["dead_r"], m["dead_frac"],
              m["n_pred"], m["med_pred"], m["n_gt"], m["med_gt"], elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--limit", type=int, default=None,
                        help="kloppy frame limit for quick smokes (period 1 only)")
    parser.add_argument("--grid", action="store_true",
                        help="sweep r_pz x t_confirm instead of a single run")
    parser.add_argument("--rpz", default="0.6,1.0,1.5",
                        help="comma list of r_pz values for --grid")
    parser.add_argument("--tcf", default="15,25,40",
                        help="comma list of t_confirm values for --grid")
    parser.add_argument("--param", action="append", default=[],
                        help="hyperparameter override key=value, repeatable")
    parser.add_argument("--tol", type=int, default=FPS,
                        help="boundary match tolerance in frames (default 25 = 1 s)")
    args = parser.parse_args()

    home_csv, away_csv, events_csv = ensure_cache(args.match)
    tracking, blocks = load_tracking(home_csv, away_csv)
    if args.limit:
        n = min(args.limit, tracking["n_frames"])
        for key in ("ball_xy",):
            tracking[key] = tracking[key][:n]
        tracking["player_xy"] = tracking["player_xy"][:n]
        tracking["n_frames"] = n
        blocks = [(p, f, min(l, n - 1)) for p, f, l in blocks if f < n]

    ev = load_events(events_csv)
    tagevents = tagging_events(ev, FPS)
    gt_spells = pseudo_spells(ev, tracking["n_frames"])
    gt_dead = pseudo_dead(ev, tracking["n_frames"], FPS)
    gt_labels = np.full(tracking["n_frames"], -1, dtype=np.int8)
    for s, e, t in gt_spells:
        gt_labels[s:e] = t

    overrides = dict(ALL_DEFAULTS)
    for pair in args.param:
        key, value = pair.split("=", 1)
        if key not in overrides:
            raise SystemExit("unknown parameter {!r}; have {}".format(key, sorted(overrides)))
        overrides[key] = type(overrides[key])(value)

    print("pseudo-GT: {} spells (median {:.1f}s), dead {:d} frames from {} BALL OUT "
          "events, {} tagging anchors".format(
              len(gt_spells),
              np.median([e - s for s, e, _ in gt_spells]) / FPS if gt_spells else 0,
              int(gt_dead.sum()), int((ev["Type"] == "BALL OUT").sum()), len(tagevents)))

    if args.grid:
        rpz_values = [float(v) for v in args.rpz.split(",")]
        tcf_values = [int(v) for v in args.tcf.split(",")]
        print("\ngrid (r_pz x t_confirm), confirm_long fixed at {}:".format(
            overrides["confirm_long"]))
        print("  r_pz  t_cf   F1     P      R      teamacc  dead%   spells/GT")
        best = None
        for r_pz in rpz_values:
            for t_confirm in tcf_values:
                params = {**overrides, "r_pz": r_pz, "t_confirm": t_confirm}
                t0 = time.time()
                codes, dead, runs, _ = run_pipeline(tracking, tagevents, blocks, params)
                m = report(codes, dead, blocks, tracking, gt_labels, gt_spells,
                           gt_dead, args.tol)
                print("  {:.1f}   {:3d}   {:.3f}  {:.3f}  {:.3f}  {:.3f}    "
                      "{:.3f}  {:4d}/{}".format(
                          r_pz, t_confirm, m["f1"], m["precision"], m["recall"],
                          m["team_acc"], m["dead_frac"], m["n_pred"], m["n_gt"]))
                if best is None or m["f1"] > best[0]:
                    best = (m["f1"], r_pz, t_confirm)
        print("\nbest: r_pz {:.1f} t_confirm {} -> F1 {:.3f}".format(best[1], best[2], best[0]))
        return

    t0 = time.time()
    codes, dead, runs, _ = run_pipeline(tracking, tagevents, blocks, overrides)
    m = report(codes, dead, blocks, tracking, gt_labels, gt_spells, gt_dead, args.tol)
    print("\ncontrol runs: {} (home {}, away {}, contact {})".format(
        len(runs), sum(1 for r in runs if r["team"] == 0),
        sum(1 for r in runs if r["team"] == 1),
        sum(1 for r in runs if r["contact"])))
    print_report("RESULT", m, time.time() - t0)
    subtypes = sorted({r["Subtype"] for r in ev.to_dict("records")
                       if r["Type"] == "SET PIECE" and r["Subtype"]})
    print("set-piece subtypes present:", subtypes)


if __name__ == "__main__":
    main()
