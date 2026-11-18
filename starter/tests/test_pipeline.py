"""Synthetic tests for the possession pipeline: control -> holder -> gaps -> deadball.

Everything here runs on hand-built tracking arrays -- no match data needed. Each test
encodes one behaviour the design depends on:

    python starter/tests/test_pipeline.py

  1. SG smoothing: constants pass through, noise shrinks
  2. pass-by without deflection  -> a proximity run WITHOUT contact (no gain)
  3. deflected reception         -> a run WITH contact
  4. holder: opponent touch below T_CONFIRM is demoted, possession continues
  5. holder: confirmed opponent control flips, boundary at the run's start
  6. gaps: a hand-over hidden inside an occlusion is backdated to the interception frame
  7. deadball: SET PIECES anchor marks dead and survives short-run absorption
  8. deadball: a short tracker-artifact dead run is absorbed into the possession
  9. fill_unknown: holds per half, never bleeds across a halftime hole
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STARTER = os.path.dirname(HERE)
ROOT = os.path.dirname(STARTER)
sys.path.insert(0, STARTER)
sys.path.insert(0, ROOT)

from control import savgol_coeffs, smooth_ball, control_runs, DEFAULTS
from holder import hold_possession, fill_unknown
from gaps import refine_transitions
from deadball import dead_mask

FPS = 25
PITCH = (105.3, 68.0)


def make_tracking(ball_xy, player_xy=None, n_slots=2):
    """A minimal tracking dict around scripted positions (metres, NaN = missing)."""
    n = len(ball_xy)
    if player_xy is None:
        player_xy = np.full((n, n_slots, 2), np.nan, dtype=np.float32)
    return {
        "n_frames": n, "fps": FPS, "pitch": PITCH,
        "ball_xy": ball_xy.astype(np.float32),
        "player_xy": player_xy.astype(np.float32),
        "slot_team": np.array([0, 1][:n_slots], dtype=np.int8),
        "team_names": ("Home FC", "Away FC"),
    }


def test_savgol():
    coeffs = savgol_coeffs(7, 2)
    assert abs(coeffs.sum() - 1.0) < 1e-9, "weights must preserve constants"
    assert abs(np.convolve(np.ones(20), coeffs[::-1], mode="valid")[5] - 1.0) < 1e-9
    t = np.linspace(0, 1, 100)
    noisy = (np.sin(2 * np.pi * t) + np.random.default_rng(0).normal(0, 0.2, 100)).astype(np.float32)
    ball = np.stack([noisy, np.zeros(100, dtype=np.float32)], axis=1)
    smooth, _ = smooth_ball(ball, DEFAULTS)
    err_before = np.abs(ball[:, 0] - np.sin(2 * np.pi * t)).mean()
    err_after = np.abs(smooth[:, 0] - np.sin(2 * np.pi * t)).mean()
    assert err_after < err_before * 0.75, f"smoothing did not denoise: {err_after} vs {err_before}"


def straight_ball(n=200, speed=5.0):
    """Ball rolling along the x axis through mid-pitch at constant speed."""
    ball = np.zeros((n, 2), dtype=np.float64)
    ball[:, 0] = np.linspace(30, 30 + speed * n / FPS, n)
    ball[:, 1] = PITCH[1] / 2
    return ball


def test_pass_by_is_not_a_gain():
    """Ball rolls straight past an opponent parked on its path: run exists, no contact."""
    ball = straight_ball()
    players = np.full((200, 2, 2), np.nan, dtype=np.float64)
    players[:, 1, 0] = 50.0
    players[:, 1, 1] = PITCH[1] / 2
    runs, _ = control_runs(make_tracking(ball, players))
    away_runs = [r for r in runs if r["team"] == 1]
    assert away_runs, "proximity run should be detected"
    assert all(not r["contact"] for r in away_runs), \
        "straight pass-by must not carry a contact signature"


def test_deflection_is_a_gain():
    """Ball deflects off an opponent mid-path: the run carries contact."""
    n = 250
    ball = straight_ball(n)
    turn = 125
    ball[turn:, 1] = PITCH[1] / 2 + 3.0
    players = np.full((n, 2, 2), np.nan, dtype=np.float64)
    players[:, 1, 0] = ball[turn - 1, 0]
    players[:, 1, 1] = PITCH[1] / 2
    runs, _ = control_runs(make_tracking(ball, players))
    away_contact = [r for r in runs if r["team"] == 1 and r["contact"]]
    assert away_contact, "a deflected run must carry the contact signature"


def _demo_runs():
    return [
        {"team": 0, "start": 0, "end": 100, "contact": True, "max_dist": 0.5},
        {"team": 1, "start": 120, "end": 140, "contact": False, "max_dist": 0.5},
        {"team": 0, "start": 160, "end": 200, "contact": True, "max_dist": 0.5},
    ]


def test_touch_is_demoted():
    codes = hold_possession(_demo_runs(), 200)
    assert (codes[120:140] == 0).all(), "short opponent run is a touch, possession continues"
    assert (codes[100:120] == 0).all(), "gap frames hold the team in possession"
    assert (codes[160:200] == 0).all()


def test_confirmed_control_flips():
    runs = [
        {"team": 0, "start": 0, "end": 100, "contact": True, "max_dist": 0.5},
        {"team": 1, "start": 150, "end": 200, "contact": True, "max_dist": 0.5},
    ]
    codes = hold_possession(runs, 200)
    assert (codes[:100] == 0).all()
    assert (codes[150:] == 1).all()
    assert codes[99] == 0 and codes[150] == 1, "boundary lands on the opponent run's start"


def test_handover_backdated_into_occlusion():
    """Ball unobserved 100..149; away player sits on the interpolated path at frame 125."""
    ball = straight_ball(200)
    gap_start, gap_end, intercept = 100, 150, 125
    ball[gap_start:gap_end] = np.nan
    players = np.full((200, 2, 2), np.nan, dtype=np.float64)
    players[:100, 0, :] = ball[:100]
    players[intercept, 1, :] = [ball[gap_start - 1, 0] + 0.2 * (intercept - gap_start + 1),
                                PITCH[1] / 2]
    players[150:, 1, :] = ball[150:]
    tracking = make_tracking(ball, players)

    runs, evidence = control_runs(tracking)
    codes = hold_possession(runs, 200)
    assert codes[150] == 1, "holder places the flip at the first post-gap control"
    assert codes[125] == 0, "pre-refinement, the occlusion holds the old team"

    codes = refine_transitions(codes, evidence, tracking)
    assert codes[125] == 1, "boundary backdated to the interception frame"
    assert codes[124] == 0, "frames before the interception keep the old team"


def _graded(n=300):
    return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)


def test_set_piece_anchor_survives_absorption():
    n = 300
    graded = np.zeros(n, dtype=bool)
    graded[50:250] = True
    blocks = [(1, 0, n - 1)]
    events = [{"code": "SET PIECES", "start_frame": 100, "end_frame": 180,
               "tags": {"Type": "Throw-in"}}]
    codes = np.zeros(n, dtype=np.int8)
    speed = np.full(n, 5.0)
    speed[100:180] = 0.2
    evidence = {
        "smooth_xy": np.full((n, 2), 50.0), "observed": np.ones(n, dtype=bool),
        "speed": speed, "dist_min": np.full(n, 0.5),
    }
    tracking = make_tracking(straight_ball(n))
    tracking["n_frames"] = n
    params = {**__import__("deadball").DEFAULTS, "min_dead": 50}
    dead = dead_mask(codes, evidence, tracking, events, blocks, params)
    assert dead[100:180].all(), "the phase's parked-ball span marks dead"
    assert not dead[99] and not dead[180], "span is the quiet run, not the whole phase"
    assert dead[100:180].all(), "anchor survives absorption despite the artifact pass"


def test_artifact_dead_run_absorbed():
    n = 300
    graded = np.zeros(n, dtype=bool)
    graded[50:250] = True
    blocks = [(1, 0, n - 1)]
    codes = np.zeros(n, dtype=np.int8)
    evidence = {
        "smooth_xy": np.full((n, 2), 50.0),
        "observed": np.ones(n, dtype=bool),
        "speed": np.full(n, 8.0),
        "dist_min": np.full(n, 0.5),
    }
    evidence["speed"][100:130] = 0.2
    evidence["dist_min"][100:130] = 5.0
    evidence["smooth_xy"][:, 0][100:130] = 1.0
    tracking = make_tracking(straight_ball(n))
    tracking["n_frames"] = n
    params = {**__import__("deadball").DEFAULTS, "min_dead": 50, "placed_time": 20}
    dead = dead_mask(codes, evidence, tracking, [], blocks, params)
    assert not dead.any(), "a sub-min_dead dead blip inside a possession must be absorbed"


def test_fill_unknown_respects_halftime():
    codes = np.array([-1, -1, 0, 0, -1, 1, -1], dtype=np.int8)
    filled = fill_unknown(codes, [(1, 0, 3), (2, 4, 6)])
    assert (filled[:4] == 0).all(), "leading unknowns back-filled inside the block"
    assert filled[5] == 1 and filled[6] == 1, "trailing unknowns forward-filled"
    assert filled[4] == 1, "block 2 fills from its own first known value"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print("\n{} passed, {} failed".format(len(tests) - failed, failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
