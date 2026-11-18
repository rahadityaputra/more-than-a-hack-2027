"""Dead-ball detection from positive evidence -- never from "the ball is missing".

"Ball missing" is the baseline's D signal and it is wrong in both directions: the ball
disappears on 43% of graded frames while play is live, and it stays visible while parked
for restarts. The GT marks a frame D where the referee has the ball out of play, so the
evidence stack below tests for exactly that, in priority order:

  a. Smart Tagging `SET PIECES` phases -- a phase GATE, not a dead window. Measured on
     this match: 59 phases, median 35 s, and they spill 8-12 s past the real restart
     into live play (first build lost 0.08 spellF1 to exactly that). Inside a phase,
     dead time is the PARKED-BALL span: the longest quiet stretch (smoothed ball speed
     below v_dead, or the ball unobserved -- parked balls interpolate as parked) before
     the players put it back in play. Live frames inside the phase move the ball and
     stay live.
  b. `GOALS` -- the tagging marks a 70-250 s phase spanning goal, celebration and
     kickoff placement. Same rule: the longest quiet span between the goal and the
     kickoff phase's end is the dead time.
  c. out-of-bounds latch -- ball observed beyond the touchlines for >= 1 s stays dead
     until it re-enters in bounds (a throw-in/corner/goal-kick signature)
  d. placed ball -- SG-smoothed speed < 0.5 m/s for >= 2 s with no player within 3 m
     and the ball near a touchline, goal-line or the centre spot: a ball placed for a
     restart that the phase segmenter never tagged

A final post-pass absorbs dead runs shorter than 2 s into the surrounding possession --
a real stoppage is never that short (shortest in the public window: 5.5 s), so a shorter
D run is a tracker artifact, and a false D inside a possession costs a whole possession
of spellF1 credit. Anchor-derived spans are exempt.
"""

import numpy as np

DEFAULTS = {
    "oob_margin": 1.0,
    "oob_min_time": 25,
    "oob_exit_speed": 2.0,
    "oob_cap": 750,
    "placed_speed": 1.0,
    "placed_time": 75,
    "placed_dist": 2.5,
    "v_dead": 1.5,
    "min_dead": 50,
}


def _runs_of(mask):
    """[(start, end)] half-open maximal True stretches of a boolean array."""
    if not len(mask):
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _longest_quiet(speed, observed, start, end, params):
    """Longest parked-ball span within [start, end), or None if shorter than min_dead.

    Quiet = smoothed speed below v_dead, or the ball unobserved (a parked ball missing
    from the track interpolates as parked; a moving ball interpolates as moving). This
    is what turns a 35 s tagging PHASE into the actual stoppage inside it.
    """
    seg = (speed[start:end] < params["v_dead"]) | (~observed[start:end])
    best_a, best_len = 0, 0
    for a, b in _runs_of(seg):
        if b - a > best_len:
            best_a, best_len = a, b - a
    if best_len >= params["min_dead"]:
        return start + best_a, start + best_a + best_len
    return None


def _set_piece_dead(events, speed, observed, graded, n, params):
    """SET PIECES phases -> their parked-ball spans. Returns (mask, kickoff_ends)."""
    dead = np.zeros(n, dtype=bool)
    kickoff_ends = []
    for event in events:
        code = (event["code"] or "").strip().upper()
        if code != "SET PIECES":
            continue
        start = max(0, event["start_frame"])
        end = min(n, event["end_frame"])
        span = _longest_quiet(speed, observed, start, end, params)
        if span:
            dead[span[0]:span[1]] |= graded[span[0]:span[1]]
        type_tag = (event["tags"].get("Type") or "").strip().lower()
        if type_tag.startswith("kick off"):
            kickoff_ends.append(end)
    return dead, sorted(kickoff_ends)


def _goal_dead(events, kickoff_ends, speed, observed, graded, n, fps, params):
    """GOALS -> the quiet span between the goal and the kickoff phase's end.

    The tagging's GOALS phase covers celebration and the kickoff placement too
    (measured: 70-250 s), so the whole [goal, kickoff_end) window is the gate and the
    parked-ball span inside it is the dead time. Without a kickoff phase, cap at 90 s.
    """
    dead = np.zeros(n, dtype=bool)
    cap = 90 * fps
    for event in events:
        if (event["code"] or "").strip().upper() != "GOALS":
            continue
        start = max(0, event["start_frame"])
        after = [end for end in kickoff_ends if end >= start]
        end = min(n, after[0] if after else start + int(cap))
        span = _longest_quiet(speed, observed, start, end, params)
        if span:
            dead[span[0]:span[1]] |= graded[span[0]:span[1]]
    return dead


def _oob_dead(evidence, tracking, graded, params):
    """Latch on the ball observed beyond the touchlines; exit when it re-enters moving."""
    ball = evidence["smooth_xy"]
    observed = evidence["observed"]
    speed = evidence["speed"]
    length, width = tracking["pitch"]
    n = tracking["n_frames"]
    margin = params["oob_margin"]

    out = observed & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin)
                      | (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
    dead = np.zeros(n, dtype=bool)
    for start, end in _runs_of(out):
        if end - start < params["oob_min_time"]:
            continue
        exit_at = min(end + params["oob_cap"], n)
        for t in range(end, min(end + params["oob_cap"], n)):
            if observed[t] and not out[t]:
                exit_at = t if speed[t] >= params["oob_exit_speed"] else t
                break
        dead[start:exit_at] |= graded[start:exit_at]
    return dead


def _placed_dead(evidence, tracking, graded, params):
    """A parked ball with nobody on it, anywhere, is a ball placed for a restart.

    The discriminator that keeps this from eating live play is WHO IS NEAR, not where the
    ball is: during real dead time the taker walks away after placing the ball (nearest
    player metres away), while a keeper holding the ball or a player dribbling slowly has
    it at their feet (well under 1 m). Goal kicks sit ~5.5 m from the goal-line, so a
    restart-spot geometric gate would miss them entirely -- measured on this match, it
    fired on 416 frames while a fifth of stoppages were never tagged as phases.
    """
    speed = evidence["speed"]
    dist_min = evidence["dist_min"]
    observed = evidence["observed"]
    n = tracking["n_frames"]

    cond = observed & graded & (speed < params["placed_speed"]) & (dist_min > params["placed_dist"])
    dead = np.zeros(n, dtype=bool)
    for start, end in _runs_of(cond):
        if end - start >= params["placed_time"]:
            dead[start:end] = True
    return dead


def dead_mask(codes, evidence, tracking, events, blocks, params=None):
    """Combined dead-ball mask over graded frames, with short artifacts absorbed.

    `blocks` is games.half_video_blocks(game) -- [(half, first, last)] inclusive. Frames
    outside the blocks are never marked: they are ungraded and become X downstream.
    """
    params = {**DEFAULTS, **(params or {})}
    n = tracking["n_frames"]
    fps = tracking["fps"]

    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    set_piece, kickoff_ends = _set_piece_dead(events, evidence["speed"],
                                              evidence["observed"], graded, n, params)
    goal = _goal_dead(events, kickoff_ends, evidence["speed"],
                      evidence["observed"], graded, n, fps, params)
    dead = (set_piece
            | goal
            | _oob_dead(evidence, tracking, graded, params)
            | _placed_dead(evidence, tracking, graded, params))
    anchored = set_piece | goal

    for start, end in _runs_of(dead):
        if end - start >= params["min_dead"] or anchored[start:end].any():
            continue
        left = codes[start - 1] if start > 0 else -1
        right = codes[end] if end < n else -1
        if left >= 0 and (left == right or right < 0):
            dead[start:end] = False
    return dead
