"""Layer 3: resolving what happened inside lost-ball stretches.

holder.py holds the previous team through every gap -- the safe default, because a held
label inside a real possession is free under spellF1. Two refinements are worth making,
both offline (the whole match is available before any decision):

1. Backdating a transition into the gap. When the first control after a gap belongs to the
   opponent, the change of possession happened somewhere inside the gap -- the holder can
   only place it at the run's start, which is systematically late by the gap length. The
   fix: linearly interpolate the ball through the gap and move the boundary to the frame
   where an opponent player was closest to that interpolated trajectory (the interception
   point). Bounded by MAX_BACKDATE so a pathological gap cannot drag a boundary halfway
   across the pitch-timeline.

2. The Smart Tagging arbiter. Where no control evidence exists at all -- long gaps with the
   ball never reappearing -- the tagged phases are the only team evidence left. Attacking
   phases carry `Team` (314 Barcelona + 288 Athletic); defending phases carry the defending
   team, so possession is the other side. These are applied ONLY to frames the tracker left
   unknown; where tracking has spoken, it wins.

Frames the arbiter still cannot label stay unknown; fill_unknown (holder.py) then holds
the nearest known label inside each half, which is the challenge's own rule for loose balls.
"""

import numpy as np

DEFAULTS = {
    "max_backdate": 50,
}

DEFENDING_PHASES = {
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}
ATTACKING_PHASES = {
    "BUILD UP",
    "PLAYERS IN THE BOX",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "LONG BALL",
    "PROGRESSION",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
}


def refine_transitions(codes, evidence, tracking, params=None):
    """Pull opponent hand-overs back into the occlusion that hid them.

    For each transition into team T at frame t, if the ball was unobserved for the frames
    before t (bounded by max_backdate), the boundary moves to the frame in that stretch
    where a T player was closest to the interpolated ball trajectory -- the plausible
    interception moment -- and T's label starts there.
    """
    params = {**DEFAULTS, **(params or {})}
    max_backdate = params["max_backdate"]
    codes = codes.copy()
    smooth = evidence["smooth_xy"]
    observed = evidence["observed"]
    player_xy, slot_team = tracking["player_xy"], tracking["slot_team"]

    changes = np.flatnonzero((codes[1:] != codes[:-1]) & (codes[1:] >= 0) & (codes[:-1] >= 0))
    for t in changes + 1:
        new_team = codes[t]
        back = 0
        while (t - 1 - back >= 0 and not observed[t - 1 - back]
               and back < max_backdate):
            back += 1
        if back == 0:
            continue

        window = slice(t - back, t)
        delta = smooth[window, None, :] - player_xy[window]
        dist = np.linalg.norm(delta, axis=-1)
        dist = np.where(np.isnan(dist), np.inf, dist)
        gaining = slot_team == new_team
        their_dist = np.where(gaining[None, :], dist, np.inf).min(axis=1)
        best = int(np.argmin(their_dist)) + t - back
        if their_dist[best - (t - back)] < np.inf:
            codes[best:t] = new_team
    return codes


def arbitrate_with_tagging(codes, events, team_names):
    """Fill tracker-unknown frames from tagged phases that carry a Team.

    Attacking phases name the team in possession; defending phases name the team WITHOUT
    it. Phases are intervals, not possessions, so this is applied as a last-resort prior
    on unknown frames only -- wherever the tracker produced evidence, that wins.
    """
    codes = codes.copy()
    home_name, away_name = team_names
    team_codes = {(home_name or "").strip(): 0, (away_name or "").strip(): 1}
    n = len(codes)

    for event in events:
        code = (event["code"] or "").strip().upper()
        if code not in ATTACKING_PHASES and code not in DEFENDING_PHASES:
            continue
        team = team_codes.get((event["tags"].get("Team") or "").strip())
        if team is None:
            continue
        possessor = team if code in ATTACKING_PHASES else 1 - team
        start = max(0, event["start_frame"])
        end = min(n, event["end_frame"])
        unknown = codes[start:end] == -1
        codes[start:end][unknown] = possessor
    return codes
