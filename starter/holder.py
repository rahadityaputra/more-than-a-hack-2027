"""Layer 2: control runs -> possession codes, by holding until control changes hands.

The ground truth's own semantics (CHALLENGE.md): possession stays with the last team that
had it while the ball is loose or contested, and a simple touch without control does not
change possession. So the label is not answered per frame at all -- it is a state that
changes only on a CONFIRMED opponent control run:

  merge rule (a): consecutive runs of the same team, any gap between them -> one possession
  demotion  (b): an opponent run shorter than T_CONFIRM, or without a contact signature,
                  is a touch or deflection -> possession continues through it
  transition (c): an opponent run >= T_CONFIRM frames WITH contact -> possession changes,
                  and the boundary is placed RETROACTIVELY at the run's first frame

Rule (c)'s retroactive placement is the part a majority-vote window can never do: the
decision uses T_CONFIRM frames of evidence, but the boundary lands on the evidence's start,
so confirming does not tax the metric's boundary accuracy. A frame-late boundary is free
under spellF1; a one-second-late one costs 0.35 (EVALUATION.md) -- this design pays neither.

Frames between runs (occlusions, loose balls, the 41% of frames with no ball) are held with
the current state and left as unknown (-1) only where no run ever covered them; gaps.py and
the Smart Tagging arbiter fill those.
"""

import numpy as np

DEFAULTS = {
    "t_confirm": 4,
    "confirm_long": 50,
}


def hold_possession(runs, n_frames, params=None):
    """codes int8[n_frames]: 0 = home, 1 = away, -1 = never covered by a run."""
    params = {**DEFAULTS, **(params or {})}
    t_confirm = params["t_confirm"]
    confirm_long = params["confirm_long"]
    codes = np.full(n_frames, -1, dtype=np.int8)

    state = None
    prev_end = None
    for run in runs:
        team, start, end = run["team"], run["start"], run["end"]
        old_state = state
        if state is None:
            state = team
        elif team == state:
            pass
        elif ((end - start) >= t_confirm
              and (run["contact"] or (end - start) >= confirm_long)):
            state = team
        else:
            pass
        if prev_end is not None and start > prev_end:
            codes[prev_end:start] = old_state
        codes[start:end] = state
        prev_end = end

    return codes


def fill_unknown(codes, blocks):
    """Forward-fill then back-fill unknown (-1) stretches inside each half block.

    Half blocks, not the whole array: halftime is 20k frames of nothing, and letting the
    first half's last possession bleed into the second half's first would create a phantom
    possession spanning the break.
    """
    filled = codes.copy()
    for _, first, last in blocks:
        window = filled[first:last + 1]
        known = window != -1
        if not known.any():
            continue
        carry = np.maximum.accumulate(np.where(known, np.arange(len(window)), -1))
        carry[carry < 0] = int(np.flatnonzero(known)[0])
        window[:] = window[carry]
    return filled
