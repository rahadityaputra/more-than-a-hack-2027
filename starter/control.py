"""Layer 1: control evidence -- who demonstrably has the ball, frame by frame.

This is the Vidal-Codina et al. 2022 scheme ("Automatic event detection in football using
tracking data", Sports Engineering) reduced to the piece this challenge needs:

  - a POSSESSION ZONE (PZ): radius R_PZ around a player; ball inside it -> that player's
    team is the candidate possessor for the frame
  - a DUEL ZONE (DZ): radius R_DZ around the ball; an opponent inside it marks the frame
    contested (v1 only records it -- the holder's confirm rule is what resolves contests)
  - a CONTACT SIGNATURE: a candidate run only evidences a control gain if the ball's
    direction or speed changed materially during it. A ball passing near a player without
    deflecting is no touch; CHALLENGE.md's "a simple touch of the ball without control"
    does not change possession, and this is the arithmetic version of that rule.

The output is NOT per-frame labels. It is control RUNS -- (team, start, end, contact)
intervals -- which holder.py merges into possessions with an asymmetric confirm rule.

Ball coordinates are Savitzky-Golay smoothed before speeds/directions are derived (window
7 frames = 0.28 s, second order), the paper's own noise treatment; the coefficients are
built in pure numpy. Gaps in the ball track are linearly interpolated before smoothing and
flagged, so downstream code can tell observed from filled frames.

Defaults are tuned on Metrica's open matches via metrica_harness.py --grid (r_pz 2.5 m,
t_confirm 4 frames; hold-out validated on a second match) -- never on this match's
public window.
"""

import numpy as np

DEFAULTS = {
    "r_pz": 2.5,
    "r_dz": 1.0,
    "eps_speed": 2.0,
    "eps_dir_deg": 30.0,
    "sg_window": 7,
    "sg_order": 2,
    "min_run": 5,
}


def savgol_coeffs(window, order):
    """Savitzky-Golay smoothing weights for a centred window, pure numpy.

    The weights are the row of the pseudo-inverse of the Vandermonde matrix that evaluates
    the least-squares polynomial at the window centre, so convolving with them fits a
    degree-`order` polynomial over the window and reads it at the middle frame.
    """
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    vandermonde = np.vander(x, order + 1, increasing=True)
    return np.linalg.pinv(vandermonde)[0]


def interpolate_nan(values):
    """Linearly fill NaN stretches of a 1-D series from the valid points around them."""
    valid = ~np.isnan(values)
    if not valid.any():
        return values.copy(), np.zeros(len(values), dtype=bool)
    indices = np.arange(len(values))
    filled = np.interp(indices, indices[valid], values[valid].astype(np.float64))
    return filled.astype(values.dtype), valid


def smooth_ball(ball_xy, params):
    """(smoothed_xy, observed_mask): SG-smoothed ball coordinates in metres.

    NaN gaps are linearly interpolated first (the SG kernel cannot span holes), smoothed,
    and the observation mask is kept so downstream code can distinguish measured from
    filled frames -- a frame counts as observed if EITHER axis was measured.
    """
    window = params["sg_window"]
    order = params["sg_order"]
    coeffs = savgol_coeffs(window, order)
    half = window // 2
    n = len(ball_xy)
    out = np.empty_like(ball_xy, dtype=np.float64)
    observed = np.zeros(n, dtype=bool)
    for axis in range(2):
        filled, valid = interpolate_nan(ball_xy[:, axis])
        padded = np.concatenate((np.full(half, filled[0]), filled, np.full(half, filled[-1])))
        out[:, axis] = np.convolve(padded, coeffs[::-1], mode="valid")
        observed |= valid
    return out, observed


def unit(vectors):
    """Row-wise unit vectors; zero rows stay zero (never produce a phantom direction)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.where(norms > 1e-6, norms, 1.0)
    return vectors / safe


def frame_evidence(tracking, params):
    """(candidate_team, dist_min, contested, smoothed_xy, speed, velocity, observed).

    Per frame, over all track slots: the nearest player overall, whether that player is
    inside the PZ, whether an opponent of theirs is inside the DZ (contested), and ball
    kinematics from the smoothed track. Chunked so the 199k x 46 distance matrix never
    materialises whole.
    """
    ball, observed = smooth_ball(tracking["ball_xy"], params)
    fps = tracking["fps"]
    velocity = np.gradient(ball, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=-1)

    player_xy = tracking["player_xy"]
    slot_team = tracking["slot_team"]
    n = tracking["n_frames"]
    r_pz, r_dz = params["r_pz"], params["r_dz"]

    candidate = np.full(n, -1, dtype=np.int8)
    dist_min = np.full(n, np.inf, dtype=np.float32)
    contested = np.zeros(n, dtype=bool)

    chunk = 50000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        delta = ball[start:stop, None, :] - player_xy[start:stop]
        dist = np.linalg.norm(delta, axis=-1)
        dist = np.where(np.isnan(dist), np.inf, dist)

        nearest = np.argmin(dist, axis=1)
        best = dist[np.arange(stop - start), nearest]
        team_of_slot = slot_team[nearest]

        opponent = slot_team[None, :] != team_of_slot[:, None]
        opp_dist = np.where(opponent, dist, np.inf).min(axis=1)

        in_pz = best <= r_pz
        candidate[start:stop] = np.where(in_pz, team_of_slot, -1)
        dist_min[start:stop] = best
        contested[start:stop] = in_pz & (opp_dist <= r_dz)

    return {
        "candidate": candidate,
        "dist_min": dist_min,
        "contested": contested,
        "smooth_xy": ball,
        "speed": speed,
        "velocity": velocity,
        "observed": observed,
    }


def control_runs(tracking, params=None):
    """[ControlRun] maximal same-team PZ stretches over observed ball frames.

    ControlRun = (team, start, end, contact, max_dist) with half-open [start, end), team in
    {0, 1}. `contact` is the paper's gain signature -- ball direction change or speed jump
    inside the run -- and decides whether holder.py may treat the run as a possession
    change or only as a touch. Runs below `min_run` frames without contact are dropped as
    tracker noise. Returns (runs, evidence) where evidence is frame_evidence's dict, kept
    for deadball.py (placed-ball test) and gaps.py (interpolated trajectory).
    """
    params = {**DEFAULTS, **(params or {})}
    evidence = frame_evidence(tracking, params)
    n = tracking["n_frames"]
    cos_eps = np.cos(np.radians(params["eps_dir_deg"]))

    candidate, observed, dist_min = evidence["candidate"], evidence["observed"], evidence["dist_min"]
    speed, velocity = evidence["speed"], evidence["velocity"]

    runs = []
    start = None
    team = -1
    for t in range(n + 1):
        active = t < n and observed[t] and candidate[t] >= 0
        if active and start is not None and candidate[t] == team:
            continue
        if start is not None:
            end = t
            length = end - start
            v_in = velocity[max(start - 1, 0)]
            v_out = velocity[end - 1]
            dir_change = (np.linalg.norm(v_in) > 1e-6 and np.linalg.norm(v_out) > 1e-6
                          and float(np.dot(unit(v_in), unit(v_out))) < cos_eps)
            speed_jump = (end - start) > 1 and float(
                np.max(np.abs(speed[start + 1:end] - speed[start:end - 1]))) > params["eps_speed"]
            contact = bool(dir_change or speed_jump)
            if length >= params["min_run"] or contact:
                runs.append({
                    "team": int(team), "start": start, "end": end,
                    "contact": contact, "max_dist": float(np.max(dist_min[start:end])),
                })
        if active:
            start, team = t, int(candidate[t])
        else:
            start, team = None, -1

    return runs, evidence
