"""Per-frame tracking arrays from the tactical ATD raw export.

The possession pipeline (control.py, holder.py, gaps.py, deadball.py) works on dense numpy
arrays, not on per-line parsing, so this module reads the FIFA/EPTS text export once and
hands back:

    ball_xy      (n_frames, 2)  float32  ball position in metres, NaN where missing
    player_xy    (n_frames, 46, 2) float32  track slots in metres, NaN where missing
    slot_team    (46,)          int8     0 = home (Barcelona), 1 = away (Athletic)
    team_names   ("FC Barcelona", "Athletic Club") -- for Smart Tagging name matching

Coordinates in the raw file are normalised 0-1 on each axis; everything here is scaled to
metres with the pitch constants from games.py, so distances downstream are comparable across
axes. The speed column of the export is deliberately NOT loaded: it is unsmoothed one-frame
differences, and control.py derives ball speed from Savitzky-Golay-smoothed coordinates
instead, which is the quantity the contact tests actually need.

Parsing is a single streaming pass over ~199k lines, one to two minutes, same cost class as
starter/solution.py. frame_id in the file counts from 1; video frames count from 0, so
video_frame = frame_id - 1 (DATA.md). The speed of the loop comes from parsing only the two
fields that are used and inlining the split, as in baseline.py.

    tracking = load_tracking()
    tracking["ball_xy"].shape  # (199319, 2)
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """(slot_team int8[n_slots], team_names) from the EPTS metadata XML.

    The metadata assigns every track slot to one of two teams; the first team in the file is
    the home side (the FIFA/EPTS convention, and what kloppy reports). These slots are
    tracklets, not people -- 46 slots for 22 players, identity swapping is expected -- but
    team membership per slot is constant for the match, which is all team-level attribution
    needs.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]
    names = tuple(team.findtext("Name") for team in root.iter("Team"))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8), names


def load_tracking(game=GAME, data_dir=None):
    """Dense tracking arrays for one game's tactical source. See module docstring."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]

    slot_team, team_names = slot_teams(meta_path)
    n_slots = len(slot_team)

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

    seen = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                continue
            seen += 1

            ball_group = parts[2]
            if ball_group[0] != MISSING:
                comma = ball_group.index(",")
                ball[frame, 0] = float(ball_group[:comma]) * length
                ball[frame, 1] = float(ball_group[comma + 1:ball_group.index(",", comma + 1)]) * width

            slots = parts[1].split(GROUP_SEP)
            count = min(len(slots), n_slots)
            for slot in range(count):
                group = slots[slot]
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                players[frame, slot, 0] = float(group[:comma]) * length
                players[frame, slot, 1] = float(group[comma + 1:group.index(",", comma + 1)]) * width

    print("loaded {}: {} frames with tracking rows, ball visible on {:.1f}%".format(
        os.path.basename(raw_path), seen, 100.0 * np.sum(~np.isnan(ball[:, 0])) / n_frames))

    return {
        "n_frames": n_frames,
        "fps": fps,
        "pitch": (length, width),
        "ball_xy": ball,
        "player_xy": players,
        "slot_team": slot_team,
        "team_names": team_names,
    }
