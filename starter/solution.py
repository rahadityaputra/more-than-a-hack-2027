"""Generate possession-class submissions from tracking and smart-tagging data."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_smart_tagging  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

VARIANTS = (
    ("solution_smooth", 0),
    ("solution_smooth25", 25),
    ("solution_smooth50", 50),
    ("solution_smooth122", 122),
    ("solution_smooth125", 125),
)

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N" 

HOME, AWAY, NOBODY = 0, 1, -1
LABELS = np.array(["H", "A", "D"])

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

DEFENDING_PHASES = {
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
    "DEFENDING IN ATTACKING THIRD",
}

def team_of_each_track(meta_path):
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    The ATD metadata lists both teams and assigns every track slot to one of them. The first team
    in the file is the home side -- the FIFA/EPTS convention, and the one kloppy follows when it
    reports metadata.teams. For this match that is Barcelona, which DATA.md states independently.

    Note what these slots are NOT: there are 46 of them for 22 players on the pitch, they are
    named "Track 1".."Track 46", and their ShirtNumber is just the slot index. A slot is a
    tracklet, not a person, and it does not follow one player through the match.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away) club names, for printing. The submission wants a side, not a club."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def nearest_player_labels(raw_path, team_of_track, n_frames, pitch):
    """Per-frame team of the player closest to the ball. NOBODY where there is no answer.

    Returned as codes rather than letters because the smoothing below counts them.

    Coordinates are normalised 0-1 and the pitch is not square, so both axes are scaled to metres
    before distances are compared. Skipping that would stretch every distance along the short axis
    and quietly pick the wrong player near the touchlines.
    """
    length, width = pitch
    out = np.full(n_frames, NOBODY, dtype=np.int8)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball = parts[2]
            if ball[0] == MISSING:
                # No ball, no answer. 41% of graded frames land here -- see DATA.md. Leaving them
                # as NOBODY rather than guessing is what makes the D class mean anything at all.
                continue

            comma = ball.index(",")
            ball_x = float(ball[:comma]) * length
            ball_y = float(ball[comma + 1:ball.index(",", comma + 1)]) * width

            best, best_team = None, NOBODY
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - ball_x
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - ball_y
                distance = dx * dx + dy * dy
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]

            out[int(parts[0]) - 1] = best_team  # frame_id counts from 1, submissions from 0
    return out


def smart_tagging_labels(codes, events, home_name, away_name):
    labels = np.full(len(codes), NOBODY, dtype=np.int8)
    team_codes = {home_name: HOME, away_name: AWAY}
    attacking = np.zeros(len(codes), dtype=bool)
    defending = np.full(len(codes), NOBODY, dtype=np.int8)
    set_pieces = np.zeros(len(codes), dtype=bool)

    for event in events:
        code = (event["code"] or "").strip().upper()
        start = max(0, event["start_frame"])
        end = min(len(codes), event["end_frame"])
        if start >= end:
            continue

        if code in ATTACKING_PHASES:
            attacking[start:end] = True
        elif code in DEFENDING_PHASES:
            team = team_codes.get((event["tags"].get("Team") or "").strip())
            if team is not None:
                defending[start:end] = 1 - team
        elif code == "SET PIECES":
            set_pieces[start:end] = True

    last_seen = np.full(len(codes), NOBODY, dtype=np.int8)
    last = NOBODY
    for frame, code in enumerate(codes):
        last_seen[frame] = last
        if code != NOBODY:
            last = code

    missing = codes == NOBODY
    alive = missing & ~set_pieces
    labels[missing & set_pieces] = NOBODY
    labels[alive & (defending != NOBODY)] = defending[alive & (defending != NOBODY)]
    labels[alive & (defending == NOBODY) & attacking] = last_seen[
        alive & (defending == NOBODY) & attacking]
    return labels


def smooth(codes, window):
    """Majority vote over a centred window of +/- `window` frames. D where nobody has the ball.

    Computed with cumulative sums, so the cost does not grow with the window -- a 125-frame window
    over 199319 frames is the same work as a 1-frame one.

    Ties go to the home side. With an odd-sized window over two classes a tie needs the two counts
    to be exactly equal, which is rare and never worth a coin flip.
    """
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def write_submission(path, labels):
    """A frame,label CSV -- the whole submission format, in four lines."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def build(game=GAME, data_dir=None):
    """Per-frame team codes for the whole video. The expensive half, run once, smoothed many ways."""
    meta_path, raw_path, smart_tagging_path = require(
        ["atd_meta", "atd_raw", "smart_tagging"], game, data_dir)
    n_frames = source_frames("tactical", game)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    codes = nearest_player_labels(raw_path, team_of_each_track(meta_path), n_frames,
                                  GAMES[game]["pitch"])
    events = load_smart_tagging(game, data_dir)
    tagged = smart_tagging_labels(codes, events, home, away)
    codes[codes == NOBODY] = tagged[codes == NOBODY]
    seen = int(np.sum(codes != NOBODY))
    print("  ball located on {} of {} frames ({:.1f}%)\n".format(
        seen, n_frames, 100.0 * seen / n_frames))
    return codes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, nargs="+",
                        help="half-widths to write instead of the four published variants")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    codes = build()
    variants = ([("baseline_smooth{}".format(w) if w else "baseline_raw", w)
                 for w in args.window] if args.window else list(VARIANTS))

    for name, window in variants:
        labels = smooth(codes, window)
        path = write_submission(os.path.join(args.out, "{}.csv".format(name)), labels)
        counts = {label: int(np.sum(labels == label)) for label in LABELS}
        print("{:<20} window +/-{:<4} H {:>6}  A {:>6}  D {:>6}".format(
            name, window, counts["H"], counts["A"], counts["D"]))
        print("{:<20} -> {}".format("", os.path.relpath(path, ROOT)))

    print("\nNow score them:  python evaluation/run_evaluation.py")
