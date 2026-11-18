# Metrica open-data tuning log

Hyperparameters for the control pipeline are tuned ONLY on Metrica's open sample matches
(RULES.md bans tuning on the provided game's public 5-minute window). This file is the
provenance record; everything here is reproducible:

```bash
uv run python starter/metrica_harness.py --match 1 --grid
uv run python starter/metrica_harness.py --match 1 --param r_pz=2.5 --param t_confirm=4
uv run python starter/metrica_harness.py --match 2 --param r_pz=2.5 --param t_confirm=4
```

## Tuning target

Metrica's tracking CSVs carry no owning-team column (`ball_owning_team_id` is 100% NaN in
kloppy), so the harness derives a **pseudo-GT from the event CSV**: consecutive possessive
events (PASS, RECOVERY, SHOT, SET PIECE, FAULT RECEIVED) of one team form one spell;
CHALLENGE / BALL LOST / BALL OUT are neutral. Match 1: 325 spells, median 9.1 s.
Match 2: 292 spells, median 11.3 s.

The headline metric is **boundary P/R/F1 at ±1 s** against pseudo-GT transitions, plus
team accuracy on pseudo-GT possession frames. Metrica game 1's tracking has 39.1% missing
ball frames — close to BAR-ATH's 41% — so occlusion behaviour transfers.

## Sweep (match 1, boundary F1)

| r_pz \ t_confirm | 3 | 4 | 5 | 8 | 10 | 15 | 25 | 40 |
|---|---|---|---|---|---|---|---|---|
| 0.6 | | | | | | 0.382 | 0.309 | 0.228 |
| 1.0 | | | | | | 0.392 | 0.341 | 0.261 |
| 1.5 | | | | | 0.552 | 0.444 | 0.374 | 0.297 |
| 2.0 | 0.854 | 0.848 | 0.798 | | | 0.515 | | |
| 2.5 | **0.854** | **0.854** | 0.841 | | | 0.562 | | |
| 3.0 | 0.828 | 0.824 | 0.822 | | | | | |
| 4.0 | 0.798 | 0.798 | | | | | | |

Reading: the Vidal-Codina defaults (r_pz 1 m, t_confirm 25 frames) under-segment badly —
141 predicted spells vs 325 in the pseudo-GT, median duration 23.7 s vs 9.1 s. The optimum
is a **plateau** at r_pz 2.0–2.5 m × t_confirm 3–4 frames (F1 0.848–0.854), with spell
counts within 3% of the pseudo-GT.

## Frozen parameters (control.py / holder.py DEFAULTS)

| param | value | evidence |
|---|---|---|
| r_pz | 2.5 m | plateau centre, match-1 grid |
| t_confirm | 4 frames | plateau centre; 25 frames costs 0.51 F1 |
| confirm_long | 50 frames | 30 → 0.838; 80 identical to 50 |
| eps_speed | 2.0 m/s | flat 0.840–0.855 over 1–3; paper value kept |
| eps_dir_deg | 30° | flat 0.852–0.856 over 20–45; paper value kept |

## Hold-out (match 2, never used for tuning)

| params | boundary P | R | F1 | team acc | spells vs GT |
|---|---|---|---|---|---|
| paper defaults | 0.576 | 0.247 | 0.346 | 0.800 | 127 vs 292 |
| frozen | 0.808 | 0.808 | **0.808** | 0.888 | 286 vs 292 |

The gain transfers across matches (0.34 → 0.81/0.85), so it is a property of the pipeline,
not of one game's event-timing quirks.
