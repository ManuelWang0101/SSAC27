# What the System Hands You, and What You Hand On

Code and data for an SSAC 2027 Research Paper Competition abstract.
Question: how does a team allocate difficult and favourable situations among its players, how does each player handle his share, and what does he hand on to the next teammate — using a single team-season (Bayer Leverkusen, Bundesliga 2023/24).

## Data
- **Raw data:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) (events, lineups, 360 freeze frames). Run `python src/fetch_data.py` to download it into `data/`. Data provided by StatsBomb (Hudl) under its open-data user agreement.
- **Derived data (included):** `data/derived/handoff_scored.csv` has one row per on-ball spell (33,315 rows, both teams). It contains the reception situation, the terminal action, the next teammate's reception, and the yardstick scores `c_in`, `c_out`, `dc`. `out/xt_grid.json` is the 12×8 expected-threat grid.

## Pipeline (run from the repo root)
| step | script | output |
|---|---|---|
| 1 | `src/fetch_data.py` | raw StatsBomb files in `data/` |
| 2 | `src/fit_xt.py` (optional; grid provided) | `out/xt_grid.json` |
| 3 | `src/handoff.py` | `out/handoff.csv`, one row per spell, linked to the next teammate's reception |
| 4 | `src/roles.py` | `out/handoff_scored.csv`: yardstick c = P(loss \| situation at reception), gradient boosting fitted on opponents only, match-grouped OOF, isotonic recalibration |
| 5 | `src/perm.py` | `out/perm_results.json`: within-(match, position group) permutation null, 500 reps |
| 6 | `src/perm_hand.py` | `out/perm_hand.json`: the same null on completed hand-offs only (Figure B) |
| 7 | `src/calib_nested.py` | `out/calib_nested.json`: fully nested calibration check on held-out opponent matches |
| 8 | `src/fig_final.py` | `out/fig_final.png/.pdf` |

To reproduce only the inference from the included derived file, copy `data/derived/handoff_scored.csv` to `out/` and run steps 5–8.

`out/SLOAN_table_players.csv` is the per-player summary for the 16 outfielders with ≥250 spells (goalkeepers excluded).

## Requirements
Python 3.10+, `pip install -r requirements.txt`.
