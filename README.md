# Roblox Up-and-Coming top-10 forecast

Entry for make.fun bounty `roblox-up-and-coming-forecast-01`: forecast Roblox's
Up-and-Coming chart top 10, place by place, for the 72 hours after entries close.

## Approach

An empirical rank-transition forecaster (Python 3.13 standard library only, no
network, deterministic):

- Pool every ordered pair of chart reads in `/input/history.ndjson` and count
  where a game starting at rank *r* ends up after a given lag — destination
  place 1–10 exact, then 11–20 / 21–40 / 41+ / left the chart — smoothed
  toward the same start rank's all-lag marginal.
- Adjust each game's starting rank by its recent momentum (rank change over
  the last ~24 h, half weight) and blend in a small weight of recent
  place-frequency (how often the game held each place in the last 10 reads).
- Average the transition kernel over every hour of the 72 h window: the
  forecast is one distribution scored at every hourly read, so the
  time-averaged marginal is the log-score-optimal single forecast.
- Per place, name the games whose probability beats the share they would get
  unnamed, keep ≥25% of every place unnamed (new games enter the chart
  constantly and can only score through the unnamed share), and never output
  a probability of 0 or 1. Games whose `created` date ages them out of the
  28-day window mid-window are down-weighted.

Practice score (public archive, 2026-11-07 → 2026-11-10 window): **-35.99**
vs the house "no-change" baseline **-43.51** (typical probability per place
0.027 vs 0.013; 6.4 top-10 hits, 48.0 agreement points).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```
