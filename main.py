#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 (make.fun bounty roblox-up-and-coming-forecast-01).

Approach: an empirical rank-transition forecaster, standard library only.
  * Pool every ordered pair of chart reads in the history and count where a game
    starting at rank r ends up after a given lag (destination place 1..10 exact,
    then 11-20 / 21-40 / 41+ / left the chart), smoothed toward the same
    start-rank's all-lag marginal. This is the transition kernel.
  * Adjust each game's starting rank by its recent momentum (rank change over
    the last ~24 h, half weight): climbers are treated as starting better.
  * Blend in a small weight of recent place-frequency (how often the game held
    each place in the last 10 reads): a game that has actually held #1 is
    stickier than a game that merely jumped to #1.
  * Average the kernel over every hour of the scored 72 h window (the forecast
    is one distribution scored at every hourly read, so the time-averaged
    marginal is the log-score-optimal single forecast).
  * Name, per place, the games whose averaged probability beats the share they
    would get unnamed, keep at least 25% of every place unnamed (new games
    enter the chart constantly and can only score through the unnamed share),
    and never name a probability of 0 or 1.
  * When the history carries games/v1 details, age out games whose `created`
    date puts them past the 28-day window during the forecast window.
  * Deterministic: no randomness, no network, no clock dependence beyond the
    history itself (window start is read from task.json).

Reads $MAKEFUN_INPUT_DIR (else /input): history.ndjson, task.json.
Writes $MAKEFUN_OUTPUT_DIR (else /output): forecast.json.
"""
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tuned on rolling-origin backtests over the public archive (14 origins):
# kernel smoothing 20, momentum 0.5, min unnamed residual 0.25,
# recent-frequency blend 0.15. Practice score -35.995 vs house baseline
# -43.512 (delta +7.52); rolling-origin backtest -39.05 vs -48.25.
ALPHA = 20.0
MOMENTUM = 0.5
MIN_RESIDUAL = 0.25
FREQ_BLEND = 0.15
MAX_NAMED_PER_PLACE = 60
NAME_FLOOR = 0.004
MAX_LAG_HOURS = 120.0
CHART_MAX_AGE_DAYS = 28.0

START_BANDS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
               (8, 8), (9, 9), (10, 10), (11, 12), (13, 15), (16, 20),
               (21, 25), (26, 30), (31, 40), (41, 60), (61, 200)]
LAG_BANDS = [(0, 6), (6, 12), (12, 24), (24, 36), (36, 48), (48, 60),
             (60, 72), (72, 120), (120, 10000)]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def band_of(rank):
    for i, (lo, hi) in enumerate(START_BANDS):
        if lo <= rank <= hi:
            return i
    return len(START_BANDS) - 1


def lag_band(lag_hours):
    for i, (lo, hi) in enumerate(LAG_BANDS):
        if lo <= lag_hours < hi:
            return i
    return len(LAG_BANDS) - 1


def dest_bucket(dest_rank):
    """Destination rank -> kernel bucket: 1..10 exact, 11 = 11-20, 12 = 21-40,
    13 = 41+, 14 = left the chart."""
    if dest_rank is None:
        return 14
    if dest_rank <= 10:
        return dest_rank
    if dest_rank <= 20:
        return 11
    if dest_rank <= 40:
        return 12
    return 13


def build_kernel(snaps):
    """snaps: list of (time, {universe_id: rank}) oldest first.
    Returns kernel[(start_band, lag_band)][bucket] = probability."""
    counts = defaultdict(Counter)
    prior = defaultdict(Counter)
    times = [t for t, _ in snaps]
    for i in range(len(snaps)):
        a_map = snaps[i][1]
        for j in range(i + 1, len(snaps)):
            lag = (times[j] - times[i]).total_seconds() / 3600.0
            if lag > MAX_LAG_HOURS:
                break
            lb = lag_band(lag)
            b_map = snaps[j][1]
            for uid, r in a_map.items():
                sb = band_of(r)
                dest = dest_bucket(b_map.get(uid))
                counts[(sb, lb)][dest] += 1
                prior[sb][dest] += 1
    kernel = {}
    for sb in range(len(START_BANDS)):
        p_tot = sum(prior[sb].values())
        for lb in range(len(LAG_BANDS)):
            cell = counts[(sb, lb)]
            tot = sum(cell.values())
            probs = {}
            for dest in range(1, 15):
                pv = prior[sb][dest] / p_tot if p_tot else 0.05
                probs[dest] = (cell[dest] + ALPHA * pv) / (tot + ALPHA) if tot + ALPHA > 0 else pv
            kernel[(sb, lb)] = probs
    return kernel


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window", {})
    window_from = window.get("from")

    snaps = []
    created = {}  # universe_id -> created datetime (from details, when present)
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chart = row.get("up_and_coming") or []
            rankmap = {}
            for r in chart:
                if r.get("universe_id") is not None and r.get("rank") is not None:
                    rankmap[int(r["universe_id"])] = int(r["rank"])
            if not rankmap:
                continue
            snaps.append((parse_time(row["at"]), rankmap))
            details = row.get("details") or {}
            if isinstance(details, dict):
                for key, d in details.items():
                    if isinstance(d, dict) and d.get("created"):
                        try:
                            created[int(key)] = parse_time(d["created"])
                        except (ValueError, TypeError):
                            pass
    if not snaps:
        raise SystemExit("no usable snapshots in history.ndjson")

    snaps.sort(key=lambda s: s[0])
    last_time, last_map = snaps[-1]
    if window_from:
        window_start = parse_time(window_from)
    else:
        window_start = last_time
    n_hours = int(window.get("slots", 72))

    kernel = build_kernel(snaps)

    # Momentum: rank ~24 h ago (nearest read 18-30 h before the last one).
    prev_map = {}
    for t, m in reversed(snaps[:-1]):
        lag = (last_time - t).total_seconds() / 3600.0
        if 18.0 <= lag <= 30.0:
            prev_map = m
            break
    eff_rank = {}
    for uid, r in last_map.items():
        e = float(r)
        if uid in prev_map:
            e = r + MOMENTUM * (r - prev_map[uid])
        eff_rank[uid] = max(1.0, min(200.0, e))

    # Average the kernel over every hour of the window.
    probs = defaultdict(float)  # (uid, place) -> mean probability
    for h in range(n_hours):
        lag = (window_start - last_time).total_seconds() / 3600.0 + h
        lb = lag_band(lag)
        for uid in last_map:
            row = kernel[(band_of(round(eff_rank[uid])), lb)]
            for k in range(1, places + 1):
                probs[(uid, k)] += row[k] / n_hours

    # Recent place-frequency blend (last 10 reads, linearly decayed).
    if FREQ_BLEND:
        recent = snaps[-10:]
        freq = defaultdict(float)
        w_tot = 0.0
        for idx, (_, m) in enumerate(recent):
            w = (idx + 1) / len(recent)
            w_tot += w
            for uid, r in m.items():
                if r <= places:
                    freq[(uid, r)] += w
        for key in list(probs):
            probs[key] = (1.0 - FREQ_BLEND) * probs[key] + FREQ_BLEND * freq.get(key, 0.0) / w_tot

    # Age-out: games published >28 days before a window hour leave the chart.
    # Scale by the fraction of the window the game is still eligible for.
    if created:
        for uid in list(last_map):
            c = created.get(uid)
            if c is None:
                continue
            age_days = (window_start - c).total_seconds() / 86400.0
            if age_days > CHART_MAX_AGE_DAYS - 3.0:
                frac = max(0.0, min(1.0, (CHART_MAX_AGE_DAYS - age_days) / 3.0))
                for k in range(1, places + 1):
                    probs[(uid, k)] *= frac

    # Per place: name games whose probability beats their unnamed share,
    # keeping at least MIN_RESIDUAL unnamed for games we cannot name.
    probabilities = []
    for k in range(1, places + 1):
        cand = sorted(((probs.get((uid, k), 0.0), uid) for uid in last_map),
                      key=lambda t: (-t[0], t[1]))
        named = []
        total = 0.0
        for p, uid in cand:
            if len(named) >= MAX_NAMED_PER_PLACE:
                break
            n_unnamed = max(1, len(last_map) - len(named) - 1)
            share = (1.0 - total) / n_unnamed
            if p > max(share, NAME_FLOOR):
                named.append((uid, p))
                total += p
            elif len(named) >= 10:
                break
        if total > 1.0 - MIN_RESIDUAL:
            f = (1.0 - MIN_RESIDUAL) / total
            named = [(uid, p * f) for uid, p in named]
        for uid, p in named:
            p = min(0.99, max(0.001, round(p, 6)))
            probabilities.append({"universe_id": uid, "place": k, "p": p})

    # top10: most total top-10 mass first, ties by best expected place.
    mass = defaultdict(float)
    expected = defaultdict(float)
    for (uid, k), p in probs.items():
        mass[uid] += p
        expected[uid] += k * p
    order = sorted(last_map,
                   key=lambda u: (-mass[u], expected[u] / max(mass[u], 1e-9), u))
    top10 = order[:places]
    # Pad defensively (should never trigger: last chart has ~97 games).
    if len(top10) < places:
        seen = set(top10)
        for uid in sorted(last_map):
            if uid not in seen:
                top10.append(uid)
            if len(top10) == places:
                break

    forecast = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":"), sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
