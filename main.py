"""Forecast GTA VI (PS5) trophy earn rates on Exophase, one week after launch.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01.

The program is run once per case (`python3 /bot/main.py`, Python 3.13,
standard library only, no network). It reads every `<id>.in` in the input
directory ($MAKEFUN_INPUT_DIR, else /input) and writes one `<id>.out` per
case to the output directory ($MAKEFUN_OUTPUT_DIR, else /output):
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, five
quantiles of trophy n's earn rate (percent) at the case's target.

Model: an earn rate is a ratio, earned / tracked, so the two sides are
forecast separately and divided at the target.

* Tracked players N(t).  Exophase refreshes its counts on its own schedule,
  so N moves in jumps between reads.  Over the reads in the case history we
  fit log N against log(1 + hours since the first read / 12) by least
  squares and extrapolate that curve to the target.  When the history shows
  no growth at all (the launch-night practice cases: every read frozen at
  638 players), we fall back on the one refresh the house documented in the
  kit rules: the first refresh, at 2026-11-19T04:30:00Z, took tracked
  players from 638 to 3,868 (x6.06), so a case whose target lies past that
  refresh assumes one such jump.  The N ratio is capped at 30x.

* Earned counts E_i(t).  A player who has not earned trophy i yet earns it
  at a hazard rate lambda_i per hour, so dE_i/dt = lambda_i * (N - E_i).
  lambda_i is estimated from the earned-count growth in the last 24 hours
  of reads when the history shows any (blended 70/30 with a tier prior);
  with a frozen history it is the tier prior alone: bronze 0.003/h,
  silver 0.002/h, gold 0.0008/h, platinum 0.0002/h, scaled by
  sqrt(rate_i / tier median rate) within the tier (clamped 0.5..1.8), so
  trophies more players already hold are assumed to be reached sooner.
  lambda is capped at 0.02/h.  E and N are then stepped hourly from the
  last read to the target and the median is 100 * E_i(target) / N(target).

* Spread.  Quantiles are the median moved on the logit scale by
  z_tau * sigma, with sigma = sqrt(0.9^2 + (0.6 * sqrt(days))^2): the 0.9
  is the refresh risk the practice cases demonstrate (a single refresh
  moved every rate outside the house baseline's narrow band), and the
  second term is a random-walk widening with horizon.  A trophy already
  at 100% in the last read keeps a median of 100 with a narrow band, since
  the documented refresh left it at 100%.

The forecast is deterministic: no randomness, no wall-clock reads, no
network.  Two runs on the same case write identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry ships none.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# First Exophase refresh the house documented (kit RULES.txt): launch night,
# 2026-11-19T04:30:00Z, tracked players 638 -> 3,868, every rate but the one
# at 100% lower than before.
FIRST_REFRESH_AT = datetime(2026, 11, 19, 4, 30, tzinfo=timezone.utc)
FIRST_REFRESH_GROWTH = 3868.0 / 638.0

# Hazard-rate priors (per hour) by tier, for a history that shows no growth.
LAMBDA_PRIOR = {
    "platinum": 0.0002,
    "gold": 0.0008,
    "silver": 0.002,
    "bronze": 0.003,
}
LAMBDA_CAP = 0.02
N_RATIO_CAP = 30.0

# Logit spread: a fixed refresh-risk term plus a random-walk term.
SIGMA_FIXED = 0.9
SIGMA_PER_SQRT_DAY = 0.6


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def fit_players_curve(reads):
    """Fit log N = a + b * log1p(hours_since_first / 12).

    Returns (a, b, t_first) or None when the history cannot support a fit.
    """
    times = [parse_time(r["at"]) for r in reads]
    t0 = times[0]
    xs = [math.log1p((t - t0).total_seconds() / 3600.0 / 12.0) for t in times]
    ys = [math.log(max(1.0, float(r["players_tracked"]))) for r in reads]
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b, t0)


def players_at(fit, when, n_last, t_last, target):
    """Tracked players at `when` under the fitted curve (or the fallback)."""
    if fit is not None:
        a, b, t0 = fit
        hours = (when - t0).total_seconds() / 3600.0
        value = math.exp(a + b * math.log1p(max(0.0, hours) / 12.0))
        return max(1.0, value)
    # No fit: flat history. The one documented first refresh happens at
    # the target itself (the answer is the refreshed read), so N stays at
    # its last value until then and jumps at the target.
    if t_last < FIRST_REFRESH_AT <= target and when >= target:
        return n_last * FIRST_REFRESH_GROWTH
    return float(n_last)


def estimate_lambda(reads, idx, tier, rate_now, tier_median):
    """Per-hour earn hazard for one trophy, from the last 24 h of reads."""
    prior = LAMBDA_PRIOR.get(tier, 0.003)
    scale = math.sqrt(max(0.01, rate_now) / max(0.01, tier_median))
    prior *= min(1.8, max(0.5, scale))
    times = [parse_time(r["at"]) for r in reads]
    t_last = times[-1]
    window = [
        (t, r) for t, r in zip(times, reads)
        if (t_last - t).total_seconds() <= 24 * 3600
    ]
    if len(window) < 2:
        return min(LAMBDA_CAP, prior)
    # Earned counts at the window ends, from the rates and tracked counts.
    def earned(read):
        rate = read["rates"][idx]
        if rate is None:
            return None
        return rate / 100.0 * float(read["players_tracked"])

    e_first = earned(window[0][1])
    e_last = earned(window[-1][1])
    if e_first is None or e_last is None:
        return min(LAMBDA_CAP, prior)
    hours = (window[-1][0] - window[0][0]).total_seconds() / 3600.0
    if hours <= 0:
        return min(LAMBDA_CAP, prior)
    growth = e_last - e_first
    if growth <= 0:
        return min(LAMBDA_CAP, prior)
    pool = []
    for _, read in window:
        e = earned(read)
        if e is not None:
            pool.append(float(read["players_tracked"]) - e)
    mean_pool = sum(pool) / len(pool) if pool else 0.0
    if mean_pool <= 0:
        return min(LAMBDA_CAP, prior)
    fitted = growth / (mean_pool * hours)
    lam = 0.7 * fitted + 0.3 * prior
    return min(LAMBDA_CAP, max(0.0, lam))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    target = parse_time(case["target"])
    times = [parse_time(r["at"]) for r in history]
    t_last = times[-1]
    n_last = float(history[-1]["players_tracked"])
    horizon_days = case["horizon_minutes"] / 1440.0

    fit = fit_players_curve(history)
    if fit is not None and fit[1] <= 0.01:
        # A flat (or shrinking) fit carries no growth signal: fall back to
        # the documented first refresh instead of a frozen N.
        fit = None
    n_target = players_at(fit, target, n_last, t_last, target)
    n_target = min(n_target, n_last * N_RATIO_CAP)
    n_target = max(n_target, n_last * 0.5)

    # Tier median of the last rates, for the within-tier hazard scaling.
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                rate = read["rates"][i]
                break
        last_rates.append(0.0 if rate is None else float(rate))
    tier_median = {}
    for tier in LAMBDA_PRIOR:
        vals = sorted(
            r for t, r in zip(trophies, last_rates)
            if t["tier"] == tier and r > 0
        )
        tier_median[tier] = vals[len(vals) // 2] if vals else 1.0

    sigma = math.sqrt(
        SIGMA_FIXED ** 2 + (SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
    )

    rows = []
    steps = max(1, int(math.ceil((target - t_last).total_seconds() / 3600.0)))
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate >= 99.995:
            # Everybody tracked already holds it, and the documented
            # refresh left it at 100%: keep it (almost) pinned there.
            rows.append(
                {"id": trophy["id"], "q": [99.9, 99.99, 100, 100, 100]}
            )
            continue
        earned = rate / 100.0 * n_last
        lam = estimate_lambda(
            history, i, trophy["tier"], rate, tier_median[trophy["tier"]]
        )
        # Step E and N hourly from the last read to the target.
        for step in range(1, steps + 1):
            when = t_last.timestamp() + step * 3600.0
            when_dt = datetime.fromtimestamp(when, tz=timezone.utc)
            if when_dt > target:
                when_dt = target
            n_now = players_at(fit, when_dt, n_last, t_last, target)
            n_now = min(n_now, n_last * N_RATIO_CAP)
            dt = (when_dt - datetime.fromtimestamp(
                t_last.timestamp() + (step - 1) * 3600.0, tz=timezone.utc
            )).total_seconds() / 3600.0
            if dt <= 0:
                continue
            earned += lam * max(0.0, n_now - earned) * dt
            earned = min(earned, n_now)
        median = clamp(100.0 * earned / max(1.0, n_target))
        lg = logit(median)
        qs = []
        prev = -1.0
        for z in Z:
            q = round4(100.0 / (1.0 + math.exp(-(lg + z * sigma))))
            q = max(prev, min(100.0, max(0.0, float(q))))
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
