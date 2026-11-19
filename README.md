# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ASEP9929/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ASEP9929/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy.

## How it works

`main.py` (Python 3.13, standard library only, no network) is run once per
case. It reads the case file from `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes one forecast file to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

An earn rate is a ratio — earned / tracked — so the two sides are forecast
separately and divided at the target:

- **Tracked players N(t).** Log N is fit against log(1 + hours/12) over the
  case history and extrapolated to the target. When the history is frozen
  (launch night: every read at 638 players), the program uses the one
  refresh the house documented: at 2026-11-19T04:30:00Z tracked players
  jumped 638 → 3,868.
- **Earned counts E(t).** Each trophy has a per-hour earn hazard λ, fitted
  from earned-count growth in the last 24 h of reads when the history shows
  any, otherwise a tier prior (bronze 0.003/h, silver 0.002/h, gold
  0.0008/h, platinum 0.0002/h, scaled within tier by the trophy's current
  rate). E and N are stepped hourly to the target; the median is
  100 · E(target) / N(target).
- **Spread.** Quantiles are the median moved on the logit scale by
  z_τ · σ, with σ = √(0.9² + (0.6·√days)²): a fixed refresh-risk term plus a
  random-walk term that widens with the horizon.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
