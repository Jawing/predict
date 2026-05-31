# Predict

A CLI toolkit for exploring Polymarket and managing predictions.

## Repository Overview

- `predict.py` - Main pipeline for discovering markets, reviewing and managing local prediction history.
- `analyse.py` - Market analytics tool for summarizing active markets, categories and computing metrics.

## Features

- Randomly discover markets from Polymarket to predict or review predicted/skipped markets
- Calculate allocations using [Kelly criterion](https://en.wikipedia.org/wiki/Kelly_criterion) with logit smoothing and weighted confidence
- Applies risk controls such as edge thresholds, minimum/maximum days, and volume impact caps
- Saves prediction history locally and calculate [Brier scores](https://en.wikipedia.org/wiki/Brier_score)

## Setup

Install [Python](https://www.python.org/)

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Run the main prediction pipeline:

```powershell
python predict.py
```

Run the market analytics tool:

```powershell
python analyse.py
```

## Configuration Settings for `predict.py`
- `HISTORY_FILE = "prediction_history.json"`
  - Local file used to store prediction history, skipped markets, and resolved outcomes.
- `BANKROLL = 42.00`
  - The total capital base used for allocation calculations and exposure tracking.
- `KELLY_FRACTION = 1.0`
  - Fraction of the theoretical Kelly bet size to use. `1.0` means full Kelly; lower values reduce position size.
- `MAX_VOLUME_IMPACT = 0.02`
  - Maximum share of available market volume allowed for a position. This helps limit market impact to 2% of liquidity.
- `MAX_GUESS = 20000`
  - Upper bound used when searching the Polymarket event universe. It limits the maximum offset probed when discovering markets.
- `GAMMA_API = "https://gamma-api.polymarket.com"`
  - Polymarket Gamma API base URL used for data fetches.
- `MIN_EDGE = 0.02`
  - Requires at least a 2% estimated edge before considering a trade.
- `MAX_DAYS = 400`
  - Excludes markets that resolve beyond 400 days to avoid overly long capital lock-up.
- `MIN_DAYS = 1`
  - Excludes very short-duration markets that resolve within 24 hours.
- `EXTREME_ODDS = 0.02`
  - Filters out extreme tail markets priced below 2% or above 98%.


