# Predict

A CLI toolkit for exploring active Polymarket markets and managing predictions.

## Repository Overview

- `predict.py` - Main pipeline for discovering markets, reviewing predictions/skips, targeting specific event URLs, and managing local prediction history.
- `analyse.py` - Market analytics utility for scanning active markets, computing volumes, spreads, velocity, and annualized yield.
- `tags.py` - Simple helper script that fetches active events and prints available tag labels for category filtering.

## Features

- Fetches active markets from the Polymarket Gamma API
- Filters markets by category, odds, timeframe, liquidity, and resolution state
- Applies risk controls such as edge thresholds, minimum/maximum days, and volume impact caps
- Saves prediction history locally
- Review previously predicted and skipped markets

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Run the main prediction engine:

```powershell
python predict.py
```

Run the comparative market analytics tool:

```powershell
python analyse.py
```

Print available Polymarket tags (Categories) for active events:

```powershell
python tags.py
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


