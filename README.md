# Predict

A CLI toolkit for exploring Polymarket and managing predictions.

## Overview

- `predict.py` - Main CLI for discovering markets, reviewing and managing prediction history.
- `analyse.py` - Analytics tool for summarizing active markets, categories and computing metrics.

## Features

- Calculate allocations using confidence weighted [Kelly criterion](https://en.wikipedia.org/wiki/Kelly_criterion) 
- Filter markets by prediction edge, time and odds
- Saves predictions locally and compare [Brier scores](https://en.wikipedia.org/wiki/Brier_score)

## Setup

Install [Python](https://www.python.org/) and dependencies:

```powershell
pip install -r requirements.txt
```

Run the CLI

```powershell
python predict.py
```

## Configuration Settings for `predict.py`
- `HISTORY_FILE = "prediction_history.json"`
  - Local file used to store prediction history, skipped markets, and resolved outcomes.
- `BANKROLL = 42.00`
  - The total capital base used for allocation calculations.
- `GAMMA_API = "https://gamma-api.polymarket.com"`
  - Polymarket Gamma API base URL used for data fetches.
- `MIN_EDGE = 0.01`
  - Requires at least a 1% estimated edge before considering a trade.
- `MAX_DAYS = 420`
  - Excludes markets that resolve beyond 420 days to avoid overly long capital lock-up.
- `MIN_DAYS = 0`
  - Excludes markets resolving within fewer than 0 days (currently no minimum time horizon).
- `EXTREME_ODDS = 0.01`
  - Filters out extreme tail markets priced below 1% or above 99%.

## Planned

- Sync with Polymarket Portfolio

