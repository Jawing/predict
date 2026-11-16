# Predict

A simple terminal tool for finding Polymarket questions, making unbiased probability estimates, sizing bets, and tracking your accuracy over time.

## How It Works

- **Unbiased Forecasting**: Estimate the chance of an event (e.g. `65` or `60-70`) before seeing the market odds.
- **Accuracy-Weighted Confidence**: Measures how well your past predictions beat the market and adjusts how heavily to weight your view vs. the market price.
- **Sized to Orderbooks**: Checks live market depth and fees to suggest sensible bet sizes and clear signals (`BUY`, `HOLD`, `ADD`, `TAKE PROFIT`, `SELL`).
- **Live & Paper Portfolios**: Sync Polymarket wallets (`0x...`) automatically or test ideas in a paper sandbox.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the CLI:
```bash
python predict.py
```

## Main Menu

- `1: Discover Markets`: Browse active Polymarket events by category, market metrics (eg. volume, liquidity, or closeness of odds).
- `2: Review Portfolio & History`: View active, exited, resolved, and skipped markets, check performance, and tune which past markets count toward calibration.
- `3: Target Specific URLs or Slugs`: Paste a Polymarket event link or slug to evaluate it right away.
- `4: Wallet Profiles & Sync`: Switch between live wallets and paper sandboxes, or sync positions on demand.
- `5: Tools & Settings`: Custom probability calculator, market overview scanner, data cleanup/backups, and filter & calibration settings.

## Settings

Edit `config.py` or adjust per profile in **Option 5 -> 4**:

- `MIN_EDGE = 0.00`: Minimum advantage over the market price required to bet (e.g. `0.02` = 2%).
- `MIN_DAYS = 0` / `MAX_DAYS = 420`: Only show markets resolving within this day range.
- `EXTREME_ODDS = 0.00`: Skip markets priced very close to 0% or 100%.
- `EGO_WEIGHTING = "uniform"`: Weight past predictions equally (`"uniform"`) or by bet size (`"kelly"`).
