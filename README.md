# Predict

A CLI prediction tool for discovering Polymarket opportunities, sizing predictions, syncing and managing portfolios.

## Features

- **Balanced Kelly Sizing**: Sizes bets using the square-root Fisher geodesic to maximize compound growth while protecting against tail drawdowns.
- **Dynamic Trade Signals**: High-conviction holding (`HOLD (+X% to exit)`), spread-tested additions (`ADD`), and disciplined exits (`TAKE PROFIT` / `SELL`) without premature micro-trimming.
- **Calibrated Beliefs**: Adjusts confidence using Bayesian log-odds pooling weighted by historical Brier score accuracy (with optional market exclusion).
- **Real-Time CLOB Execution**: Walks live orderbook depth to calculate exact VWAP prices, slippage, fees, and compounding APY.
- **Multi-Portfolio & Off-Platform Capital**: Manage live Polymarket wallets and paper sandboxes with auto-sync and persistent off-platform bankroll tracking.
- **Discovery Vectors**: Scan markets by Volume, 24h Trending, Liquidity, Soonest, Competitive (50/50), Low Volume, Newest, or Multi-Vector.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the CLI:
```bash
python predict.py
```

## Main Menu Options

- `1: Discover Markets`: Scan Polymarket using discovery vectors up to 2,000 events deep.
- `2: Review Portfolio`: Inspect active, exited, and resolved positions; sort by Kelly, Edge, Exit Proximity, or Ego Impact.
- `3: Target Specific URLs`: Paste any Polymarket event URL or slug to analyze and size instantly.
- `4: Wallet Profiles & Sync`: Sync balances/positions on launch or on-demand, switch profiles, or create paper sandboxes.
- `5: Tools & Settings`: Custom calculators, data management (reset, manual resolve, Ego calibration selector), and filter settings.

## Configuration & Capital

Settings can be edited in `config.py` or interactively via **Option 5 -> 4**:

- `MIN_EDGE = 0.00`: Minimum required edge before considering a trade (e.g. `0.02` for 2%).
- `MIN_DAYS = 0`: Minimum market duration in days.
- `MAX_DAYS = 420`: Maximum market duration in days.
- `EXTREME_ODDS = 0.00`: Tail filter to ignore extreme odds near 0% or 100%.

### Bankroll Modes

- **Live Wallet (`0x...`)**: Automatically syncs on-chain cash and open positions from Polymarket on startup. Use `Bankroll Add` (Option 5 -> 5) to include persistent off-platform funds (`Total Bankroll = Cash + Positions + Bankroll Add`).
- **Paper Sandbox (`local` / custom)**: Hypothetical trading capital defined on profile creation and adjustable anytime via Option 5 -> 5. Paper profiles never query blockchain endpoints or overwrite local capital.
