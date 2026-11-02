# Predict

A quantitative CLI engine for discovering Polymarket opportunities, sizing predictions, and managing portfolios.

## Overview

`predict.py` is an interactive terminal tool that connects directly to Polymarket to help you find mispriced markets, size bets with optimal Kelly allocation, and track live positions across multiple portfolios.

## Features

- **Balanced Kelly Sizing**: Sizes bets using the square-root Fisher geodesic to maximize compound growth while protecting against tail drawdowns.
- **Calibrated Beliefs**: Adjusts your confidence using Bayesian log-odds pooling weighted by your historical Brier score accuracy.
- **Real-Time CLOB Execution**: Walks live order-book depth to calculate exact VWAP prices, slippage, fees, and compounding APY.
- **Multi-Wallet & Paper Profiles**: Track live Polymarket portfolios, or create local paper-trading sandboxes.
- **Discovery Vectors**: Scan markets by Volume, 24h Trending, Liquidity, Soonest, Competitive (50/50), Low Volume, Newest, or Multi-Vector.
- **Portfolio Management**: Review open positions, add tranches on pullbacks, and stage limit sells when targets are reached.

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

- `1: Discover Markets`: Scan Polymarket using 9 discovery vectors up to 2,000 events deep.
- `2: Review Portfolio`: View open positions, stats, and add tranches or limit sells by row number.
- `3: Target Specific URLs`: Paste any Polymarket event URL or slug to analyze and size instantly.
- `4: Wallet Profiles`: Sync on-chain USDC balances, import open positions, or switch between wallet and paper accounts.
- `5: Tools & Settings`: Access the custom market calculator, manage stored data, or adjust threshold filters.

## Configuration

Settings can be edited in `config.py` or interactively via **Option 5 -> 3**:

- `KELLY_GAMMA = 0.5`: Sizing mode (`0.5` = Balanced Kelly, `1.0` = Full Efficient, `0.0` = Normal Kelly).
- `BANKROLL = 100.00`: Default capital base (automatically updated when syncing a live wallet).
- `MIN_EDGE = 0.00`: Minimum required edge before considering a trade (e.g. `0.02` for 2%).
- `MIN_DAYS = 0`: Minimum market duration in days.
- `MAX_DAYS = 420`: Maximum market duration in days.
- `EXTREME_ODDS = 0.00`: Tail filter to ignore extreme odds near 0% or 100%.
