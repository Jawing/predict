"""Utility functions for data formatting, date parsing, terminal rendering, and position valuation."""
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict


class ReturnToMainMenu(Exception):
    """Control-flow signal to jump directly back to the main menu from any submenu."""


class ExitProgram(Exception):
    """Control-flow signal to exit the CLI cleanly from any menu prompt."""


class WalletSummary(TypedDict, total=False):
    cash: float
    portfolio: float
    bankroll_add: float
    bankroll: float
    updated_at: str


class SnapshotRecord(TypedDict, total=False):
    ts: int
    date: str
    pu: float
    pm: float
    entry_price: float
    kelly: float
    entry_kelly: float
    raw_kelly: float
    edge: float
    apy: Optional[float]
    dynamic_ego: float
    action: str
    source: str


class PredictionRecord(TypedDict, total=False):
    question: str
    slug: str
    date: str
    pu: float
    pm: float
    cur_pm: float
    tranche_pm: float
    entry_price: float
    avg_price: float
    exit_price: Optional[float]
    dynamic_ego: float
    kelly: float
    entry_kelly: float
    raw_kelly: float
    cost_basis: float
    entry_cost: float
    total_size: float
    entry_bankroll: float
    shares: float
    current_value: float
    realized_pnl: Optional[float]
    edge: float
    apy: Optional[float]
    outcome: Optional[float]
    conditionId: str
    position: str
    outcomeIndex: int
    source: str
    wallet: str
    exited: bool
    is_watch: bool
    action: str
    endDate: Optional[str]
    last_synced_ts: int
    snapshots: List[SnapshotRecord]


class HistoryStore(TypedDict, total=False):
    profile_id: str
    wallet_address: Optional[str]
    alias: str
    bankroll_add: float
    wallet_summary: WalletSummary
    predicted: Dict[str, Any]
    exited: Dict[str, Any]
    resolved: Dict[str, Any]
    skipped: Dict[str, Any]
    settings: Dict[str, Any]


def ensure_list(val):
    """Ensures a value is wrapped in a list if not already one."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def parse_outcome_prices(m):
    """Safely extracts outcomePrices as a list of floats from a market or event response."""
    if not isinstance(m, dict):
        return []
    try:
        prices = m.get('outcomePrices')
        if isinstance(prices, str):
            return [float(x) for x in json.loads(prices)]
        elif isinstance(prices, list):
            return [float(x) for x in prices]
    except Exception:
        pass
    return []


def parse_market_date(date_val):
    """Parses a market endDate into (datetime_utc, seconds_left, days_until).

    Returns:
        tuple: (res_dt, seconds_left, days_until). All None if unparseable.
    """
    if not date_val:
        return None, None, None
    try:
        if isinstance(date_val, datetime):
            res_dt = date_val
        else:
            date_str = str(date_val).strip().replace('Z', '+00:00')
            res_dt = datetime.fromisoformat(date_str)
        if res_dt.tzinfo is None:
            res_dt = res_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds_left = (res_dt - now).total_seconds()
        days_until = max(0.0, seconds_left / 86400.0)
        return res_dt, seconds_left, days_until
    except Exception:
        return None, None, None


def format_time_remaining(seconds):
    """Formats remaining seconds into a readable string ('3d 04h' or 'HH:MM:SS' or 'Expired')."""
    if seconds is None:
        return "unknown"
    if seconds <= 0:
        return "Expired"
    d = int(seconds // 86400)
    rem = seconds % 86400
    h = int(rem // 3600)
    rem %= 3600
    mins = int(rem // 60)
    s = int(rem % 60)
    if d > 0:
        return f"{d}d {h:02d}h"
    return f"{h:02d}:{mins:02d}:{s:02d}"


def stringify_overflow(obj):
    """Recursively converts infinite and NaN floats into strings for JSON serialization."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        elif math.isnan(obj):
            return "NaN"
    elif isinstance(obj, dict):
        return {k: stringify_overflow(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [stringify_overflow(x) for x in obj]
    return obj


def ansi_link(url, label=None):
    """Formats an ANSI OSC 8 terminal hyperlink, falling back to label if url is missing."""
    if not label:
        label = url or ""
    if not url or url == "unknown":
        return str(label)
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def strip_ansi(text):
    """Strips ANSI escape codes (hyperlinks and colors) to measure visible length."""
    if not text:
        return ""
    return re.sub(r'\033\]8;;.*?\033\\|\033\[[0-9;]*m', '', str(text))


def get_history_counts(history):
    """Calculates unified counts of predictions and markets across history categories."""
    if not isinstance(history, dict):
        return {
            "active": 0, "exited": 0, "resolved": 0, "skipped": 0,
            "active_markets": 0, "exited_markets": 0, "resolved_markets": 0,
            "total_predictions": 0, "total_markets": 0
        }
    pred_dict = history.get("predicted", {})
    exit_dict = history.get("exited", {})
    res_dict = history.get("resolved", {})
    skip_dict = history.get("skipped", {})

    active_cnt = sum(len(ensure_list(v)) for v in pred_dict.values())
    exited_cnt = sum(len(ensure_list(v)) for v in exit_dict.values())
    resolved_cnt = sum(len(ensure_list(v)) for v in res_dict.values())
    skipped_cnt = len(skip_dict)

    active_mkts = len(pred_dict)
    exited_mkts = len(exit_dict)
    resolved_mkts = len(res_dict)

    return {
        "active": active_cnt,
        "exited": exited_cnt,
        "resolved": resolved_cnt,
        "skipped": skipped_cnt,
        "active_markets": active_mkts,
        "exited_markets": exited_mkts,
        "resolved_markets": resolved_mkts,
        "total_predictions": active_cnt + exited_cnt + resolved_cnt,
        "total_markets": active_mkts + exited_mkts + resolved_mkts + skipped_cnt,
    }


def calculate_position_holding(pred, cur_pm=None, bankroll=100.0):
    """Calculates mark-to-market live valuation, held Kelly fraction, and shares for a position.

    Returns:
        tuple: (live_value, held_mtm_k, shares)
    """
    if not isinstance(pred, dict):
        return 0.0, 0.0, 0.0

    is_wallet = (pred.get("source", "local") == "wallet")
    if pred.get("exited"):
        return 0.0, 0.0, 0.0
    has_holdings = is_wallet or (float(pred.get("shares", 0.0) or 0.0) > 0) or (float(pred.get("cost_basis", 0.0) or 0.0) > 0)
    if not has_holdings:
        return 0.0, 0.0, 0.0

    k = max(0.0, float(pred.get("kelly", 0.0) or 0.0))
    sh = float(pred.get("shares", 0.0) or 0.0)
    cb = float(pred.get("cost_basis", 0.0) or 0.0)

    if cur_pm is None:
        cur_pm = float(pred.get("cur_pm") if pred.get("cur_pm") is not None else pred.get("pm", 0.5) or 0.5)
    else:
        cur_pm = float(cur_pm)

    if sh > 0 and cur_pm > 0:
        live_val = sh * cur_pm
    elif cb > 0:
        entry_p = float(pred.get("pm", cur_pm) or cur_pm)
        live_val = cb * (cur_pm / entry_p if entry_p > 0 else 1.0)
        if sh == 0 and entry_p > 0:
            sh = cb / entry_p
    else:
        live_val = k * bankroll

    held_mtm_k = (live_val / bankroll) if bankroll > 0 else k
    return live_val, held_mtm_k, sh


def get_effective_bankroll(history, default_bankroll=100.0):
    """Returns active bankroll from history['wallet_summary'] or fallback default."""
    if isinstance(history, dict):
        ws = history.get("wallet_summary")
        if isinstance(ws, dict):
            br = ws.get("bankroll")
            if br is not None and float(br or 0.0) > 0:
                return float(br)
    return float(default_bankroll)


def calculate_implied_exit_price(avg_price, realized_pnl, total_size, fallback_price=None):
    """Computes implied exit price from realized PnL and total size, clamped to (0.0001, 0.9999)."""
    if total_size is not None and float(total_size) > 0:
        px = round(float(avg_price) + (float(realized_pnl) / float(total_size)), 4)
        return max(0.0001, min(0.9999, px))
    return fallback_price


def calculate_resolved_pnl(
    entry,
    outcome,
    bankroll=100.0,
    *,
    allow_kelly_fallback=True,
    use_entry_cost_fallback=False,
    fallback_api_pnl=None,
    preserve_exited=True,
):
    """Computes realized PnL for a resolved prediction entry.

    Returns:
        Optional[float]: Realized PnL in USDC, or None if watch-only / unsized.
    """
    if not isinstance(entry, dict) or outcome is None:
        return None

    if preserve_exited and entry.get("exited") and entry.get("realized_pnl") is not None:
        return entry.get("realized_pnl")

    if entry.get("is_watch") or entry.get("action") == "WATCH":
        return None

    cb = float(entry.get("cost_basis", 0.0) or 0.0)
    if use_entry_cost_fallback and cb == 0.0:
        cb = float(entry.get("entry_cost", 0.0) or 0.0)
    sh = float(entry.get("shares", 0.0) or 0.0)

    if allow_kelly_fallback and cb == 0.0:
        k_eval = float(entry.get("entry_kelly") or entry.get("kelly", 0.0) or 0.0)
        if k_eval > 0:
            cb = k_eval * float(bankroll)
            pm_val = float(entry.get("pm", 0.0) or 0.0)
            if sh == 0.0 and pm_val > 0:
                sh = cb / pm_val

    if sh > 0 or cb > 0:
        out_f = float(outcome)
        if out_f == 1.0:
            return round(sh * 1.0 - cb, 2)
        if out_f == 0.0:
            return round(-cb, 2)
        if out_f == 0.5:
            return round(sh * 0.5 - cb, 2)

    if fallback_api_pnl is not None and float(fallback_api_pnl) != 0.0 and entry.get("source") == "wallet":
        return float(fallback_api_pnl)
    return entry.get("realized_pnl")


def apply_closed_exit_fields(
    entry,
    *,
    avg_price=None,
    total_size=None,
    realized_pnl=None,
    cur_price=None,
    exit_price=None,
    exit_date=None,
    bankroll=None,
    mark_exited=False,
):
    """Normalizes entry_cost, size, entry prices, exit price, and date on a closed/exited record."""
    if not isinstance(entry, dict):
        return

    if cur_price is not None:
        entry["cur_pm"] = cur_price
    if realized_pnl is not None:
        entry["realized_pnl"] = realized_pnl

    tb = float(total_size or 0.0)
    ap = float(avg_price or 0.0)
    if tb > 0 and ap > 0:
        entry["entry_cost"] = round(tb * ap, 4)
        entry["total_size"] = tb
        entry["avg_price"] = ap
        entry["entry_price"] = ap
        snaps = entry.get("snapshots", [])
        if len(snaps) <= 1:
            entry["pm"] = ap
            if snaps:
                snaps[0]["pm"] = ap
                snaps[0]["entry_price"] = ap
    elif not entry.get("entry_cost") and float(entry.get("cost_basis", 0.0) or 0.0) > 0:
        entry["entry_cost"] = float(entry.get("cost_basis", 0.0))

    if bankroll is not None:
        entry["entry_bankroll"] = float(entry.get("entry_bankroll") or bankroll)

    if mark_exited:
        entry["exited"] = True
        entry["cost_basis"] = 0.0

    if exit_price is not None:
        entry["exit_price"] = exit_price
    if exit_date:
        entry["date"] = exit_date
