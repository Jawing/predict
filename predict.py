import math
import requests
import json
import os
import random
import re
import threading
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
HISTORY_FILE = "prediction_history.json"
BANKROLL = 100.00
GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_WALLET = ""  # Set your wallet address (0x...) to auto-fetch USDC balance

# --- FILTERING PARAMETERS ---
MIN_EDGE = 0.00      # X% minimum mathematical edge to bother executing (e.g., 0.01 = 1%)
MAX_DAYS = 420       # Ignore markets locking up capital for more than X days
MIN_DAYS = 0         # Ignore markets resolving within X days (e.g., 1 day = 24h, 0.5 days = 12h)
EXTREME_ODDS = 0.00  # Ignore tail-odds markets below X% (e.g., 0.01 = 1%)

# --- LOGISTIC DIMINISHING-RETURN LANDMARKS ---
# For sigma(z) = 1 / (1 + e^-z), normalized slope efficiency is eta(p) = 4*p*(1 - p) in [0, 1]
LOGISTIC_DECEL_LOW = (3.0 - math.sqrt(3.0)) / 6.0   # ~21.13%: Max Rate of Deceleration (sigma'''(z)=0, eta=66.7%)
LOGISTIC_DECEL_HIGH = (3.0 + math.sqrt(3.0)) / 6.0  # ~78.87%: Max Rate of Deceleration (sigma'''(z)=0, eta=66.7%)
LOGISTIC_HALF_LOW = (2.0 - math.sqrt(2.0)) / 4.0    # ~14.64%: Half-Maximum Slope (sigma'(z)=0.5*sigma'(0), eta=50.0%)
LOGISTIC_HALF_HIGH = (2.0 + math.sqrt(2.0)) / 4.0   # ~85.36%: Half-Maximum Slope (sigma'(z)=0.5*sigma'(0), eta=50.0%)

api = requests.Session()

from utils import ensure_list, parse_outcome_prices, format_time_remaining, stringify_overflow

def compute_dynamic_ego(base_ego, u_spread, m_spread=0.0):
    u_conviction = base_ego * (1.0 - u_spread)
    m_conviction = (1.0 - base_ego) * (1.0 - m_spread)
    if (u_conviction + m_conviction) > 0:
        return u_conviction / (u_conviction + m_conviction)
    return base_ego

def calculate_logistic_metrics(pu, pm, ego=0.5, held_kelly=None, cur_pm=None, fee_rate=0.0):
    eps = 0.01
    pu = float(pu if pu is not None else 0.5)
    pm = float(pm if pm is not None else 0.5)
    ego = float(ego if ego is not None else 0.5)
    eval_pm = float(cur_pm if cur_pm is not None else pm)

    if abs(pu - pm) < 1e-9:
        pu_norm = pm
    else:
        pu_clip = float(np.clip(pu, eps, 1.0 - eps))
        pm_clip = float(np.clip(pm, eps, 1.0 - eps))
        logit_pu = math.log(pu_clip / (1.0 - pu_clip))
        logit_pm = math.log(pm_clip / (1.0 - pm_clip))
        logit_norm = (ego * logit_pu) + ((1.0 - ego) * logit_pm)
        pu_norm = 1.0 / (1.0 + math.exp(-logit_norm))

    is_yes = pu >= pm
    win_prob_bal = pu_norm if is_yes else (1.0 - pu_norm)
    win_prob_u = pu if is_yes else (1.0 - pu)
    win_prob_m = pm if is_yes else (1.0 - pm)
    eval_win_m = eval_pm if is_yes else (1.0 - eval_pm)

    # Logistic slope efficiency eta(p) = 4*p*(1-p)
    eff_m = 4.0 * max(0.0, min(1.0, pm)) * (1.0 - max(0.0, min(1.0, pm)))
    eff_live = 4.0 * max(0.0, min(1.0, eval_pm)) * (1.0 - max(0.0, min(1.0, eval_pm)))
    eff_norm = 4.0 * pu_norm * (1.0 - pu_norm)

    true_cost = win_prob_m + (fee_rate * win_prob_m * (1.0 - win_prob_m))
    active_edge = win_prob_bal - true_cost
    if abs(active_edge) < 1e-9:
        active_edge = 0.0
    model_kelly = max(0.0, active_edge / (1.0 - true_cost)) if (true_cost < 1.0 and active_edge > 0.0) else 0.0
    eff_model_kelly = eff_m * model_kelly

    live_true_cost = eval_win_m + (fee_rate * eval_win_m * (1.0 - eval_win_m))
    live_active_edge = win_prob_bal - live_true_cost
    if abs(live_active_edge) < 1e-9:
        live_active_edge = 0.0
    live_model_kelly = max(0.0, live_active_edge / (1.0 - live_true_cost)) if (live_true_cost < 1.0 and live_active_edge > 0.0) else 0.0
    eff_live_model_kelly = eff_live * live_model_kelly

    base_k = max(0.0, float(held_kelly)) if held_kelly is not None else model_kelly
    if base_k < 1e-9:
        base_k = 0.0
    eff_kelly = eff_live * base_k

    # Exit / Diminishing-Return Regime on the winning side
    if eval_win_m >= LOGISTIC_HALF_HIGH or eval_win_m <= LOGISTIC_HALF_LOW:
        regime = "HALF-SLOPE"
    elif eval_win_m >= LOGISTIC_DECEL_HIGH or eval_win_m <= LOGISTIC_DECEL_LOW:
        regime = "DECEL-EXIT"
    else:
        regime = "ACTIVE"

    dist_to_decel = LOGISTIC_DECEL_HIGH - eval_win_m
    dist_to_half = LOGISTIC_HALF_HIGH - eval_win_m
    target_win_m = min(win_prob_bal, LOGISTIC_DECEL_HIGH)
    dist_to_target = target_win_m - eval_win_m
    delta_kelly = live_model_kelly - base_k
    if abs(delta_kelly) < 1e-9:
        delta_kelly = 0.0

    return {
        "pu_norm": pu_norm,
        "is_yes": is_yes,
        "win_prob_bal": win_prob_bal,
        "win_prob_u": win_prob_u,
        "win_prob_m": win_prob_m,
        "eval_pm": eval_pm,
        "eval_win_m": eval_win_m,
        "target_win_m": target_win_m,
        "dist_to_target": dist_to_target,
        "delta_kelly": delta_kelly,
        "eff_m": eff_m,
        "eff_live": eff_live,
        "eff_norm": eff_norm,
        "active_edge": active_edge,
        "live_active_edge": live_active_edge,
        "model_kelly": model_kelly,
        "eff_model_kelly": eff_model_kelly,
        "live_model_kelly": live_model_kelly,
        "eff_live_model_kelly": eff_live_model_kelly,
        "eff_kelly": eff_kelly,
        "regime": regime,
        "dist_to_decel": dist_to_decel,
        "dist_to_half": dist_to_half
    }

def get_wallet_usdc_balance(wallet_address):
    if not wallet_address or not wallet_address.startswith("0x"):
        return None
    rpcs = [
        "https://polygon-bor-rpc.publicnode.com",
        "https://1rpc.io/matic",
        "https://polygon.drpc.org"
    ]
    # Polymarket USD (pUSD collateral), Bridged USDC.e, and Native USDC on Polygon (all 6 decimals)
    collateral_contracts = [
        "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb",
        "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"
    ]
    addr_clean = wallet_address.lower().replace('0x', '').zfill(64)
    data = '0x70a08231' + addr_clean
    batch_payload = [
        {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": contract, "data": data}, "latest"],
            "id": idx
        }
        for idx, contract in enumerate(collateral_contracts)
    ]
    for rpc in rpcs:
        try:
            resp = api.post(rpc, json=batch_payload, timeout=5).json()
            if isinstance(resp, list):
                valid_results = [item.get("result") for item in resp if isinstance(item, dict) and item.get("result") and "error" not in item]
                if valid_results:
                    return sum(int(r_hex, 16) / 10**6 for r_hex in valid_results if r_hex != "0x")
        except Exception:
            continue
    return None

def get_wallet_portfolio_value(wallet_address, positions=None):
    if not wallet_address or not wallet_address.startswith("0x"):
        return 0.0
    try:
        val_resp = api.get(f"https://data-api.polymarket.com/value?user={wallet_address}", timeout=5).json()
        if isinstance(val_resp, list) and len(val_resp) > 0 and "value" in val_resp[0]:
            return float(val_resp[0]["value"] or 0.0)
    except Exception:
        pass
    if positions:
        total = 0.0
        for pos in positions:
            size = float(pos.get("size", 0.0))
            avg_price = float(pos.get("avgPrice", 0.0))
            value = float(pos.get("currentValue", pos.get("initialValue", size * avg_price)))
            total += value
        return total
    return 0.0

def calculate_vwap(asks, target_usdc):
    if not asks or target_usdc <= 0:
        return 0.0, 0.0
    valid_asks = []
    for ask in asks:
        try:
            price = float(ask['price'])
            size = float(ask['size'])
            if price > 0 and size > 0:
                valid_asks.append((price, size))
        except Exception:
            continue
    if not valid_asks:
        return 0.0, 0.0

    # CLOB API returns asks in descending order (highest price first).
    # To execute a buy order, sort ascending to consume the cheapest asks first.
    valid_asks.sort(key=lambda x: x[0])

    total_shares = 0.0
    remaining_usdc = target_usdc
    total_spent = 0.0
    for price, size in valid_asks:
        max_usdc_at_this_level = price * size
        if remaining_usdc >= max_usdc_at_this_level:
            total_shares += size
            remaining_usdc -= max_usdc_at_this_level
            total_spent += max_usdc_at_this_level
        else:
            shares = remaining_usdc / price
            total_shares += shares
            total_spent += remaining_usdc
            remaining_usdc = 0.0
            break
    if total_shares > 0:
        vwap = total_spent / total_shares
        return vwap, total_spent
    return 0.0, 0.0

def refresh_prices_from_clob(m):
    clob_ids_str = m.get('clobTokenIds')
    if not clob_ids_str:
        return m
    try:
        clob_ids = json.loads(clob_ids_str) if isinstance(clob_ids_str, str) else clob_ids_str
        if not clob_ids or len(clob_ids) == 0:
            return m
        yes_token = clob_ids[0]
        r = api.get(f"https://clob.polymarket.com/book?token_id={yes_token}", timeout=5)
        if r.status_code == 200:
            book = r.json()
            bids = [float(b['price']) for b in book.get('bids', [])]
            asks = [float(a['price']) for a in book.get('asks', [])]
            if bids and asks:
                best_bid = max(bids)
                best_ask = min(asks)
                m['bestBid'] = best_bid
                m['bestAsk'] = best_ask
                midpoint = (best_bid + best_ask) / 2.0
                m['outcomePrices'] = [midpoint, 1.0 - midpoint]
    except Exception:
        pass
    return m

def get_market_clob_tokens(m):
    clob_ids_str = m.get('clobTokenIds')
    if not clob_ids_str:
        return None, None
    try:
        clob_ids = json.loads(clob_ids_str) if isinstance(clob_ids_str, str) else clob_ids_str
        if not clob_ids:
            return None, None
        yes_token = clob_ids[0] if len(clob_ids) > 0 else None
        no_token = clob_ids[1] if len(clob_ids) > 1 else None
        return yes_token, no_token
    except Exception:
        return None, None

def stream_clob_book(m, custom_token_id=None):
    yes_token, no_token = get_market_clob_tokens(m)
    print("\n--- CLOB ORDERBOOK & TOKEN IDs ---")
    print(f"Market ID    : {m.get('id', 'unknown')}")
    print(f"Condition ID : {m.get('conditionId', 'unknown')}")
    print(f"YES Token ID : {yes_token or 'N/A'}")
    print(f"NO Token ID  : {no_token or 'N/A'}")

    if custom_token_id:
        if custom_token_id.lower() in ("no", "2") and no_token:
            target_token = no_token
            token_label = "NO"
        elif custom_token_id.lower() in ("yes", "1") and yes_token:
            target_token = yes_token
            token_label = "YES"
        else:
            target_token = custom_token_id
            token_label = "CUSTOM" if target_token not in (yes_token, no_token) else ("YES" if target_token == yes_token else "NO")
    else:
        target_token = yes_token
        token_label = "YES"

    if not target_token:
        print("[!] No valid CLOB Token ID available to query.\n")
        return ""

    print(f"Querying Book: [{token_label}] {target_token}")
    print("[*] Live CLOB refresh active -- Press Enter to stop (or type % bounds / command + Enter)...")

    is_running = True
    captured_input = []

    def wait_for_user():
        nonlocal is_running
        try:
            user_text = input()
            captured_input.append(user_text.strip())
        except EOFError:
            pass
        is_running = False

    listener = threading.Thread(target=wait_for_user, daemon=True)
    listener.start()

    while is_running:
        try:
            r = api.get(f"https://clob.polymarket.com/book?token_id={target_token}", timeout=5)
            if r.status_code == 200:
                book = r.json()
                bids_raw = book.get('bids', [])
                asks_raw = book.get('asks', [])
                bids = [(float(b['price']), float(b['size'])) for b in bids_raw if float(b.get('price', 0)) > 0]
                asks = [(float(a['price']), float(a['size'])) for a in asks_raw if float(a.get('price', 0)) > 0]

                if bids:
                    best_bid, bid_size = max(bids, key=lambda x: x[0])
                else:
                    best_bid, bid_size = 0.0, 0.0

                if asks:
                    best_ask, ask_size = min(asks, key=lambda x: x[0])
                else:
                    best_ask, ask_size = 1.0, 0.0

                midpoint = (best_bid + best_ask) / 2.0
                spread = max(0.0, best_ask - best_bid)

                if target_token == yes_token:
                    m['bestBid'] = best_bid
                    m['bestAsk'] = best_ask
                    m['outcomePrices'] = [midpoint, 1.0 - midpoint]
                    m['cached_bid'], m['cached_ask'], m['cached_mid'] = best_bid, best_ask, midpoint
                elif target_token == no_token:
                    yes_bid = 1.0 - best_ask
                    yes_ask = 1.0 - best_bid
                    yes_mid = (yes_bid + yes_ask) / 2.0
                    m['bestBid'] = yes_bid
                    m['bestAsk'] = yes_ask
                    m['outcomePrices'] = [yes_mid, 1.0 - yes_mid]
                    m['cached_bid'], m['cached_ask'], m['cached_mid'] = yes_bid, yes_ask, yes_mid

                now_str = datetime.now().strftime("%H:%M:%S")
                status_line = (
                    f"[{now_str} {token_label}] "
                    f"Bid: {best_bid*100:.1f}% ({bid_size:,.0f} sh) | "
                    f"Ask: {best_ask*100:.1f}% ({ask_size:,.0f} sh) | "
                    f"Mid: {midpoint*100:.1f}% | Spread: {spread*100:.1f}%   "
                )
                print(status_line, end="\r", flush=True)
            else:
                print(f"[!] CLOB HTTP {r.status_code} for token {target_token[:16]}...   ", end="\r", flush=True)
        except Exception as e:
            print(f"[!] CLOB refresh error: {e}   ", end="\r", flush=True)

        for _ in range(10):
            if not is_running:
                break
            time.sleep(0.15)

    print()
    return captured_input[0] if captured_input else ""

def resolve_position_market_metadata(pos, event_cache=None):
    if event_cache is None:
        event_cache = {}
    cond_id = pos.get("conditionId")
    event_id = pos.get("eventId")
    event_slug = pos.get("eventSlug")
    pos_slug = pos.get("slug")
    m = None

    # 1. Lookup via parent Event ID (accurate & fast, caches multi-market events)
    if event_id:
        event_id_str = f"id:{event_id}"
        if event_id_str not in event_cache:
            try:
                ev_resp = api.get(f"{GAMMA_API}/events/{event_id}", timeout=5).json()
                event_cache[event_id_str] = ev_resp if isinstance(ev_resp, dict) else {}
            except Exception:
                event_cache[event_id_str] = {}
        ev_data = event_cache.get(event_id_str, {})
        for cand in ev_data.get("markets", []):
            if cand.get("conditionId") == cond_id or (pos_slug and cand.get("slug") == pos_slug):
                m = cand
                break

    # 2. Lookup via parent Event Slug (handles closed/redeemed multi-market events without eventId)
    if not m and event_slug:
        ev_slug_key = f"slug:{event_slug}"
        if ev_slug_key not in event_cache:
            try:
                ev_list = api.get(f"{GAMMA_API}/events?slug={event_slug}", timeout=5).json()
                event_cache[ev_slug_key] = ev_list[0] if (isinstance(ev_list, list) and ev_list) else {}
            except Exception:
                event_cache[ev_slug_key] = {}
        ev_data = event_cache.get(ev_slug_key, {})
        for cand in ev_data.get("markets", []):
            if cand.get("conditionId") == cond_id or (pos_slug and cand.get("slug") == pos_slug):
                m = cand
                break

    # 3. Fallback: Lookup via market slug on Gamma API and verify conditionId
    if not m and pos_slug:
        try:
            resp = api.get(f"{GAMMA_API}/markets?slug={pos_slug}", timeout=5).json()
            if isinstance(resp, list):
                for cand in resp:
                    if cand.get("conditionId") == cond_id or cand.get("slug") == pos_slug:
                        m = cand
                        break
        except Exception:
            pass

    # 4. Extract metadata using Gamma market or position fields directly
    market_id = str(m.get("id")) if (m and m.get("id")) else str(cond_id)
    question = (m.get("question") if m else None) or pos.get("title") or "unknown"
    slug = pos.get("eventSlug") or pos.get("slug") or (m.get("slug") if m else "unknown")
    end_date = (m.get("endDate") if m else None) or pos.get("endDate") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return m, market_id, question, slug, end_date

def determine_redeemable_outcome(pos, m=None):
    # Returns 1.0 if the held position (outcomeIndex) WON, 0.0 if it LOST, 0.5 if 50/50
    try:
        outcome_idx = int(pos.get("outcomeIndex", 0))
    except Exception:
        outcome_idx = 0
    if m:
        prices = parse_outcome_prices(m)
        if len(prices) > outcome_idx:
            if prices[outcome_idx] >= 0.99:
                return 1.0
            elif prices[outcome_idx] <= 0.01:
                return 0.0
    try:
        cur_price = float(pos.get("curPrice", 0.5))
        if cur_price >= 0.99:
            return 1.0
        elif cur_price <= 0.01:
            return 0.0
    except Exception:
        pass
    return 0.5

def import_polymarket_positions(wallet_address, history, auto_sync=False):
    global BANKROLL
    saved_wallet = history.get("wallet_address") or POLYGON_WALLET
    if not wallet_address:
        if auto_sync and saved_wallet:
            wallet_address = saved_wallet
        else:
            prompt_str = f"Enter Polymarket wallet address (0x...) [Press Enter for {saved_wallet}]: " if saved_wallet else "Enter Polymarket wallet address (0x...): "
            user_w = input(prompt_str).strip()
            wallet_address = user_w if user_w else saved_wallet

    if not wallet_address or not wallet_address.startswith("0x"):
        print("[!] Invalid address format.")
        return history

    history["wallet_address"] = wallet_address
    history.setdefault("predicted", {})
    history.setdefault("exited", {})
    history.setdefault("resolved", {})
    history.setdefault("skipped", {})
    profile_url = f"https://polymarket.com/profile/{wallet_address}"
    linked_wallet = f"\033]8;;{profile_url}\033\\{wallet_address}\033]8;;\033\\"
    base_ego = calculate_base_ego(history)

    print(f"[*] Syncing Polymarket wallet {linked_wallet}...")
    try:
        url = f"https://data-api.polymarket.com/positions?user={wallet_address}&sizeThreshold=0&limit=500"
        open_positions = api.get(url, timeout=10).json()
        if not isinstance(open_positions, list):
            open_positions = []
    except Exception as e:
        print(f"[!] Error fetching open positions: {e}")
        open_positions = []

    try:
        closed_url = f"https://data-api.polymarket.com/closed-positions?user={wallet_address}&limit=500"
        closed_positions = api.get(closed_url, timeout=10).json()
        if not isinstance(closed_positions, list):
            closed_positions = []
    except Exception as e:
        print(f"[!] Error fetching closed positions: {e}")
        closed_positions = []

    open_keys = {(p.get("conditionId"), p.get("outcome")) for p in open_positions if isinstance(p, dict)}
    for cp in closed_positions:
        if not isinstance(cp, dict):
            continue
        cp["_from_closed"] = True
        tb = float(cp.get("totalBought") or 0.0)
        avg_p = float(cp.get("avgPrice") or 0.5)
        cur_p = float(cp.get("curPrice") if cp.get("curPrice") is not None else avg_p)
        r_pnl = round(float(cp.get("realizedPnl") or 0.0), 2)
        gross_val = round(tb * avg_p, 2)
        # For a resolved loss (curPrice == 0), totalBought includes shares bought & sold mid-trade; abs(realizedPnl) is the net USDC lost
        if cur_p <= 0.01 and r_pnl < 0:
            cp["initialValue"] = round(abs(r_pnl), 2)
        else:
            cp["initialValue"] = gross_val
        cp["size"] = tb

    positions = list(open_positions) + [
        cp for cp in closed_positions
        if isinstance(cp, dict) and (cp.get("conditionId"), cp.get("outcome")) not in open_keys
    ]

    trades_by_cond = {}
    try:
        act_url = f"https://data-api.polymarket.com/activity?user={wallet_address}&limit=500"
        act_resp = api.get(act_url, timeout=10).json()
        if isinstance(act_resp, list):
            for ev in act_resp:
                if not isinstance(ev, dict) or ev.get("type") != "TRADE":
                    continue
                c_id = ev.get("conditionId")
                if not c_id:
                    continue
                trades_by_cond.setdefault(c_id, []).append({
                    "ts": int(ev.get("timestamp") or 0),
                    "side": str(ev.get("side") or "BUY").upper(),
                    "usdc": float(ev.get("usdcSize") or 0.0),
                    "shares": float(ev.get("size") or 0.0),
                    "price": float(ev.get("price") or 0.5)
                })
            for c_id in trades_by_cond:
                trades_by_cond[c_id].sort(key=lambda x: x["ts"])
    except Exception:
        trades_by_cond = {}

    print("[*] Calculating Live Cash & Portfolio Values...")
    prev_bankroll = float(history.get("wallet_summary", {}).get("bankroll", BANKROLL) or BANKROLL)
    cash_balance = get_wallet_usdc_balance(wallet_address)
    if cash_balance is None:
        cash_balance = history.get("wallet_summary", {}).get("cash", 0.0)
    portfolio_value = get_wallet_portfolio_value(wallet_address, open_positions)

    BANKROLL = round(cash_balance + portfolio_value, 4)
    history["wallet_summary"] = {
        "cash": round(cash_balance, 4),
        "portfolio": round(portfolio_value, 4),
        "bankroll": BANKROLL,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    }
    print(f"Live Bankroll Updated: ${BANKROLL:,.2f} (Cash: ${cash_balance:,.2f} | Portfolio: ${portfolio_value:,.2f})")

    if not positions:
        print("[*] No positions returned from Polymarket endpoints.")
    else:
        print(f"Found {len(open_positions)} active/unclosed + {len(closed_positions)} closed/redeemed positions. Syncing...")

    imported_predicted = 0
    imported_exited = 0
    imported_resolved = 0
    updated_existing = 0
    event_cache = {}
    wallet_cond_ids = {p.get("conditionId") for p in open_positions if isinstance(p, dict) and p.get("conditionId")}
    to_import = []

    for pos in positions:
        cond_id = pos.get("conditionId")
        if not cond_id:
            continue
        pos_outcome_str = pos.get("outcome", "Yes")
        try:
            pos_outcome_idx = int(pos.get("outcomeIndex", 0))
        except Exception:
            pos_outcome_idx = 0

        avg_price = round(float(pos.get("avgPrice", 0.5)), 6)
        cur_price = round(float(pos.get("curPrice", avg_price)), 6)
        size = float(pos.get("size", 0.0))
        initial_val = round(float(pos.get("initialValue", size * avg_price)), 4)
        realized_pnl = round(float(pos.get("realizedPnl", pos.get("cashPnl", 0.0)) or 0.0), 4)
        is_from_closed = bool(pos.get("_from_closed"))

        # Fast-path 1: Skip Gamma API call if already permanently stored in history["resolved"]
        already_in_resolved = False
        for m_id, res_val in list(history.get("resolved", {}).items()):
            res_list = ensure_list(res_val)
            matching_res = [
                r for r in res_list
                if r.get("conditionId") == cond_id
                and (r.get("source") != "wallet" or r.get("position", pos_outcome_str) == pos_outcome_str)
            ]
            if matching_res:
                already_in_resolved = True
                for r in matching_res:
                    if r.get("realized_pnl") is None and realized_pnl != 0.0:
                        r["realized_pnl"] = realized_pnl
                    updated_existing += 1
                break
        if already_in_resolved:
            continue

        # Fast-path 2: If already in history["exited"] and still closed on-chain, update live price / check resolution without Gamma call
        if is_from_closed:
            already_in_exited = False
            for m_id, ex_val in list(history.get("exited", {}).items()):
                ex_list = ensure_list(ex_val)
                matching_ex = [
                    e for e in ex_list
                    if e.get("conditionId") == cond_id
                    and (e.get("source") != "wallet" or e.get("position", pos_outcome_str) == pos_outcome_str)
                ]
                if not matching_ex:
                    continue
                already_in_exited = True
                if bool(pos.get("redeemable")) or cur_price >= 0.99 or cur_price <= 0.01:
                    out_v = 1.0 if cur_price >= 0.99 else 0.0
                    for e in matching_ex:
                        e["cur_pm"] = cur_price
                        e["realized_pnl"] = realized_pnl
                        e["outcome"] = out_v
                        updated_existing += 1
                    rem_ex = [e for e in ex_list if e not in matching_ex]
                    if rem_ex:
                        history["exited"][m_id] = rem_ex
                    else:
                        history["exited"].pop(m_id, None)
                    history["resolved"].setdefault(m_id, []).extend(matching_ex)
                else:
                    for e in matching_ex:
                        e["cur_pm"] = cur_price
                        e["realized_pnl"] = realized_pnl
                        updated_existing += 1
                break
            if already_in_exited:
                continue

        m, market_id, question, slug, end_date = resolve_position_market_metadata(pos, event_cache)
        cond_trades = trades_by_cond.get(cond_id, [])
        latest_trade_ts = max((t["ts"] for t in cond_trades), default=int(pos.get("timestamp") or 0))

        is_market_resolved = bool(m and (m.get("closed") or m.get("umaResolutionStatus") == "resolved"))
        is_resolved_pos = bool(pos.get("redeemable")) or is_market_resolved or (is_from_closed and (cur_price >= 0.99 or cur_price <= 0.01))
        is_exited_open_pos = is_from_closed and not is_resolved_pos
        pos["redeemable"] = is_resolved_pos
        pos["_exited"] = is_exited_open_pos

        try:
            res_dt = datetime.fromisoformat(str(end_date).replace('Z', '+00:00'))
            if res_dt.tzinfo is None:
                res_dt = res_dt.replace(tzinfo=timezone.utc)
            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            days_until = seconds_left / 86400.0
        except Exception:
            days_until = 0.0

        # Check if already stored in history["predicted"] and update live values
        already_stored = False
        for m_id in list(history.get("predicted", {}).keys()):
            preds_list = ensure_list(history["predicted"][m_id])
            matching_preds = [
                p for p in preds_list
                if p.get("conditionId") == cond_id and (
                    p.get("source") != "wallet" or p.get("position", pos_outcome_str) == pos_outcome_str
                )
            ]
            if not matching_preds:
                continue

            already_stored = True
            wallet_preds = [p for p in matching_preds if p.get("source") == "wallet"]
            local_preds = [p for p in matching_preds if p.get("source") != "wallet"]

            prev_wallet_usdc = sum(
                float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                for wp in wallet_preds
            )
            max_synced_ts = max((int(wp.get("last_synced_ts", 0)) for wp in wallet_preds), default=0)
            if wallet_preds and max_synced_ts == 0 and abs(initial_val - prev_wallet_usdc) <= 0.05:
                max_synced_ts = latest_trade_ts

            unseen_trades = [t for t in cond_trades if t["ts"] > max_synced_ts] if wallet_preds else cond_trades
            if unseen_trades:
                existing_rem_usdc = prev_wallet_usdc
                prev_sh_est = sum(
                    float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                    / max(0.001, float(wp.get("tranche_pm", wp.get("pm", avg_price))))
                    for wp in wallet_preds
                )
                existing_rem_shares = prev_sh_est
                new_buys = []
                for tr in unseen_trades:
                    if tr["side"] == "BUY":
                        new_buys.append({
                            "usdc": tr["usdc"],
                            "shares": tr["shares"],
                            "pm": tr["price"],
                            "ts": tr["ts"]
                        })
                    elif tr["side"] == "SELL":
                        sell_sh = tr["shares"]
                        tot_sh = existing_rem_shares + sum(b["shares"] for b in new_buys)
                        if tot_sh > 0:
                            keep_ratio = max(0.0, (tot_sh - sell_sh) / tot_sh)
                            existing_rem_shares *= keep_ratio
                            existing_rem_usdc *= keep_ratio
                            for b in new_buys:
                                b["shares"] *= keep_ratio
                                b["usdc"] *= keep_ratio
                delta_usdc = round(sum(b["usdc"] for b in new_buys), 4) if wallet_preds else initial_val
                new_buy_sh = sum(b["shares"] for b in new_buys)
                if new_buy_sh > 0 and sum(b["usdc"] for b in new_buys) > 0:
                    implied_pm = round(sum(b["usdc"] for b in new_buys) / new_buy_sh, 6)
                    delta_pm = implied_pm if 0.001 <= implied_pm <= 0.999 else avg_price
                else:
                    delta_pm = avg_price
            else:
                delta_usdc = round(max(0.0, initial_val - prev_wallet_usdc), 4)
                if not wallet_preds:
                    delta_pm = avg_price
                else:
                    prev_shares = sum(
                        float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                        / max(0.001, float(wp.get("tranche_pm", wp.get("pm", avg_price))))
                        for wp in wallet_preds
                    )
                    delta_shares = size - prev_shares
                    if delta_shares > 0 and delta_usdc > 0:
                        implied_pm = round(delta_usdc / delta_shares, 6)
                        delta_pm = implied_pm if 0.001 <= implied_pm <= 0.999 else avg_price
                    else:
                        delta_pm = avg_price

            # Always let the user choose when linking local prediction(s) to a new wallet position or new delta tranche
            unmatched_delta_for_import = False
            if local_preds and delta_usdc > 0.05:
                default_idx = min(
                    range(1, len(local_preds) + 1),
                    key=lambda idx_i: abs(float(local_preds[idx_i - 1].get("kelly", 0.0)) * BANKROLL - delta_usdc)
                )
                web_link = f"https://polymarket.com/event/{slug}"
                linked_q = f"\033]8;;{web_link}\033\\{question}\033]8;;\033\\"
                print(f"\n[SYNC TRANCHE] On-chain execution detected (+${delta_usdc:,.2f} @ {delta_pm*100:.2f}%) for: {linked_q}")
                for i_lp, lp_item in enumerate(local_preds, 1):
                    rec_tag = " (Recommended match)" if i_lp == default_idx else ""
                    t_usd = float(lp_item.get("kelly", 0.0)) * BANKROLL
                    print(f"  [{i_lp}] Date: {lp_item.get('date')} | pu: {float(lp_item.get('pu', 0.5))*100:.1f}% | pm: {float(lp_item.get('pm', 0.5))*100:.1f}% | Target: ${t_usd:,.2f} ({float(lp_item.get('kelly', 0))*100:.2f}%){rec_tag}")
                print("  [0] None (keep local prediction(s) unexecuted & import on-chain trade separately)")
                sel_raw = input(f"  Select which local prediction was executed [Enter for {default_idx}]: ").strip()
                try:
                    sel_idx = int(sel_raw) if sel_raw else default_idx
                except ValueError:
                    sel_idx = default_idx

                if 1 <= sel_idx <= len(local_preds):
                    chosen_lp = local_preds.pop(sel_idx - 1)
                    if wallet_preds:
                        prev_tot = sum(float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) for wp in wallet_preds)
                        for wp in wallet_preds:
                            w_sh = (float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) / prev_tot) if prev_tot > 0 else (1.0 / len(wallet_preds))
                            wp["cost_basis"] = round(max(0.0, initial_val - delta_usdc) * w_sh, 4)
                    ego_lp = float(chosen_lp.get("dynamic_ego", base_ego))
                    lm_lp = calculate_logistic_metrics(chosen_lp.get("pu", delta_pm), delta_pm, ego_lp)
                    chosen_lp["question"] = question
                    chosen_lp["slug"] = slug
                    chosen_lp["tranche_pm"] = delta_pm
                    chosen_lp["pm"] = delta_pm
                    chosen_lp["cur_pm"] = cur_price
                    chosen_lp["cost_basis"] = round(0.0 if is_exited_open_pos else delta_usdc, 4)
                    chosen_lp["realized_pnl"] = realized_pnl
                    chosen_lp["kelly"] = lm_lp["model_kelly"]
                    chosen_lp["edge"] = lm_lp["active_edge"]
                    _, chosen_lp["apy"] = calculate_annualized_yield(chosen_lp["edge"], lm_lp["win_prob_m"], days_until)
                    chosen_lp["position"] = pos.get("outcome", "Yes")
                    chosen_lp["outcomeIndex"] = pos_outcome_idx
                    chosen_lp["source"] = "wallet"
                    chosen_lp["exited"] = is_exited_open_pos
                    if latest_trade_ts > 0:
                        chosen_lp["last_synced_ts"] = latest_trade_ts
                    wallet_preds.append(chosen_lp)
                    if is_exited_open_pos:
                        imported_exited += 1
                    else:
                        imported_predicted += 1
                else:
                    unmatched_delta_for_import = True
                    if wallet_preds:
                        prev_tot = sum(float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) for wp in wallet_preds)
                        for wp in wallet_preds:
                            w_sh = (float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) / prev_tot) if prev_tot > 0 else (1.0 / len(wallet_preds))
                            wp["cost_basis"] = round(max(0.0, initial_val - delta_usdc) * w_sh, 4)

            wallet_target_usdc = max(0.0, initial_val - delta_usdc) if unmatched_delta_for_import else initial_val

            # Update existing wallet tranche(s) using the true Kelly formula with ego, pu, and tranche pm
            if wallet_preds:
                raw_weights = [
                    float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                    for wp in wallet_preds
                ]
                total_w = sum(raw_weights)
                for wp, w_i in zip(wallet_preds, raw_weights):
                    share = (w_i / total_w) if total_w > 0 else (1.0 / len(wallet_preds))
                    tranche_pm = float(wp.get("tranche_pm", wp.get("pm", avg_price)))
                    ego_wp = float(wp.get("dynamic_ego", base_ego))
                    lm_wp = calculate_logistic_metrics(wp.get("pu", tranche_pm), tranche_pm, ego_wp)
                    wp["question"] = question
                    wp["slug"] = slug
                    wp["pm"] = tranche_pm
                    wp["cur_pm"] = cur_price
                    wp["cost_basis"] = round(0.0 if is_exited_open_pos else (wallet_target_usdc * share), 4)
                    wp["realized_pnl"] = round(realized_pnl * share, 4)
                    wp["kelly"] = lm_wp["model_kelly"]
                    wp["edge"] = lm_wp["active_edge"]
                    _, wp["apy"] = calculate_annualized_yield(wp["edge"], lm_wp["win_prob_m"], days_until)
                    wp["position"] = pos.get("outcome", "Yes")
                    wp["outcomeIndex"] = pos_outcome_idx
                    wp["source"] = "wallet"
                    wp["exited"] = is_exited_open_pos
                    if latest_trade_ts > 0:
                        wp["last_synced_ts"] = latest_trade_ts
                    updated_existing += 1

            # Keep any remaining local predictions untouched as unexecuted local entries
            for lp in local_preds:
                lp["source"] = "local"

            if unmatched_delta_for_import:
                to_import.append({
                    "pos": pos,
                    "m": m,
                    "market_id": market_id,
                    "question": question,
                    "slug": slug,
                    "end_date": end_date,
                    "avg_price": delta_pm,
                    "cur_price": cur_price,
                    "size": size,
                    "initial_val": delta_usdc,
                    "realized_pnl": realized_pnl,
                    "days_until": days_until,
                    "cond_id": cond_id,
                    "latest_trade_ts": latest_trade_ts
                })

            # Re-key if previously stored under wrong market_id
            target_key = market_id
            if m_id != target_key:
                history["predicted"].pop(m_id, None)
                if preds_list:
                    history["predicted"].setdefault(target_key, []).extend(preds_list)
            else:
                if preds_list:
                    history["predicted"][target_key] = preds_list
                else:
                    history["predicted"].pop(target_key, None)

            if pos.get("redeemable"):
                outcome_val = determine_redeemable_outcome(pos, m)
                cur_preds = ensure_list(history["predicted"].get(target_key, []))
                to_resolve = [
                    p for p in cur_preds
                    if p.get("conditionId") == cond_id and p.get("source") == "wallet"
                    and p.get("position", pos_outcome_str) == pos_outcome_str
                ]
                remaining_pred = [p for p in cur_preds if p not in to_resolve]
                if remaining_pred:
                    history["predicted"][target_key] = remaining_pred
                else:
                    history["predicted"].pop(target_key, None)
                for p in to_resolve:
                    p["outcome"] = outcome_val
                    p["position"] = pos_outcome_str
                    p["outcomeIndex"] = pos_outcome_idx
                    p["realized_pnl"] = realized_pnl
                    p["source"] = "wallet"
                if to_resolve:
                    history["resolved"].setdefault(target_key, []).extend(to_resolve)
            elif is_exited_open_pos:
                cur_preds = ensure_list(history["predicted"].get(target_key, []))
                to_exit = [
                    p for p in cur_preds
                    if p.get("conditionId") == cond_id and p.get("source") == "wallet"
                    and p.get("position", pos_outcome_str) == pos_outcome_str
                ]
                remaining_pred = [p for p in cur_preds if p not in to_exit]
                if remaining_pred:
                    history["predicted"][target_key] = remaining_pred
                else:
                    history["predicted"].pop(target_key, None)
                for p in to_exit:
                    p["exited"] = True
                    p["cost_basis"] = 0.0
                    p["realized_pnl"] = realized_pnl
                if to_exit:
                    history["exited"].setdefault(target_key, []).extend(to_exit)
            break

        if already_stored:
            continue

        to_import.append({
            "pos": pos,
            "m": m,
            "market_id": market_id,
            "question": question,
            "slug": slug,
            "end_date": end_date,
            "avg_price": avg_price,
            "cur_price": cur_price,
            "size": size,
            "initial_val": initial_val,
            "realized_pnl": realized_pnl,
            "days_until": days_until,
            "cond_id": cond_id,
            "latest_trade_ts": latest_trade_ts
        })

    # Move any remaining wallet positions in history["predicted"] that are no longer in open_positions into history["exited"]
    for m_id in list(history.get("predicted", {}).keys()):
        preds_list = ensure_list(history["predicted"][m_id])
        kept_preds = []
        moved_exits = []
        for p in preds_list:
            c_id = p.get("conditionId")
            if p.get("source") == "wallet" and c_id and c_id not in wallet_cond_ids:
                p["cost_basis"] = 0.0
                p["exited"] = True
                sells = [t for t in trades_by_cond.get(c_id, []) if t["side"] == "SELL"]
                if sells:
                    p["cur_pm"] = round(sells[-1]["price"], 6)
                    p["last_synced_ts"] = max(int(p.get("last_synced_ts", 0)), sells[-1]["ts"])
                moved_exits.append(p)
            else:
                kept_preds.append(p)
        if moved_exits:
            history["exited"].setdefault(m_id, []).extend(moved_exits)
            if kept_preds:
                history["predicted"][m_id] = kept_preds
            else:
                history["predicted"].pop(m_id, None)

    if auto_sync and to_import:
        n_res_imp = sum(1 for x in to_import if x["pos"].get("redeemable"))
        n_ex_imp = sum(1 for x in to_import if x["pos"].get("_exited"))
        n_open_imp = len(to_import) - n_res_imp - n_ex_imp
        imp_ans = input(f"\n[?] Found {len(to_import)} unimported wallet position(s) ({n_open_imp} active, {n_ex_imp} exited, {n_res_imp} resolved). Import now? (y/n) [Enter=n]: ").strip().lower()
        if imp_ans != 'y':
            to_import = []

    imported_records = {}
    idx = 0
    skip_all_remaining = False
    while idx < len(to_import):
        item = to_import[idx]
        pos = item["pos"]
        m = item["m"]
        market_id = item["market_id"]
        question = item["question"]
        slug = item["slug"]
        end_date = item["end_date"]
        avg_price = item["avg_price"]
        cur_price = item.get("cur_price", avg_price)
        size = item["size"]
        initial_val = item["initial_val"]
        realized_pnl = item.get("realized_pnl", 0.0)
        days_until = item["days_until"]
        cond_id = item["cond_id"]
        latest_trade_ts = item.get("latest_trade_ts", 0)

        outcome_str = pos.get("outcome", "Yes")
        try:
            outcome_idx = int(pos.get("outcomeIndex", 0))
        except Exception:
            outcome_idx = 0
        web_link = f"https://polymarket.com/event/{slug}"
        linked_question = f"\033]8;;{web_link}\033\\{question}\033]8;;\033\\"

        if not skip_all_remaining:
            progress_str = f" {idx + 1}/{len(to_import)}" if len(to_import) > 1 else ""
            print(f"\n[IMPORT{progress_str}] Question: {linked_question}")
            if pos.get("redeemable"):
                res_out = determine_redeemable_outcome(pos, m)
                pos_res = "WON" if res_out == 1.0 else ("LOST" if res_out == 0.0 else "50/50")
                status_str = f"RESOLVED ({pos_res} | PnL: ${realized_pnl:+,.2f})"
            elif pos.get("_exited"):
                status_str = f"EXITED EARLY (Sold | PnL: ${realized_pnl:+,.2f} | Live: {cur_price*100:.1f}%)"
            else:
                status_str = f"OPEN (Live: {cur_price*100:.1f}% | PnL: ${realized_pnl:+,.2f})"

            print(f"         Position: {outcome_str}  |  Entry Price (pm): {avg_price*100:.1f}%  |  Cost Basis: ${initial_val:,.2f}  |  Status: {status_str}")

        go_back = False
        while True:
            if skip_all_remaining:
                lower, upper = avg_price, avg_price
                break
            pu_input = input("         Enter your subjective probability bounds % (e.g., 60-70 or 65), 'b' (back), 'a' (skip all), 'q' (quit), or enter to skip: ").strip()
            if pu_input.lower() in ('b', 'back', 'r', 'redo'):
                if idx > 0:
                    idx -= 1
                    if idx in imported_records:
                        prev_cat, prev_mid, prev_entry = imported_records.pop(idx)
                        if prev_mid in history.get(prev_cat, {}) and prev_entry in history[prev_cat][prev_mid]:
                            history[prev_cat][prev_mid].remove(prev_entry)
                            if not history[prev_cat][prev_mid]:
                                history[prev_cat].pop(prev_mid, None)
                        if prev_cat == "resolved":
                            imported_resolved = max(0, imported_resolved - 1)
                        elif prev_cat == "exited":
                            imported_exited = max(0, imported_exited - 1)
                        else:
                            imported_predicted = max(0, imported_predicted - 1)
                    print("\n[!] Rolling back to previous question...")
                    go_back = True
                    break
                else:
                    print("         [!] Cannot go back. This is the first question to import.\n")
                    continue
            elif pu_input.lower() in ('q', 'quit'):
                print("\n[!] Quitting position import...")
                idx = len(to_import)
                break
            elif pu_input.lower() == 'a':
                skip_all_remaining = True
                lower, upper = avg_price, avg_price
                break
            elif pu_input.lower() in ('s', 'skip') or not pu_input:
                lower, upper = avg_price, avg_price
                break
            else:
                try:
                    lower, upper = parse_user_input(pu_input)
                    break
                except ValueError as e:
                    print(f"         [!] Error: {e}. Please try again (or enter 'b' to go back, enter to skip).")
                except Exception as e:
                    print(f"         [!] Error: {e}. Please try again.")

        if go_back:
            continue
        if idx >= len(to_import):
            break

        pu_val = round((lower + upper) / 2.0, 6)
        dynamic_ego = compute_dynamic_ego(base_ego, upper - lower, 0.0)
        lm = calculate_logistic_metrics(pu_val, avg_price, dynamic_ego)
        edge = lm["active_edge"]
        kelly_val = lm["model_kelly"]
        _, apy = calculate_annualized_yield(edge, lm["win_prob_m"], days_until)

        is_exited_flag = bool(pos.get("_exited"))
        pred_entry = {
            "question": question,
            "slug": slug,
            "date": pos.get("endDate", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
            "pu": pu_val,
            "pm": avg_price,
            "cur_pm": cur_price,
            "dynamic_ego": dynamic_ego,
            "kelly": kelly_val,
            "cost_basis": round(0.0 if is_exited_flag else initial_val, 4),
            "realized_pnl": realized_pnl,
            "edge": edge,
            "apy": apy,
            "conditionId": cond_id,
            "position": outcome_str,
            "outcomeIndex": outcome_idx,
            "source": "wallet",
            "exited": is_exited_flag
        }
        if latest_trade_ts > 0:
            pred_entry["last_synced_ts"] = latest_trade_ts

        if pos.get("redeemable"):
            pred_entry["outcome"] = determine_redeemable_outcome(pos, m)
            history["resolved"].setdefault(market_id, []).append(pred_entry)
            imported_resolved += 1
            imported_records[idx] = ("resolved", market_id, pred_entry)
        elif is_exited_flag:
            history["exited"].setdefault(market_id, []).append(pred_entry)
            imported_exited += 1
            imported_records[idx] = ("exited", market_id, pred_entry)
        else:
            history["predicted"].setdefault(market_id, []).append(pred_entry)
            imported_predicted += 1
            imported_records[idx] = ("predicted", market_id, pred_entry)

        idx += 1

    # Tag & collect unexecuted local predictions (kept by default; removal available in Delete Data menu)
    local_unexecuted = []
    for m_id in list(history.get("predicted", {}).keys()):
        preds_list = ensure_list(history["predicted"][m_id])
        for p in preds_list:
            if p.get("source") != "wallet":
                p["source"] = "local"
                local_unexecuted.append(p)

    print(f"\n[+] Sync complete: {imported_predicted} active imported, {imported_exited} exited imported, {imported_resolved} resolved imported, {updated_existing} existing synced.")
    if local_unexecuted:
        print(f"[!] Unexecuted Local Predictions (Not in Polymarket Wallet): {len(local_unexecuted)}")
        for lp in local_unexecuted:
            q_short = lp.get("question", "Unknown")
            if len(q_short) > 65:
                q_short = q_short[:62] + "..."
            lp_link = f"https://polymarket.com/event/{lp.get('slug', 'unknown')}"
            q_linked = f"\033]8;;{lp_link}\033\\{q_short}\033]8;;\033\\"
            k_pct = float(lp.get("kelly", 0.0)) * 100.0
            pu_pct = float(lp.get("pu", 0.5)) * 100.0
            pm_pct = float(lp.get("pm", 0.5)) * 100.0
            print(f"    - [LOCAL] ({lp.get('date', 'unknown')}) {q_linked} | pu: {pu_pct:.1f}% | pm: {pm_pct:.1f}% | Target Kelly: {k_pct:.2f}%")
    return history

def load_history():
    global BANKROLL
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f: 
                data = json.load(f)
                if isinstance(data, list) or "seen_market_ids" in data:
                    return {"predicted": {}, "exited": {}, "skipped": {}, "resolved": {}}
                data.setdefault("predicted", {})
                data.setdefault("exited", {})
                data.setdefault("skipped", {})
                data.setdefault("resolved", {})
                # Auto-migrate any exited entries out of "predicted" into "exited"
                for m_id in list(data["predicted"].keys()):
                    preds = ensure_list(data["predicted"][m_id])
                    active_p = [p for p in preds if not p.get("exited")]
                    exited_p = [p for p in preds if p.get("exited")]
                    if exited_p:
                        data["exited"].setdefault(m_id, []).extend(exited_p)
                        if active_p:
                            data["predicted"][m_id] = active_p
                        else:
                            data["predicted"].pop(m_id, None)
                if "wallet_summary" in data and isinstance(data["wallet_summary"], dict):
                    saved_br = data["wallet_summary"].get("bankroll")
                    if saved_br and saved_br > 0:
                        BANKROLL = float(saved_br)
                return data
        except json.JSONDecodeError:
            pass
    return {"predicted": {}, "exited": {}, "skipped": {}, "resolved": {}}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f: 
        json.dump(history, f, indent=4)

def extract_prices(m):
    prices = parse_outcome_prices(m)
    if len(prices) < 1:
        return None, None, None
    pm_mid = prices[0]
    try: 
        pm_bid = float(m.get('bestBid', pm_mid))
        pm_ask = float(m.get('bestAsk', pm_mid))
    except: 
        pm_bid, pm_ask = pm_mid, pm_mid
    return pm_bid, pm_ask, pm_mid

def update_resolutions(history):
    resolved_count = 0
    history.setdefault("exited", {})
    history.setdefault("resolved", {})
    open_buckets = [("predicted", list(history.get("predicted", {}).keys())), ("exited", list(history.get("exited", {}).keys()))]
    total_to_audit = sum(len(k_list) for _, k_list in open_buckets)
    if total_to_audit:
        print(f"[*] Auditing {total_to_audit} unresolved market(s) (active + exited) for resolution...")

    for bucket_name, market_ids in open_buckets:
        for market_id in market_ids:
            if str(market_id).startswith("custom_"):
                continue
            try:
                resp = api.get(f"{GAMMA_API}/markets/{market_id}", timeout=10).json()
                is_closed = resp.get("closed", False)
                uma_resolved = resp.get("umaResolutionStatus") == "resolved"

                if is_closed or uma_resolved:
                    prices = parse_outcome_prices(resp)
                    if len(prices) < 2:
                        prices = [0.5, 0.5]

                    pred_list = ensure_list(history[bucket_name][market_id])
                    history["resolved"].setdefault(market_id, [])

                    for pred_data in pred_list:
                        if pred_data.get("source") == "wallet":
                            try:
                                out_idx = int(pred_data.get("outcomeIndex", 0))
                            except Exception:
                                out_idx = 0
                            if len(prices) > out_idx:
                                outcome = 1.0 if prices[out_idx] >= 0.99 else (0.0 if prices[out_idx] <= 0.01 else 0.5)
                            else:
                                outcome = 0.5
                        else:
                            if prices[0] >= 0.99: outcome = 1.0
                            elif prices[1] >= 0.99: outcome = 0.0
                            else: outcome = 0.5
                        resolved_entry = dict(pred_data)
                        resolved_entry["outcome"] = outcome
                        history["resolved"][market_id].append(resolved_entry)

                    del history[bucket_name][market_id]
                    resolved_count += 1
            except Exception:
                continue

    if resolved_count > 0:
        print(f"[+] Auto-resolved and scored {resolved_count} closed market(s).")
    return history

def print_review_table(history, sub_mode):
    base_ego = calculate_base_ego(history, verbose=False)
    TABLE_CONFIGS = {
        "predicted": {
            "headers": ["Date", "Src", "Side", "Question", "Kelly", "EffK", "pu", "p_norm", "pm", "Live", "Target", "Action / Signal"],
            "widths": [10, 6, 6, 40, 7, 7, 6, 6, 6, 6, 6, 22],
            "aligns": ["left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left"],
            "q_idx": 3
        },
        "exited": {
            "headers": ["Date", "Side", "Question", "Kelly", "pu", "p_norm", "Entry", "Live", "Realized PnL", "Status"],
            "widths": [10, 6, 44, 7, 6, 6, 6, 6, 12, 18],
            "aligns": ["left", "left", "left", "right", "right", "right", "right", "right", "right", "left"],
            "q_idx": 2
        },
        "resolved": {
            "headers": ["Date", "Src", "Side", "Question", "Kelly", "pu", "p_norm", "pm", "Realized PnL", "Result"],
            "widths": [10, 6, 11, 42, 7, 6, 6, 6, 12, 6],
            "aligns": ["left", "left", "left", "left", "right", "right", "right", "right", "right", "left"],
            "q_idx": 3
        },
        "skipped": {
            "headers": ["Date", "Question"],
            "widths": [16, 80],
            "aligns": ["left", "left"],
            "q_idx": 1
        },
        "all": {
            "headers": ["State", "Side", "Date", "Question", "Kelly", "pu", "p_norm", "pm", "Target", "Status / Result"],
            "widths": [8, 9, 10, 42, 7, 6, 6, 6, 6, 22],
            "aligns": ["left", "left", "left", "left", "right", "right", "right", "right", "right", "left"],
            "q_idx": 3
        }
    }

    rows = []

    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        for market_id, value in history.get("predicted", {}).items():
            for pred in ensure_list(value):
                pu_v = float(pred.get("pu", 0.5))
                pm_v = float(pred.get("pm", 0.5))
                cur_pm_v = pred.get("cur_pm")
                k_v = float(pred.get("kelly", 0.0))
                ego_v = float(pred.get("dynamic_ego", base_ego))
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v, cur_pm=cur_pm_v)
                tgt_v = lm["target_win_m"]
                if pred.get("source", "local") != "wallet":
                    regime_str = f"EXEC BUY({lm['live_model_kelly']*100:.1f}%)" if lm["live_active_edge"] > 0 else "SKIP: EDGE GONE"
                elif lm["regime"] == "HALF-SLOPE":
                    regime_str = f"SELL: HALF({lm['eval_win_m']*100:.1f}%)"
                elif lm["regime"] == "DECEL-EXIT":
                    regime_str = f"SELL: DECEL({lm['eval_win_m']*100:.1f}%)"
                elif lm["live_active_edge"] <= 0:
                    regime_str = f"SELL: OVER-TGT({lm['live_active_edge']*100:+.1f}%)"
                elif lm["delta_kelly"] >= 0.01:
                    regime_str = f"ADD +{lm['delta_kelly']*100:.1f}% KELLY"
                else:
                    regime_str = f"HOLD(+{lm['dist_to_target']*100:.1f}%->{tgt_v*100:.1f}%)"
                rows.append({
                    "Mode": "active",
                    "Source": pred.get("source", "local").upper(),
                    "Side": pred.get("position", "Yes" if lm["is_yes"] else "No"),
                    "Market ID": market_id,
                    "Date": str(pred.get("date", "unknown"))[:10],
                    "Question": pred.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{pred.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "Edge": lm["live_active_edge"],
                    "DistDecel": lm["dist_to_target"],
                    "pu": pu_v,
                    "p_norm": lm["pu_norm"],
                    "pm": pm_v,
                    "cur_pm": float(cur_pm_v) if cur_pm_v is not None else pm_v,
                    "Target": tgt_v,
                    "PnL": pred.get("realized_pnl"),
                    "Regime": regime_str,
                    "Outcome": ""
                })

    if sub_mode in ["exited", "all"]:
        for market_id, value in history.get("exited", {}).items():
            for pred in ensure_list(value):
                pu_v = float(pred.get("pu", 0.5))
                pm_v = float(pred.get("pm", 0.5))
                cur_pm_v = pred.get("cur_pm")
                k_v = float(pred.get("kelly", 0.0))
                ego_v = float(pred.get("dynamic_ego", base_ego))
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v, cur_pm=cur_pm_v)
                rows.append({
                    "Mode": "exited",
                    "Source": pred.get("source", "wallet").upper(),
                    "Side": pred.get("position", "Yes" if lm["is_yes"] else "No"),
                    "Market ID": market_id,
                    "Date": str(pred.get("date", "unknown"))[:10],
                    "Question": pred.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{pred.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "Edge": lm["active_edge"],
                    "DistDecel": lm["dist_to_target"],
                    "pu": pu_v,
                    "p_norm": lm["pu_norm"],
                    "pm": pm_v,
                    "cur_pm": float(cur_pm_v) if cur_pm_v is not None else pm_v,
                    "Target": lm["target_win_m"],
                    "PnL": float(pred.get("realized_pnl", 0.0) or 0.0),
                    "Regime": "EXITED (Await Res)",
                    "Outcome": ""
                })

    if sub_mode in ["skipped", "predicted_skipped"]:
        for market_id, value in history.get("skipped", {}).items():
            rows.append({
                "Mode": "skipped",
                "Source": "-",
                "Side": "-",
                "Market ID": market_id,
                "Date": str(value.get("date", "unknown"))[:16],
                "Question": value.get("question", "unknown"),
                "Web Link": f"https://polymarket.com/event/{value.get('slug', 'unknown')}"
            })

    if sub_mode in ["resolved", "all"]:
        for market_id, value in history.get("resolved", {}).items():
            for res in ensure_list(value):
                pu_v = float(res.get("pu", 0.5))
                pm_v = float(res.get("pm", 0.5))
                k_v = float(res.get("kelly", 0.0))
                ego_v = float(res.get("dynamic_ego", base_ego))
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v)
                rows.append({
                    "Mode": "resolved",
                    "Source": res.get("source", "local").upper(),
                    "Side": res.get("position", "Yes" if lm["is_yes"] else "No"),
                    "Market ID": market_id,
                    "Date": str(res.get("date", "unknown"))[:10],
                    "Question": res.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{res.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "Edge": lm["active_edge"],
                    "DistDecel": lm["dist_to_target"],
                    "pu": pu_v,
                    "p_norm": lm["pu_norm"],
                    "pm": pm_v,
                    "cur_pm": pm_v,
                    "Target": lm["target_win_m"],
                    "PnL": res.get("realized_pnl"),
                    "Regime": lm["regime"],
                    "Outcome": str(res.get("outcome", 0.5))
                })

    if not rows:
        print(f"\n[!] No markets found in '{sub_mode}'.")
        return

    sort_by = "date"
    if sub_mode in ["predicted", "exited", "resolved", "all"]:
        print(f"\nSort '{sub_mode}' by: [1] Date (default)  [2] Kelly  [3] Edge  [4] Exit Proximity")
        sort_choice = input("> ").strip()
        sort_by = {"2": "kelly", "3": "edge", "4": "exit_prox"}.get(sort_choice, "date")

    if sort_by == "date":
        rows.sort(key=lambda x: x.get("Date", "") if x.get("Date", "") != "unknown" else "", reverse=True)
    elif sort_by == "kelly":
        rows.sort(key=lambda x: (0 if x.get("Kelly") is None else 1, x.get("Kelly") or 0.0), reverse=True)
    elif sort_by == "edge":
        rows.sort(key=lambda x: (0 if x.get("Edge") is None else 1, x.get("Edge") or 0.0), reverse=True)
    elif sort_by == "exit_prox":
        rows.sort(key=lambda x: (1 if x.get("DistDecel") is None else 0, x.get("DistDecel") if x.get("DistDecel") is not None else float("inf")))

    config = TABLE_CONFIGS.get(sub_mode, TABLE_CONFIGS["all"])
    headers = config["headers"]
    widths = config["widths"]
    aligns = config["aligns"]
    q_idx = config["q_idx"]

    def format_cell(text, width, align):
        return str(text).rjust(width)[:width] if align == "right" else str(text).ljust(width)[:width]

    header_str = " | ".join(format_cell(h, w, a) for h, w, a in zip(headers, widths, aligns))
    separator_str = "-+-".join("-" * w for w in widths)

    print("\n" + "=" * len(header_str))
    print(f" [{sub_mode.upper()} MARKETS - sorted by {sort_by.upper()}] ".center(len(header_str), "="))
    print("=" * len(header_str))
    print(header_str)
    print(separator_str)

    for r in rows:
        kelly_val = r.get("Kelly")
        eff_kelly_val = r.get("EffKelly")
        pu_val = r.get("pu")
        p_norm_val = r.get("p_norm")
        pm_val = r.get("pm")
        cur_pm_val = r.get("cur_pm")
        tgt_val = r.get("Target")
        pnl_val = r.get("PnL")
        regime_val = r.get("Regime", "-")
        side_str = r.get("Side", "-")
        src_str = r.get("Source", "-")

        kelly_str = f"{kelly_val*100:.2f}%" if kelly_val is not None else "-"
        eff_kelly_str = f"{eff_kelly_val*100:.2f}%" if eff_kelly_val is not None else "-"
        pu_str = f"{pu_val*100:.1f}%" if pu_val is not None else "-"
        p_norm_str = f"{p_norm_val*100:.1f}%" if p_norm_val is not None else "-"
        pm_str = f"{pm_val*100:.1f}%" if pm_val is not None else "-"
        live_str = f"{cur_pm_val*100:.1f}%" if cur_pm_val is not None else "-"
        tgt_str = f"{tgt_val*100:.1f}%" if tgt_val is not None else "-"
        pnl_str = f"${pnl_val:+,.2f}" if pnl_val is not None else "-"

        outcome_val = r.get("Outcome")
        is_wallet_row = src_str == "WALLET"
        if outcome_val == "1.0": outcome_str = "WON" if is_wallet_row else "YES"
        elif outcome_val == "0.0": outcome_str = "LOST" if is_wallet_row else "NO"
        elif outcome_val == "0.5": outcome_str = "HALF"
        else: outcome_str = "-"

        if sub_mode == "predicted":
            vals = [r["Date"], src_str, side_str, r["Question"], kelly_str, eff_kelly_str, pu_str, p_norm_str, pm_str, live_str, tgt_str, regime_val]
        elif sub_mode == "exited":
            vals = [r["Date"], side_str, r["Question"], kelly_str, pu_str, p_norm_str, pm_str, live_str, pnl_str, regime_val]
        elif sub_mode == "resolved":
            vals = [r["Date"], src_str, side_str, r["Question"], kelly_str, pu_str, p_norm_str, pm_str, pnl_str, outcome_str]
        elif sub_mode == "skipped":
            vals = [r["Date"], r["Question"]]
        else:
            st_res = f"{outcome_str} ({pnl_str})" if r["Mode"] == "resolved" else (f"EXITED ({pnl_str})" if r["Mode"] == "exited" else regime_val)
            vals = [r["Mode"].upper(), side_str, r["Date"], r["Question"], kelly_str, pu_str, p_norm_str, pm_str, tgt_str, st_res]

        row_parts = []
        for i, (v, w, a) in enumerate(zip(vals, widths, aligns)):
            if i == q_idx:
                q_visible = (v[:w-3] + "...") if len(v) > w else v
                q_padded = q_visible.ljust(w)
                row_parts.append(f"\033]8;;{r['Web Link']}\033\\{q_padded}\033]8;;\033\\")
            else:
                row_parts.append(format_cell(v, w, a))
        print(" | ".join(row_parts))

    print("=" * len(header_str) + "\n")
    if sub_mode in ["predicted", "exited", "resolved", "all"]:
        print_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego)

def calculate_prediction_stats(history, sub_mode="predicted", base_ego=None):
    if base_ego is None:
        base_ego = calculate_base_ego(history, verbose=False)

    preds_list = []
    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        for m_preds in history.get("predicted", {}).values():
            for p in ensure_list(m_preds):
                preds_list.append((p, "active"))
    if sub_mode in ["exited", "all"]:
        for m_ex in history.get("exited", {}).values():
            for p in ensure_list(m_ex):
                preds_list.append((p, "exited"))
    if sub_mode in ["resolved", "all"]:
        for m_res in history.get("resolved", {}).values():
            for p in ensure_list(m_res):
                preds_list.append((p, "resolved"))

    if not preds_list:
        return None

    items = []
    for p, state in preds_list:
        pu = float(p.get("pu", 0.5))
        pm = float(p.get("pm", 0.5))
        cur_pm = p.get("cur_pm")
        kelly = max(0.0, float(p.get("kelly", 0.0)))
        if kelly < 1e-9:
            kelly = 0.0
        ego = float(p.get("dynamic_ego", base_ego))
        outcome = p.get("outcome")
        is_exited = (state == "exited") or bool(p.get("exited"))
        is_res = (state == "resolved")

        lm = calculate_logistic_metrics(pu, pm, ego, held_kelly=kelly, cur_pm=cur_pm)
        win_prob_bal = lm["win_prob_bal"]
        var_i = win_prob_bal * (1.0 - win_prob_bal)
        alloc_i = (kelly * BANKROLL) if state == "active" else 0.0
        eff_alloc_i = (lm["eff_kelly"] * BANKROLL) if state == "active" else 0.0
        items.append({
            "state": state,
            "question": p.get("question", "Unknown"),
            "slug": p.get("slug", "unknown"),
            "position": p.get("position", "Yes" if lm["is_yes"] else "No"),
            "pu": pu,
            "pm": pm,
            "cur_pm": float(cur_pm) if cur_pm is not None else pm,
            "pu_norm": lm["pu_norm"],
            "target_win_m": lm["target_win_m"],
            "dist_to_target": lm["dist_to_target"],
            "delta_kelly": lm["delta_kelly"],
            "kelly": kelly,
            "eff_kelly": lm["eff_kelly"],
            "active_edge": lm["active_edge"],
            "live_active_edge": lm["live_active_edge"],
            "eff_m": lm["eff_m"],
            "eff_live": lm["eff_live"],
            "eff_norm": lm["eff_norm"],
            "regime": "EXITED" if is_exited else lm["regime"],
            "is_exited": is_exited,
            "realized_pnl": float(p.get("realized_pnl", 0.0) or 0.0),
            "alloc": alloc_i,
            "eff_alloc": eff_alloc_i,
            "win_prob_bal": win_prob_bal,
            "win_prob_u": lm["win_prob_u"],
            "win_prob_m": lm["win_prob_m"],
            "var_i": var_i,
            "source": p.get("source", "local"),
            "is_resolved": is_res,
            "outcome": float(outcome) if (is_res and outcome is not None) else None
        })

    sized_items = [x for x in items if x["kelly"] > 0]
    total_kelly = sum(x["kelly"] for x in items)
    total_eff_kelly = sum(x["eff_kelly"] for x in items)
    total_alloc = sum(x["alloc"] for x in items)
    total_eff_alloc = sum(x["eff_alloc"] for x in items)
    total_pnl = sum(x["realized_pnl"] for x in items)
    local_count = sum(1 for x in items if x["source"] != "wallet")
    wallet_count = sum(1 for x in items if x["source"] == "wallet")

    if total_kelly > 0 and sized_items:
        weights = [x["kelly"] / total_kelly for x in sized_items]
        used_items = sized_items
    else:
        n_all = len(items)
        weights = [1.0 / n_all for _ in items]
        used_items = items

    exp_win_bal = sum(w * x["win_prob_bal"] for w, x in zip(weights, used_items))
    exp_win_u = sum(w * x["win_prob_u"] for w, x in zip(weights, used_items))
    exp_win_m = sum(w * x["win_prob_m"] for w, x in zip(weights, used_items))
    avg_eff_m = sum(w * x["eff_m"] for w, x in zip(weights, used_items))
    avg_eff_live = sum(w * x["eff_live"] for w, x in zip(weights, used_items))
    avg_eff_norm = sum(w * x["eff_norm"] for w, x in zip(weights, used_items))
    avg_active_edge = sum(w * x["active_edge"] for w, x in zip(weights, used_items))

    norm_variance = sum((w ** 2) * x["var_i"] for w, x in zip(weights, used_items))
    norm_std_dev = math.sqrt(norm_variance)
    sum_w_sq = sum(w ** 2 for w in weights)
    effective_n = (1.0 / sum_w_sq) if sum_w_sq > 0 else 0.0

    return {
        "total_count": len(items),
        "active_count": len(sized_items),
        "local_count": local_count,
        "wallet_count": wallet_count,
        "total_kelly": total_kelly,
        "total_eff_kelly": total_eff_kelly,
        "total_alloc": total_alloc,
        "total_eff_alloc": total_eff_alloc,
        "total_pnl": total_pnl,
        "bankroll": BANKROLL,
        "exp_win_bal": exp_win_bal,
        "exp_win_u": exp_win_u,
        "exp_win_m": exp_win_m,
        "avg_eff_m": avg_eff_m,
        "avg_eff_live": avg_eff_live,
        "avg_eff_norm": avg_eff_norm,
        "avg_active_edge": avg_active_edge,
        "norm_variance": norm_variance,
        "norm_std_dev": norm_std_dev,
        "effective_n": effective_n,
        "items": items
    }

def print_prediction_stats(history, sub_mode="predicted", base_ego=None):
    stats = calculate_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego)
    if not stats:
        return
    n_denom = stats['active_count'] if stats['active_count'] > 0 else stats['total_count']
    bal_pct = (stats['effective_n'] / n_denom * 100.0) if n_denom > 0 else 0.0
    status_lbl = {"predicted": "active", "exited": "exited", "resolved": "resolved", "all": "total"}.get(sub_mode, "active")
    sized_note = f" ({stats['active_count']} sized)" if stats['active_count'] != stats['total_count'] else ""

    print(f"--- PORTFOLIO & PREDICTION STATS [{status_lbl.upper()}] ---")
    print(f"Predictions      : {stats['total_count']} {status_lbl}{sized_note} ({stats['local_count']} LOCAL, {stats['wallet_count']} WALLET)")
    print(f"Effective N      : {stats['effective_n']:.2f} / {n_denom} equal-sized bets ({bal_pct:.1f}% size-balanced)")
    w_sum = history.get("wallet_summary")
    w_addr = history.get("wallet_address") or POLYGON_WALLET
    if isinstance(w_sum, dict) and "cash" in w_sum and "portfolio" in w_sum:
        w_label = f"\033]8;;https://polymarket.com/profile/{w_addr}\033\\Wallet Bankroll\033]8;;\033\\" if w_addr else "Wallet Bankroll"
        print(f"{w_label}  : ${stats['bankroll']:,.2f} (Cash: ${w_sum['cash']:,.2f} | Open Positions: ${w_sum['portfolio']:,.2f})")

    if sub_mode == "predicted":
        print(f"Model Allocation : ${stats['total_alloc']:,.2f} / ${stats['bankroll']:,.2f} ({stats['total_kelly']*100:.1f}% Kelly | Eff-Scaled: ${stats['total_eff_alloc']:,.2f} / {stats['total_eff_kelly']*100:.1f}%)")
    else:
        avg_k = stats['total_kelly'] / max(1, stats['active_count'])
        wins = sum(1 for x in stats["items"] if x["outcome"] == 1.0)
        losses = sum(1 for x in stats["items"] if x["outcome"] == 0.0)
        wl_str = f" | Record: {wins}W - {losses}L" if sub_mode == "resolved" else ""
        print(f"Realized PnL     : ${stats['total_pnl']:+,.2f}{wl_str} | Avg Sizing: {avg_k*100:.2f}% Kelly")

    print(f"Expected Win Rate: {stats['exp_win_bal']*100:.2f}% Balanced (User: {stats['exp_win_u']*100:.2f}% | Market: {stats['exp_win_m']*100:.2f}%)")
    print(f"Prob. Variance   : {stats['norm_variance']:.4f} (Std Dev: +/-{stats['norm_std_dev']*100:.2f}%)")
    if sub_mode == "resolved":
        print(f"Logistic Eff.    : Entry {stats['avg_eff_m']*100:.1f}% | Pooled (p_norm) {stats['avg_eff_norm']*100:.1f}% | Avg Entry Edge: {stats['avg_active_edge']*100:+.2f}%")
    else:
        print(f"Logistic Eff.    : Entry {stats['avg_eff_m']*100:.1f}% | Live {stats['avg_eff_live']*100:.1f}% | Pooled (p_norm) {stats['avg_eff_norm']*100:.1f}% | Avg Edge: {stats['avg_active_edge']*100:+.2f}%")
    print(f"Exit Landmarks   : Max Decel: {LOGISTIC_DECEL_LOW*100:.1f}% & {LOGISTIC_DECEL_HIGH*100:.1f}% (66.7% Eff) | Half-Slope: {LOGISTIC_HALF_LOW*100:.1f}% & {LOGISTIC_HALF_HIGH*100:.1f}% (50.0% Eff)\n")

def calculate_base_ego(history, verbose=True):
    resolved = history.get("resolved", {})
    if not resolved:
        return 0.50
        
    bs_u_total, bs_m_total = 0.0, 0.0
    total_weight = 0.0
    
    for market_preds in resolved.values():
        market_preds = ensure_list(market_preds)
        for data in market_preds:
            outcome = data.get("outcome", 0)
            
            # Fetch the Kelly fraction used
            weight = data.get("kelly", 0.0)
            
            if weight <= 1e-9: 
                continue 
                
            # Kelly-Weighted Brier Penalty
            bs_u_total += weight * ((data.get("pu", 0.5) - outcome) ** 2)
            bs_m_total += weight * ((data.get("pm", 0.5) - outcome) ** 2)
            total_weight += weight
            
    # If there are resolved markets, but all of them had $0 allocated
    if total_weight == 0:
        return 0.50

    bs_u_total /= total_weight
    bs_m_total /= total_weight
        
    if verbose:
        print(f"[*] Current Brier Score - User: {bs_u_total:.4f} | Market: {bs_m_total:.4f}")
    
    if bs_u_total == 0 and bs_m_total == 0:
        return 0.50
    return bs_m_total / (bs_u_total + bs_m_total)

def calculate_annualized_yield(edge, true_price, days_until):
    # Prevent division by zero for expired markets, 0 prices, or float dust
    if days_until <= 0.0 or true_price <= 0 or abs(edge) < 1e-9:
        return 0.0, 0.0
        
    # 1. Raw Expected Return on Investment
    roi = edge / true_price
    
    # 2. Annualized Yield (APY - Compounding)
    # If the edge is negative, the APY will geometrically compound the loss
    try:
        apy = -1.0 if roi <= -1.0 else ((1 + roi) ** (365.0 / days_until)) - 1
        if isinstance(apy, complex): apy = -1.0
    except:
        apy = 0.0
        
    return roi, apy



def parse_user_input(user_input):
    # Normalize range hyphens into spaces so they aren't mistaken for negative numbers
    # e.g., "10 - 20" -> "10   20"
    cleaned = user_input.replace(' - ', ' ')
    # e.g., "16-51" -> "16 51" (hyphen preceded by a digit)
    cleaned = re.sub(r'(?<=\d)-', ' ', cleaned)
    # e.g., "10%-20%" -> "10% 20%" (hyphen preceded by a % sign)
    cleaned = re.sub(r'(?<=%)-', ' ', cleaned)
    
    # Now extract the remaining numbers
    str_numbers = re.findall(r'-?\d+\.?\d*', cleaned)
    
    if not str_numbers:
        raise ValueError("Invalid input")
        
    numbers = []
    # Validate extracted strings directly to catch "-0" before it turns into a float
    for s in str_numbers[:2]:
        if s.startswith('-'):
            raise ValueError(f"Negative values ({s}) are not allowed. Please enter 0-100.")
        
        num = float(s)
        if num < 0 or num > 100:
            raise ValueError(f"Value '{num}' is out of bounds. Percentages must be between 0 and 100")
        numbers.append(num)
        
    if len(numbers) == 1:
        return round(numbers[0] / 100.0, 6), round(numbers[0] / 100.0, 6)
    else:
        return round(min(numbers[0], numbers[1]) / 100.0, 6), round(max(numbers[0], numbers[1]) / 100.0, 6)

def calculate_allocation(lower_bound, upper_bound, pm_bid, pm_ask, fee_rate, bankroll, base_ego, custom_min_edge=None):
    if custom_min_edge is None:
        custom_min_edge = MIN_EDGE
    
    # 1. Midpoints and Uncertainty Spreads
    pu_mid = (lower_bound + upper_bound) / 2.0
    u_spread = upper_bound - lower_bound
    m_spread = pm_ask - pm_bid
    pm_mid = (pm_bid + pm_ask) / 2.0
    
    # 2. Bidirectional Conviction Weighting & Bayesian Logit Pooling
    dynamic_ego = compute_dynamic_ego(base_ego, u_spread, m_spread)
    lm = calculate_logistic_metrics(pu_mid, pm_mid, dynamic_ego, fee_rate=fee_rate)
    pu_norm = lm["pu_norm"]
    
    # 3. Pricing and Action Mapping
    cost_yes, cost_no = pm_ask, 1.0 - pm_bid 
    
    if pu_norm > cost_yes:
        action, base_price, win_prob = "YES", cost_yes, pu_norm
    elif (1.0 - pu_norm) > cost_no:
        action, base_price, win_prob = "NO", cost_no, 1.0 - pu_norm
    else:
        return "NONE", 0.0, 0.0, 0.0, dynamic_ego, 0.0
    
    # 4. Edge and Fee Evaluation
    true_price = base_price + (fee_rate * base_price * (1 - base_price))
    edge = win_prob - true_price

    # 5. Uncapped Fractional Kelly Allocation
    raw_kelly = max(0.0, edge / (1 - true_price)) if true_price < 1.0 else 0.0
    final_allocation = raw_kelly * bankroll
    
    if edge < custom_min_edge:
        return "THIN_EDGE", true_price, raw_kelly, 0.0, dynamic_ego, edge
    
    return action, true_price, raw_kelly, final_allocation, dynamic_ego, edge

def find_platform_brink(limit=100):
    print("[*] Probing platform boundaries via binary search...")
    max_guess = 2**16 # Set a large number as initial upper bound for offset
    low, high, max_valid = 0, max_guess, 0
    while low <= high:
        mid = ((low + high) // 2) // limit * limit
        try:
            res = api.get(f"{GAMMA_API}/events?active=true&closed=false&limit={limit}&offset={mid}").json()
            if isinstance(res, dict) and 'error' in res:
                high = mid - limit
            elif isinstance(res, list) and len(res) > 0:
                max_valid, low = mid, mid + limit
            else:
                high = mid - limit 
        except Exception:
            high = mid - limit
    return max_valid

# --- UNIFIED PIPELINE ARCHITECTURE ---

def validate_market(m, history, mode, exclude_mode="none"):
    if m.get('closed') or m.get('umaResolutionStatus') == 'resolved':
        return False, "Market closed or resolved"
    if not m.get('active'):
        return False, "Market betting paused (Awaiting UMA resolution)"
        
    try:
        outcomes = json.loads(m.get('outcomes', '["Yes", "No"]'))
        if len(outcomes) != 2:
            return False, f"Non-binary categorical market ({len(outcomes)} outcomes)"
    except:
        return False, "Invalid outcomes structure"

    market_id = m.get('id')
    
    # --- DYNAMIC EXCLUSION FILTER ---
    if mode == "discover":
        if (market_id in history.get("predicted", {}) or
            market_id in history.get("exited", {}) or
            market_id in history.get("resolved", {}) or
            market_id in history.get("skipped", {})):
            return False, "Already in history"
    elif mode == "target":
        if exclude_mode in ["predicted", "both"] and (market_id in history.get("predicted", {}) or market_id in history.get("exited", {})):
            return False, "Already predicted"
        if exclude_mode in ["skipped", "both"] and market_id in history.get("skipped", {}):
            return False, "Already skipped"
            
    _, _, pm_mid = extract_prices(m)
    if pm_mid is None:
        return False, "Broken or missing price data from API"
    if pm_mid < EXTREME_ODDS or pm_mid > (1.0 - EXTREME_ODDS):
        return False, f"Odds ({pm_mid*100:.1f}%) hit extreme tail-risk filter"
        
    try:
        res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
        seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()

        if seconds_left <= 0:
            return False, "Market has expired"
        
        days_until = seconds_left / 86400.0
        if days_until < MIN_DAYS or days_until > MAX_DAYS:
            return False, f"Time horizon ({days_until:.1f} days) outside bounds"
    except:
        return False, "Invalid end date format"
            
    return True, "Valid"

def generate_market_stream(mode, sub_mode, target_slugs, history, category="All", max_offset=0, exclude_mode="none", discovery_order="volume"):
    if mode == "discover":
        seen_categories = set()
        seen_market_ids = set()
        while True:
            offset = random.randint(0, max_offset // 100) * 100
            print(f"[*] Fetching market...",  end="\r")
            
            if discovery_order == "multi":
                order_param, ascending_param = random.choice([
                    ("volume", "false"),        # High volume
                    ("volume", "true"),         # Low volume / niche
                    ("volume24hr", "false"),    # High 24h volume
                    ("liquidity", "false"),     # Deep liquidity
                    ("endDate", "true"),        # Ending soonest
                    ("endDate", "false"),       # Far horizon
                    ("competitive", "false"),   # Close 50/50 odds
                    ("startDate", "false")      # Freshly listed
                ])
            elif discovery_order == "volume_asc":
                order_param, ascending_param = "volume", "true"
            elif discovery_order == "startDate":
                order_param, ascending_param = "startDate", "false"
            else:
                order_param = discovery_order
                ascending_param = "true" if discovery_order == "endDate" else "false"
                
            try:
                # Push MIN_DAYS/MAX_DAYS window to the API so out-of-range markets are
                # never fetched. end_date_min also prevents returning expired events.
                _now = datetime.now(timezone.utc)
                _min_dt = (_now + timedelta(days=MIN_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
                _max_dt = (_now + timedelta(days=MAX_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
                events = api.get(f"{GAMMA_API}/events?active=true&closed=false&limit=100&offset={offset}&order={order_param}&ascending={ascending_param}&end_date_min={_min_dt}&end_date_max={_max_dt}").json()
            except Exception:
                continue
            
            if not events or not isinstance(events, list): continue
            random.shuffle(events)
            
            for event in events:
                if not isinstance(event, dict): continue
                # --- CATEGORY FILTERING ---
                if category and category.lower() != "all":
                    tags = [t.get('label', '').lower() for t in event.get('tags', []) if isinstance(t, dict)]
                    matches = [t for t in tags if category.lower() in t]
                    if not matches:
                        continue
                        
                    # Print matched categories once during discovery
                    for match in matches:
                        full_name = match.title()
                        if full_name not in seen_categories:
                            print(f"[+] Category Matched: '{full_name}'")
                            seen_categories.add(full_name)

                valid_markets = []
                for m in event.get('markets', []):
                    m_id = m.get('id')
                    if m_id in seen_market_ids:
                        continue
                        
                    m['parent_slug'] = event.get('slug', 'unknown')
                    m['parent_name'] = event.get('title', 'unknown')
                    
                    is_valid, reason = validate_market(m, history, mode, exclude_mode)
                    if is_valid: 
                        valid_markets.append(m)
                    elif reason.startswith("Non-binary"):
                        print(f"[-] Ignored Categorical Market: '{m.get('question')}' | Reason: {reason}")
                        print(f"URL: https://polymarket.com/market/{m.get('id')}\n")
                
                if valid_markets:
                    chosen_m = random.choice(valid_markets)
                    seen_market_ids.add(chosen_m.get('id'))
                    yield chosen_m 
                    
    elif mode == "target":
        if not target_slugs:
            print("[!] Error: No valid URL/slug provided.")
            return
            
        for slug in target_slugs:
            try:
                events = api.get(f"{GAMMA_API}/events?slug={slug}").json()
                if not events or not isinstance(events, list):
                    print(f"[-] Invalid or empty data returned for slug: {slug}")
                    continue
                    
                for event in events:
                    if not isinstance(event, dict): continue
                    for m in event.get('markets', []):
                        m['parent_slug'] = event.get('slug', 'unknown')
                        m['parent_name'] = event.get('title', 'unknown')
                        is_valid, reason = validate_market(m, history, mode, exclude_mode)
                        if is_valid: yield m
                        else: print(f"[-] Ignored Target Market: '{m.get('question')}' | Reason: {reason}")
            except Exception as e:
                print(f"[!] Error fetching slug {slug}: {e}")
                
    elif mode == "review":
        target_ids = []
        if sub_mode in ["predicted", "predicted_skipped", "all"]: target_ids.extend(list(history.get("predicted", {}).keys()))
        if sub_mode in ["skipped", "predicted_skipped", "all"]: target_ids.extend(list(history.get("skipped", {}).keys()))
            
        if not target_ids:
            print("\n[!] History is empty. No markets available to review.")
            return
            
        random.shuffle(target_ids)
        for market_id in target_ids:
            try:
                # Custom markets have no Polymarket API endpoint — skip lookup
                if str(market_id).startswith("custom_"):
                    continue
                
                m = api.get(f"{GAMMA_API}/markets/{market_id}").json()
                if m:
                    m['parent_slug'] = m.get('slug', 'unknown') 
                    m['parent_name'] = m.get('question', 'unknown')
                    
                    is_valid, reason = validate_market(m, history, mode, exclude_mode)
                    if is_valid: 
                        yield m
                    else:
                        if market_id in history.get("skipped", {}) and ("closed" in reason or "paused" in reason or "has expired" in reason):
                            print(f"[-] Auto-Purged Dead Skipped Market: '{m.get('question')}'")
                            del history["skipped"][market_id]
                        elif market_id in history.get("predicted", {}) and ("paused" in reason):
                            print(f"[-] Review Paused (Awaiting UMA Resolution): '{m.get('question')}'")
                        else:
                            print(f"[-] Ignored Review Market: '{m.get('question')}' | Reason: {reason}")
            except Exception:
                continue

def run_prediction_session(mode="discover", sub_mode="all", target_slugs=None, category="All", max_offset=0, exclude_mode="none", discovery_order="volume"):
    global BANKROLL
    
    history = load_history()
    save_history(stringify_overflow(history))
    
    if mode == "review":
        print_review_table(history, sub_mode)
        if sub_mode in ["resolved", "exited"]:
            return None
        
        active_market_count = 0
        if sub_mode in ["predicted", "predicted_skipped", "all"]:
            active_market_count += len(history.get("predicted", {}))
        if sub_mode in ["skipped", "predicted_skipped"]:
            active_market_count += len(history.get("skipped", {}))
            
        if active_market_count == 0:
            print("[*] No active/open markets to re-predict in this view.")
            return None
            
        review_choice = input(f"Re-predict on the {active_market_count} open market(s) in this selection? (y/n) [Enter=n]: ").strip().lower()
        if review_choice != 'y':
            return None
            
    base_ego = calculate_base_ego(history)
    
    print(f"\n--- Starting Session [{mode.upper()} - {sub_mode.upper()}] ---")
    if mode == "discover":
        print(f"Category Filter : {category.upper()}")
        print(f"Discovery Vector: {discovery_order.upper()}")
    print(f"Bankroll        : ${BANKROLL:,.2f}")
    print(f"Base Ego        : {base_ego:.3f} (Brier Score Ratio)")
    
    market_stream = generate_market_stream(mode, sub_mode, target_slugs, history, category, max_offset, exclude_mode, discovery_order)

    portfolio_data = []
    session_trades = {}
    
    # Seed cumulative exposure from existing predicted history
    cumulative_exposure = 0.0
    for pred_value in history.get("predicted", {}).values():
        preds = ensure_list(pred_value)
        for p in preds:
            cumulative_exposure += p.get("kelly", 0.0)
    print(f"Past Exposure  : {cumulative_exposure*100:.1f}% of Bankroll Used\n")
    session_markets = []
    current_index = 0
    
    while True:
            
        if current_index >= len(session_markets):
            try:
                new_m = next(market_stream)
                
                pm_bid, pm_ask, pm_mid = extract_prices(new_m)
                if pm_mid is None:
                    continue
                new_m['cached_bid'], new_m['cached_ask'], new_m['cached_mid'] = pm_bid, pm_ask, pm_mid
                
                new_m['original_state'] = {
                    "in_pred": new_m.get('id') in history["predicted"],
                    "pred_data": history["predicted"].get(new_m.get('id')),
                    "in_skip": new_m.get('id') in history["skipped"],
                    "skip_data": history["skipped"].get(new_m.get('id')),
                    "portfolio_len": len(portfolio_data)
                }
                session_markets.append(new_m)
                
            except StopIteration:
                print("\n[*] Data stream ended or exhausted.")
                break
            except TypeError:
                break
                
        m = session_markets[current_index]
        # Real-time CLOB update right before display/evaluation
        m = refresh_prices_from_clob(m)
        pm_bid, pm_ask, pm_mid = extract_prices(m)
        m['cached_bid'], m['cached_ask'], m['cached_mid'] = pm_bid, pm_ask, pm_mid
        
        market_id = m.get('id')
        question = m.get('question', 'unknown')
        event_slug = m.get('parent_slug', 'unknown')
        event_url = f"https://polymarket.com/event/{event_slug}"
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        
        try:
            res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
            exact_date_str = res_dt.strftime("%B %d, %Y")
            
            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            days_str = format_time_remaining(seconds_left)
        except: exact_date_str, days_str = "unknown", "unknown"
 
        pm_bid, pm_ask, pm_mid = m['cached_bid'], m['cached_ask'], m['cached_mid']
 
        print(f"==================================================")
        print(f"Event    : {m.get('parent_name')}")
        print(f"URL      : {event_url}")
        print(f"Market   : {question}")
        print(f"Resolves : {exact_date_str} ({days_str})\n")
 
        while True:
            user_input = input("Enter % bounds ('36-48' or '42'), 'l' (live), 's' (skip), 'r' (redo), 'q' (quit): ").strip()
            cmd_parts = user_input.split()
            if cmd_parts and cmd_parts[0].lower() in ('l', 'live'):
                custom_tok = cmd_parts[1] if len(cmd_parts) > 1 else None
                follow_up = stream_clob_book(m, custom_token_id=custom_tok)
                pm_bid, pm_ask, pm_mid = m['cached_bid'], m['cached_ask'], m['cached_mid']
                if follow_up:
                    user_input = follow_up
                    break
                continue
            break
 
        if user_input.lower() == 'q':
            print("\nSaving and quitting session...")
            break
            
        elif user_input.lower() == 'r':
            if current_index > 0:
                print("\n[!] Rolling back to previous market...")
                current_index -= 1 
                prev_m = session_markets[current_index]
                prev_market_id = prev_m.get('id')
                
                if prev_market_id in session_trades:
                    refund_amount = session_trades.pop(prev_market_id)
                    cumulative_exposure -= (refund_amount / BANKROLL)
                
                orig = prev_m['original_state']
                if orig['in_pred']: history["predicted"][prev_market_id] = orig['pred_data']
                else: history["predicted"].pop(prev_market_id, None)
                    
                if orig['in_skip']: history["skipped"][prev_market_id] = orig['skip_data']
                else: history["skipped"].pop(prev_market_id, None)
                
                portfolio_data = portfolio_data[:orig['portfolio_len']]
            else:
                print("\n[!] Cannot redo. This is the first market of the session.\n")
            continue

        elif user_input.lower() == 's': 
            print("Status: Skipped.\n")
            if market_id not in history["predicted"]:
                history["skipped"][market_id] = {
                    "question": question,
                    "slug": event_slug,
                    "date": today_str
                }
            current_index += 1 
            continue
        
        try:
            lower, upper = parse_user_input(user_input)
            fee_rate = (m.get('feeSchedule') or {}).get('rate')
            if fee_rate is None:
                try:
                    f = float(m.get('fee') or 0)
                    fee_rate = f / 1e18 if f > 1 else f
                except:
                    fee_rate = 0.0
            
            action, true_price, raw_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
                lower, upper, pm_bid, pm_ask, fee_rate, BANKROLL, base_ego
            )
            
            try:
                res_dt = datetime.fromisoformat(m.get('endDate', '').replace('Z', '+00:00'))
                seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
                days_until = max(0.0, seconds_left / 86400.0)
            except:
                days_until = 0.0

            roi, apy = calculate_annualized_yield(edge, true_price, days_until)

            # Walk order book to calculate VWAP and slippage if we have an action and allocation
            vwap_true = true_price
            slippage = 0.0
            adjusted_edge = edge
            adjusted_apy = apy
            
            if action in ["YES", "NO"] and final_alloc > 0:
                clob_ids_str = m.get('clobTokenIds')
                asks_list = []
                if clob_ids_str:
                    try:
                        clob_ids = json.loads(clob_ids_str) if isinstance(clob_ids_str, str) else clob_ids_str
                        token_id = None
                        tokens = m.get('tokens', [])
                        if tokens and len(tokens) == len(clob_ids):
                            for idx, tok in enumerate(tokens):
                                if tok.get('outcome', '').upper() == action:
                                    token_id = clob_ids[idx]
                                    break
                        if not token_id:
                            token_id = clob_ids[0] if action == "YES" else clob_ids[1]
                        r = api.get(f"https://clob.polymarket.com/book?token_id={token_id}", timeout=5)
                        if r.status_code == 200:
                            book = r.json()
                            asks_list = book.get('asks', [])
                    except Exception:
                        pass
                if asks_list:
                    vwap, total_spent = calculate_vwap(asks_list, final_alloc)
                    if vwap > 0:
                        vwap_true = vwap + (fee_rate * vwap * (1.0 - vwap))
                        slippage = max(0.0, vwap_true - true_price)
                        
                        # Re-calculate edge and APY using VWAP price
                        win_prob = true_price + edge
                        adjusted_edge = win_prob - vwap_true
                        _, adjusted_apy = calculate_annualized_yield(adjusted_edge, vwap_true, days_until)
                        if adjusted_edge < MIN_EDGE:
                            action = "THIN_EDGE"
            # 1. Ensure the market ID key contains a list
            if market_id not in history["predicted"]:
                history["predicted"][market_id] = []
            else:
                history["predicted"][market_id] = ensure_list(history["predicted"][market_id])
            
            pu_mid_val = (lower + upper) / 2.0
            lm = calculate_logistic_metrics(pu_mid_val, pm_mid, dynamic_ego, held_kelly=raw_kelly, fee_rate=fee_rate)

            # 2. Append the new independent prediction tranche
            history["predicted"][market_id].append({
                "question": question,
                "slug": event_slug,
                "date": today_str,
                "pu": pu_mid_val,
                "pm": pm_mid,
                "dynamic_ego": dynamic_ego,
                "kelly": raw_kelly, # Required for new Weighted Brier calculation
                "edge": adjusted_edge,
                "apy": adjusted_apy,
                "conditionId": m.get("conditionId"),
                "source": "local"
            })
            
            history["skipped"].pop(market_id, None)
            
            print(f"\n--- MARKET ANALYSIS ---")
            print(f"Market Spread: Bid {pm_bid*100:.1f}% | Ask {pm_ask*100:.1f}% (Spread: {(pm_ask - pm_bid)*100:.1f}%)")
            print(f"User Bounds  : {lower*100:.1f}% to {upper*100:.1f}% (Spread: {(upper-lower)*100:.1f}%)")
            print(f"Dynamic Ego  : {dynamic_ego:.2f} (Base {base_ego:.2f}) -> Active Prob (p_norm): {lm['pu_norm']*100:.2f}%")
            print(f"Logistic Eff : Market {lm['eff_m']*100:.1f}% | Active {lm['eff_norm']*100:.1f}% | Regime: {lm['regime']} (+{lm['dist_to_decel']*100:.1f}% to 78.9% Decel)")
            
            if action in ["YES", "NO"]:
                cumulative_exposure += final_alloc / BANKROLL
                session_trades[market_id] = final_alloc
                eff_kelly_val = lm["eff_m"] * raw_kelly
                eff_alloc_val = eff_kelly_val * BANKROLL
                
                print(f"ACTION       : BUY {action} @ {true_price*100:.1f}%")
                if slippage > 0.0001:
                    print(f"VWAP PRICE   : {vwap_true*100:.2f}% (Slippage: +{slippage*100:.2f}%)")
                    print(f"ADJUSTED EDGE: {adjusted_edge*100:.2f}% (Target: {edge*100:.2f}%)")
                    print(f"ADJUSTED APY : {adjusted_apy*100:.2f}% (Target: {apy*100:.2f}%)")
                else:
                    print(f"USER EDGE    : {adjusted_edge*100:.2f}% (After Fees & Spread)")
                    print(f"APY          : {apy*100:.2f}% (Compounded)")
                    
                print(f"KELLY %      : {raw_kelly*100:.2f}% of bankroll (${final_alloc:,.2f})")
                print(f"EFF KELLY %  : {eff_kelly_val*100:.2f}% of bankroll (${eff_alloc_val:,.2f}) [4p(1-p) Scaled]")
                
                portfolio_data.append({
                    "Question": question[:50] + "..",
                    "Action": action,
                    "Ego": f"{dynamic_ego:.2f}",
                    "Price": f"{vwap_true*100:.1f}%",
                    "Kelly": f"{raw_kelly*100:.2f}%",
                    "EffKelly": f"{eff_kelly_val*100:.2f}%",
                    "Alloc": f"${final_alloc:,.2f}"
                })
            else:
                if action == "THIN_EDGE": reason = f"Edge below {MIN_EDGE*100}% threshold"
                elif action == "NONE": reason = "Trapped inside bid-ask spread"
                else: reason = "Unknown"
                
                print(f"ACTION       : $0 Allocation ({reason})")
                print(f"USER EDGE    : {adjusted_edge*100:.2f}% (After Fees & Spread)")
            
            print(f"EXPOSURE     : {cumulative_exposure*100:.1f}% of Bankroll Used")
            print(f"SESSION DATA : {len(session_trades)} predictions | Total Allocated: ${sum(session_trades.values()):,.2f}\n")
            print_prediction_stats(history, sub_mode="predicted", base_ego=base_ego)

            current_index += 1 
                
        except ValueError as e: 
            print(f"\n[!] Error: {e}. Please try again.\n")
            
    save_history(stringify_overflow(history))
    return pd.DataFrame(portfolio_data)

def delete_specific_markets(history):
    pred_count = len(history.get("predicted", {}))
    ex_count = len(history.get("exited", {}))
    res_count = len(history.get("resolved", {}))
    skip_count = len(history.get("skipped", {}))
    
    print("\nSelect Category to Delete From:")
    print(f" 1: Active Predictions ({pred_count})")
    print(f" 2: Exited Positions ({ex_count})")
    print(f" 3: Resolved Markets ({res_count})")
    print(f" 4: Skipped Markets ({skip_count})")
    print(f" 5: All Categories ({pred_count + ex_count + res_count + skip_count})")
    print(" 6: Cancel")
    
    cat_sel = input("> ").strip()
    cat_map = {
        "1": ["predicted"],
        "2": ["exited"],
        "3": ["resolved"],
        "4": ["skipped"],
        "5": ["predicted", "exited", "resolved", "skipped"]
    }
    if cat_sel not in cat_map:
        print("[!] Deletion cancelled.")
        return
    categories = cat_map[cat_sel]

    entries = []
    for cat in categories:
        cat_data = history.get(cat, {})
        for market_id, val in cat_data.items():
            if cat in ["predicted", "exited", "resolved"]:
                preds = ensure_list(val)
                for idx, p in enumerate(preds):
                    entries.append({
                        "category": cat,
                        "market_id": str(market_id),
                        "pred_idx": idx,
                        "total_preds": len(preds),
                        "question": p.get("question", "Unknown"),
                        "date": p.get("date", "Unknown"),
                        "slug": p.get("slug", "")
                    })
            else:  # skipped
                q = val.get("question", "Unknown") if isinstance(val, dict) else "Unknown"
                d = val.get("date", "Unknown") if isinstance(val, dict) else "Unknown"
                s = val.get("slug", "") if isinstance(val, dict) else ""
                entries.append({
                    "category": "skipped",
                    "market_id": str(market_id),
                    "pred_idx": None,
                    "total_preds": 1,
                    "question": q,
                    "date": d,
                    "slug": s
                })
                
    if not entries:
        print("[!] No markets found in selected category.")
        return

    print(f"\n--- SELECT MARKETS TO DELETE ({len(entries)} items) ---")
    for i, e in enumerate(entries, 1):
        cat_tag = e["category"].upper()
        sub_tag = f" (#{e['pred_idx']+1})" if e["total_preds"] > 1 else ""
        q_text = e["question"]
        if len(q_text) > 65:
            q_text = q_text[:62] + "..."
        print(f" [{i}] [{cat_tag}] ID: {e['market_id']}{sub_tag} | {e['date']} | {q_text}")

    prompt_msg = "\nEnter item number(s) (e.g. 1, 3 or 1-3), Market ID, keyword, or 'q' to cancel:\n> "
    user_input = input(prompt_msg).strip()
    
    if not user_input or user_input.lower() in ["q", "quit", "cancel", "exit"]:
        print("[!] Deletion cancelled.")
        return

    to_delete = []
    
    if user_input.lower() == "all":
        to_delete = entries
    else:
        by_id = [e for e in entries if e["market_id"].lower() == user_input.lower()]
        if by_id:
            to_delete = by_id
        else:
            tokens = user_input.replace(",", " ").split()
            selected_indices = set()
            is_numeric_parse = True
            for token in tokens:
                if "-" in token:
                    parts = token.split("-")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        start, end = int(parts[0]), int(parts[1])
                        if start > end:
                            start, end = end, start
                        for idx in range(start, end + 1):
                            if 1 <= idx <= len(entries):
                                selected_indices.add(idx)
                    else:
                        is_numeric_parse = False
                        break
                elif token.isdigit():
                    idx = int(token)
                    if 1 <= idx <= len(entries):
                        selected_indices.add(idx)
                else:
                    is_numeric_parse = False
                    break
            
            if is_numeric_parse and selected_indices:
                to_delete = [entries[i - 1] for i in sorted(selected_indices)]
            else:
                matched = [e for e in entries if user_input.lower() in e["question"].lower() or user_input.lower() in e["slug"].lower()]
                if matched:
                    to_delete = matched
                else:
                    print(f"[!] No entries matching '{user_input}'.")
                    return

    if not to_delete:
        print("[!] No valid items selected for deletion.")
        return

    print(f"\nItem(s) selected for deletion ({len(to_delete)}):")
    for e in to_delete:
        sub_tag = f" (#{e['pred_idx']+1})" if e["total_preds"] > 1 else ""
        print(f" - [{e['category'].upper()}] ID: {e['market_id']}{sub_tag} | {e['question']}")
        
    confirm = input(f"\nConfirm deletion of {len(to_delete)} item(s) from history? (y/n): ").strip().lower()
    if confirm != "y":
        print("[!] Deletion cancelled.")
        return

    by_market = {}
    for e in to_delete:
        key = (e["category"], e["market_id"])
        by_market.setdefault(key, []).append(e["pred_idx"])

    for (cat, m_id), del_indices in by_market.items():
        if cat == "skipped":
            history.get("skipped", {}).pop(m_id, None)
        else:
            cat_dict = history.get(cat, {})
            if m_id in cat_dict:
                curr_list = ensure_list(cat_dict[m_id])
                if None in del_indices or set(del_indices) >= set(range(len(curr_list))):
                    cat_dict.pop(m_id, None)
                else:
                    new_list = [p for idx, p in enumerate(curr_list) if idx not in del_indices]
                    if new_list:
                        cat_dict[m_id] = new_list
                    else:
                        cat_dict.pop(m_id, None)

    save_history(stringify_overflow(history))
    print(f"[+] Successfully deleted {len(to_delete)} item(s) from prediction history.\n")

def clear_specific_category(history):
    pred_count = len(history.get("predicted", {}))
    ex_count = len(history.get("exited", {}))
    res_count = len(history.get("resolved", {}))
    skip_count = len(history.get("skipped", {}))
    
    print("\nSelect Category to Clear:")
    print(f" 1: Active Predictions ({pred_count})")
    print(f" 2: Exited Positions ({ex_count})")
    print(f" 3: Resolved Markets ({res_count})")
    print(f" 4: Skipped Markets ({skip_count})")
    print(" 5: Cancel")
    
    cat_sel = input("> ").strip()
    cat_map = {"1": "predicted", "2": "exited", "3": "resolved", "4": "skipped"}
    if cat_sel not in cat_map:
        print("[!] Cancelled.")
        return
        
    target_cat = cat_map[cat_sel]
    count = len(history.get(target_cat, {}))
    if count == 0:
        print(f"[!] '{target_cat}' is already empty.")
        return
        
    confirm = input(f"WARNING: Type 'CONFIRM' to clear all {count} '{target_cat}' markets: ").strip()
    if confirm == "CONFIRM":
        history[target_cat] = {}
        save_history(stringify_overflow(history))
        print(f"[+] All '{target_cat}' markets cleared from history.\n")
    else:
        print("[!] Category clear cancelled.")

def handle_delete_data():
    history = load_history()
    pred_count = len(history.get("predicted", {}))
    ex_count = len(history.get("exited", {}))
    res_count = len(history.get("resolved", {}))
    skip_count = len(history.get("skipped", {}))
    total_markets = pred_count + ex_count + res_count + skip_count

    print("\n--- DELETE DATA ---")
    print(" 1: Delete all data (Full reset)")
    print(f" 2: Selectively delete specific markets ({total_markets} in history)")
    print(" 3: Clear specific category (Active, Exited, Resolved, or Skipped)")
    print(" 4: Cancel")
    
    sub_choice = input("> ").strip()
    
    if sub_choice == "1":
        confirm = input("WARNING: Type 'CONFIRM' to delete all local data: ").strip()
        if confirm == "CONFIRM":
            if os.path.exists(HISTORY_FILE): 
                os.remove(HISTORY_FILE)
            print("All data deleted.")
        else:
            print("[!] Data deletion cancelled.")
            
    elif sub_choice == "2":
        if total_markets == 0:
            print("[!] History is empty. No data to delete.")
            return
        delete_specific_markets(history)
        
    elif sub_choice == "3":
        if total_markets == 0:
            print("[!] History is empty. No data to delete.")
            return
        clear_specific_category(history)
        
    elif sub_choice in ["4", ""]:
        print("[!] Deletion cancelled.")
    else:
        print("[!] Invalid selection.")

if __name__ == "__main__":
    while True:
        _h_menu = load_history()
        _ego_menu = calculate_base_ego(_h_menu, verbose=False)
        _n_pred = sum(len(ensure_list(v)) for v in _h_menu.get("predicted", {}).values())
        _n_exit = sum(len(ensure_list(v)) for v in _h_menu.get("exited", {}).values())
        _n_res = sum(len(ensure_list(v)) for v in _h_menu.get("resolved", {}).values())
        _n_skip = len(_h_menu.get("skipped", {}))
        print(f"\n[Bankroll: ${BANKROLL:,.2f} | Base Ego: {_ego_menu:.3f} | Active: {_n_pred} | Exited: {_n_exit} | Resolved: {_n_res} | Skipped: {_n_skip}]")
        print(" 1: Discover Markets")
        print(" 2: Review Portfolio & History")
        print(" 3: Target URLs / Slugs")
        print(" 4: Delete Data")
        print(" 5: Sync Polymarket Wallet & Resolutions")
        print(" 6: Custom Market Calculator")
        print(" 7: Resolve Markets (Auto-Audit or Manual)")
        print(" 8: Settings / Filters")
        print(" 9: Exit")
        print(" 0: Market Scanner (analyse.py)")
        
        choice = input("> ").strip()
        
        targets = None
        category = "All"
        session_max_offset = 0
        target_exclude_mode = "none"
        discovery_order = "volume"
        
        if choice == "9":
            print("Exiting program.")
            break
            
        elif choice == "0":
            import analyse
            analyse.run_interactive_analyzer()
            continue
            
        elif choice == "8":
            print("\n--- SETTINGS / FILTERS ---")
            print(f" 1: Min Edge       = {MIN_EDGE*100:.2f}%  (skip bets below this edge)")
            print(f" 2: Max Days       = {MAX_DAYS}  (ignore markets beyond this horizon)")
            print(f" 3: Min Days       = {MIN_DAYS}  (ignore markets resolving sooner than this)")
            print(f" 4: Extreme Odds   = {EXTREME_ODDS*100:.2f}%  (tail-risk filter - skip near 0% or 100%)")
            print(f" 5: Bankroll       = ${BANKROLL:,.2f}  (manual override; overwritten by wallet sync)")
            print("\nEnter setting number to edit [Enter to cancel]:")
            s_choice = input("> ").strip()
            setting_map = {"1": "MIN_EDGE", "2": "MAX_DAYS", "3": "MIN_DAYS", "4": "EXTREME_ODDS", "5": "BANKROLL"}
            if s_choice in setting_map:
                param = setting_map[s_choice]
                current_val = {"MIN_EDGE": MIN_EDGE, "MAX_DAYS": MAX_DAYS, "MIN_DAYS": MIN_DAYS,
                               "EXTREME_ODDS": EXTREME_ODDS, "BANKROLL": BANKROLL}[param]
                new_raw = input(f"New value for {param} (currently {current_val}): ").strip()
                if new_raw:
                    try:
                        new_val = float(new_raw)
                        if param == "MIN_EDGE": MIN_EDGE = new_val
                        elif param == "MAX_DAYS": MAX_DAYS = new_val
                        elif param == "MIN_DAYS": MIN_DAYS = new_val
                        elif param == "EXTREME_ODDS": EXTREME_ODDS = new_val
                        elif param == "BANKROLL": BANKROLL = new_val
                        print(f"[+] {param} updated to {new_val}.")
                    except ValueError:
                        print("[!] Invalid value - setting unchanged.")
            continue
            
        elif choice == "7":
            history = load_history()
            print("\n--- RESOLVE MARKETS ---")
            print(" 1: Auto-Audit Unresolved Markets via Polymarket API")
            print(" 2: Manually Resolve a Specific Market")
            print(" 3: Cancel")
            r_mode = input("> ").strip()
            if r_mode == "1":
                history = update_resolutions(history)
                save_history(stringify_overflow(history))
                continue
            elif r_mode != "2":
                continue

            combined_open = {}
            for m_id, preds in history.get("predicted", {}).items():
                combined_open[m_id] = ("predicted", preds)
            for m_id, preds in history.get("exited", {}).items():
                combined_open[m_id] = ("exited", preds)

            if not combined_open:
                print("[!] No active or exited predictions to resolve.")
                continue
                
            for m_id, (b_name, preds) in combined_open.items():
                q = ensure_list(preds)[0].get("question", "Unknown")
                print(f" [{b_name.upper()}] ID: {m_id} | {q}")
                
            sel = input("\nEnter Market ID to resolve [Enter to cancel]: ").strip()
            if not sel or sel not in combined_open:
                continue
                
            bucket_name, preds = combined_open[sel]
            print(f"\nResolving: {ensure_list(preds)[0].get('question')}")
            print(" 1: WON / YES (1.0)   2: LOST / NO (0.0)   3: HALF (0.5)")
            out_sel = input("> ").strip()
            outcome = {"1": 1.0, "2": 0.0, "3": 0.5}.get(out_sel)
            if outcome is None:
                print("[!] Resolve cancelled.")
                continue
                
            history.setdefault("resolved", {}).setdefault(sel, [])
            for p in ensure_list(preds):
                p["outcome"] = outcome
                history["resolved"][sel].append(p)
                
            history[bucket_name].pop(sel, None)
            save_history(stringify_overflow(history))
            print("[+] Market manually resolved and saved.")
            continue
            
        elif choice == "6":
            print("\n--- CUSTOM MARKET CALCULATOR ---")
            history = load_history()
            ego_mode = input("Custom Base Ego (0.0-1.0) [Enter for Historical Ego]: ").strip()
            try:
                base_ego = float(ego_mode) if (ego_mode and 0.0 <= float(ego_mode) <= 1.0) else calculate_base_ego(history)
            except ValueError:
                base_ego = calculate_base_ego(history)
            
            try:
                m_lower, m_upper = parse_user_input(input("Market Probability Bounds % (e.g., 42-52 or 47): ").strip())
                lower, upper = parse_user_input(input("User Probability Bounds % (e.g., 16-42 or 29): ").strip())
                fee_input = input("Fee Rate % [Enter for 0%]: ").strip()
                fee_rate = float(fee_input) / 100.0 if fee_input else 0.0
                days_input = input("Days until resolution [Enter for 0]: ").strip()
                days_until = float(days_input) if days_input else 0.0
                bankroll_input = input(f"Bankroll [Enter for ${BANKROLL:,.2f}]: ").strip()
                calc_bankroll = float(bankroll_input) if bankroll_input else BANKROLL
                
                action, true_price, raw_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
                    lower, upper, m_lower, m_upper, fee_rate, calc_bankroll, base_ego
                )
                _, apy = calculate_annualized_yield(edge, true_price, days_until)
                pu_mid_c = round((lower + upper) / 2.0, 6)
                pm_mid_c = round((m_lower + m_upper) / 2.0, 6)
                lm_c = calculate_logistic_metrics(pu_mid_c, pm_mid_c, dynamic_ego, held_kelly=raw_kelly, fee_rate=fee_rate)
                eff_kelly_c = lm_c["eff_m"] * raw_kelly
                eff_alloc_c = eff_kelly_c * calc_bankroll
                
                print(f"\n--- CALCULATION RESULTS ---")
                print(f"ACTIVE PROB  : {lm_c['pu_norm']*100:.2f}% (Eff: Market {lm_c['eff_m']*100:.1f}% | Active {lm_c['eff_norm']*100:.1f}% | Regime: {lm_c['regime']})")
                if action in ["YES", "NO"]:
                    print(f"ACTION       : BUY {action} @ {true_price*100:.1f}% | Edge: {edge*100:+.2f}% | Ego: {dynamic_ego:.2f}")
                    if days_until > 0: print(f"APY          : {apy*100:.2f}%")
                    print(f"KELLY %      : {raw_kelly*100:.2f}% (${final_alloc:,.2f}) | Eff-Kelly: {eff_kelly_c*100:.2f}% (${eff_alloc_c:,.2f})")
                else:
                    print(f"ACTION       : $0 Allocation | Edge: {edge*100:+.2f}% | Ego: {dynamic_ego:.2f}")
                    
                if input("\nSave custom prediction to history? (y/n) [Enter=n]: ").strip().lower() == 'y':
                    q_title = input("Enter title: ").strip() or "Custom Market"
                    custom_id = f"custom_{int(datetime.now().timestamp())}"
                    history.setdefault("predicted", {}).setdefault(custom_id, []).append({
                        "question": f"[CUSTOM] {q_title}",
                        "slug": "custom",
                        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "pu": pu_mid_c,
                        "pm": pm_mid_c,
                        "dynamic_ego": dynamic_ego,
                        "kelly": raw_kelly,
                        "edge": edge,
                        "apy": apy,
                        "conditionId": "custom",
                        "source": "local"
                    })
                    save_history(stringify_overflow(history))
                    print("[+] Saved to active predictions.")
            except ValueError as e:
                print(f"\n[!] Error: {e}.\n")
            continue
            
        elif choice == "5":
            history = load_history()
            history = import_polymarket_positions(POLYGON_WALLET, history)
            history = update_resolutions(history)
            save_history(stringify_overflow(history))
            continue
            
        elif choice == "4":
            handle_delete_data()
            continue
            
        elif choice == "3":
            urls_input = input("Paste Polymarket URLs/Slugs (comma or space separated):\n> ").replace(',', ' ').split()
            targets = []
            for item in urls_input:
                if not item: continue
                if "/event/" in item:
                    targets.append(item.split("/event/")[1].split("/")[0].split("?")[0])
                else:
                    targets.append(item.split("?")[0])
            
            if not targets:
                print("[!] No valid inputs parsed.\n")
                continue
            print(f"Parsed Targets ({len(targets)}): {', '.join(targets)}")
            print("Exclusion Filter: [1] None (default)  [2] Exclude Active/Exited  [3] Exclude Skipped  [4] Exclude Both")
            ex_choice = input("> ").strip()
            target_exclude_mode = {"2": "predicted", "3": "skipped", "4": "both"}.get(ex_choice, "none")
            op_mode, sub_mode = "target", "all"
            
        elif choice == "1":
            category_input = input("\nCategory Tag (e.g., 'Politics', 'Crypto') [Enter for 'All']: ").strip()
            category = category_input if category_input else "All"
            print("Discovery Vector: [1] Vol (default) [2] 24h Vol [3] Liq [4] Soonest [5] Competitive [6] Niche [7] Newest [8] Multi")
            vec_choice = input("> ").strip()
            vec_map = {
                "1": "volume", "2": "volume24hr", "3": "liquidity", "4": "endDate",
                "5": "competitive", "6": "volume_asc", "7": "startDate", "8": "multi"
            }
            discovery_order = vec_map.get(vec_choice, "volume")
            platform_max_offset = find_platform_brink()
            max_pages = (platform_max_offset // 100) + 1
            market_mode = input(f"Reach: [1] Max ({max_pages} pages, default)  [2] Custom Top Pages (1-{max_pages})\n> ").strip()
            op_mode = "discover"
            if market_mode == "2":
                sub_mode = "custom"
                try:
                    custom_limit = int(input(f"Pages (1-{max_pages}): ").strip())
                    session_max_offset = max(0, min(max_pages - 1, custom_limit - 1)) * 100
                except ValueError:
                    session_max_offset = 0
            else:
                sub_mode = "all"
                session_max_offset = platform_max_offset
                
        elif choice == "2":
            print(f"\nReview Portfolio & History:")
            print(f" 1: Active Predictions ({_n_pred})")
            print(f" 2: Exited Positions ({_n_exit})")
            print(f" 3: Resolved Markets ({_n_res})")
            print(f" 4: Skipped Markets ({_n_skip})")
            print(f" 5: All Predictions (Active + Exited + Resolved)")
            review_mode = input("> ").strip()
            if review_mode not in ["1", "2", "3", "4", "5"]:
                review_mode = "1"
            op_mode = "review"
            sub_mode = {"1": "predicted", "2": "exited", "3": "resolved", "4": "skipped", "5": "all"}[review_mode]
                
        else:
            print("[!] Invalid selection.")
            continue
            
        portfolio = run_prediction_session(mode=op_mode, sub_mode=sub_mode, target_slugs=targets, category=category, max_offset=session_max_offset, exclude_mode=target_exclude_mode, discovery_order=discovery_order)
        
        if portfolio is not None:
            if not portfolio.empty:
                print("\n--- Final Session Allocations ---")
                pd.set_option('display.max_colwidth', None) 
                print(portfolio.to_string(index=False))
            else:
                print("\nNo allocations made.")