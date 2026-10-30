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

api = requests.Session()

from utils import ensure_list, parse_outcome_prices, format_time_remaining, stringify_overflow

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
    pos_slug = pos.get("slug")
    m = None

    # 1. Lookup via parent Event ID (accurate & fast, caches multi-market events)
    if event_id:
        event_id_str = str(event_id)
        if event_id_str not in event_cache:
            try:
                ev_resp = api.get(f"{GAMMA_API}/events/{event_id_str}", timeout=5).json()
                event_cache[event_id_str] = ev_resp if isinstance(ev_resp, dict) else {}
            except Exception:
                event_cache[event_id_str] = {}
        ev_data = event_cache.get(event_id_str, {})
        for cand in ev_data.get("markets", []):
            if cand.get("conditionId") == cond_id or (pos_slug and cand.get("slug") == pos_slug):
                m = cand
                break

    # 2. Fallback: Lookup via market slug on Gamma API and verify conditionId
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

    # 3. Extract metadata using Gamma market or position fields directly
    market_id = str(m.get("id")) if (m and m.get("id")) else str(cond_id)
    question = (m.get("question") if m else None) or pos.get("title") or "unknown"
    slug = pos.get("eventSlug") or pos.get("slug") or (m.get("slug") if m else "unknown")
    end_date = (m.get("endDate") if m else None) or pos.get("endDate") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return m, market_id, question, slug, end_date

def determine_redeemable_outcome(pos, m=None):
    if m:
        prices = parse_outcome_prices(m)
        if len(prices) >= 2:
            if prices[0] == 1.0:
                return 1.0
            elif prices[1] == 1.0:
                return 0.0
    try:
        cur_price = float(pos.get("curPrice", 0.5))
        outcome_idx = int(pos.get("outcomeIndex", 0))
        if cur_price >= 0.99:
            return 1.0 if outcome_idx == 0 else 0.0
        elif cur_price <= 0.01:
            return 0.0 if outcome_idx == 0 else 1.0
    except Exception:
        pass
    outcome_str = pos.get("outcome", "Yes")
    return 1.0 if outcome_str == "Yes" else 0.0 if outcome_str == "No" else 0.5

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
    base_ego = calculate_base_ego(history)

    print(f"[*] Syncing Polymarket wallet {wallet_address}...")
    try:
        url = f"https://data-api.polymarket.com/positions?user={wallet_address}"
        positions = api.get(url, timeout=10).json()
        if not isinstance(positions, list):
            positions = []
    except Exception as e:
        print(f"[!] Error fetching positions: {e}")
        positions = []

    print("[*] Calculating Live Cash & Portfolio Values...")
    cash_balance = get_wallet_usdc_balance(wallet_address)
    if cash_balance is None:
        cash_balance = history.get("wallet_summary", {}).get("cash", 0.0)
    portfolio_value = get_wallet_portfolio_value(wallet_address, positions)

    BANKROLL = cash_balance + portfolio_value
    history["wallet_summary"] = {
        "cash": cash_balance,
        "portfolio": portfolio_value,
        "bankroll": BANKROLL,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    }
    print(f"Live Bankroll Updated: ${BANKROLL:,.2f} (Cash: ${cash_balance:,.2f} | Portfolio: ${portfolio_value:,.2f})")

    if not positions:
        print("[*] No positions returned from Polymarket positions endpoint.")
        return history

    print(f"Found {len(positions)} positions. Syncing updates and importing new ones...")
    imported_predicted = 0
    imported_resolved = 0
    updated_existing = 0
    event_cache = {}

    for pos in positions:
        cond_id = pos.get("conditionId")
        if not cond_id:
            continue

        m, market_id, question, slug, end_date = resolve_position_market_metadata(pos, event_cache)
        avg_price = float(pos.get("avgPrice", 0.5))
        size = float(pos.get("size", 0.0))
        initial_val = float(pos.get("initialValue", size * avg_price))
        kelly_val = initial_val / BANKROLL if BANKROLL > 0 else 0.0

        try:
            res_dt = datetime.fromisoformat(str(end_date).replace('Z', '+00:00'))
            if res_dt.tzinfo is None:
                res_dt = res_dt.replace(tzinfo=timezone.utc)
            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            days_until = seconds_left / 86400.0
        except Exception:
            days_until = 0.0

        # Check if already stored in history (under predicted or resolved keys) and update live values
        already_stored = False
        for m_id in list(history.get("predicted", {}).keys()):
            preds_list = ensure_list(history["predicted"][m_id])
            matched = False
            for p in preds_list:
                if p.get("conditionId") == cond_id:
                    already_stored = True
                    matched = True
                    p["question"] = question
                    p["slug"] = slug
                    p["pm"] = avg_price
                    p["kelly"] = kelly_val
                    p["edge"] = p.get("pu", avg_price) - avg_price
                    _, p["apy"] = calculate_annualized_yield(p["edge"], avg_price, days_until)
                    updated_existing += 1
            if matched:
                # Re-key if previously stored under wrong market_id (e.g. legacy conditionId bug)
                target_key = market_id
                if m_id != target_key:
                    history["predicted"].pop(m_id, None)
                    history["predicted"].setdefault(target_key, []).extend(preds_list)
                if pos.get("redeemable"):
                    outcome_val = determine_redeemable_outcome(pos, m)
                    moved = ensure_list(history["predicted"].pop(target_key, []))
                    for p in moved:
                        p["outcome"] = outcome_val
                    history["resolved"].setdefault(target_key, []).extend(moved)
                break

        if not already_stored:
            for m_id in list(history.get("resolved", {}).keys()):
                res_list = ensure_list(history["resolved"][m_id])
                matched = False
                for r in res_list:
                    if r.get("conditionId") == cond_id:
                        already_stored = True
                        matched = True
                        r["question"] = question
                        r["slug"] = slug
                        r["pm"] = avg_price
                        r["kelly"] = kelly_val
                        r["outcome"] = determine_redeemable_outcome(pos, m)
                        updated_existing += 1
                if matched and m_id != market_id:
                    history["resolved"].pop(m_id, None)
                    history["resolved"].setdefault(market_id, []).extend(res_list)
                if matched:
                    break

        if already_stored:
            continue

        outcome_str = pos.get("outcome", "Yes")

        print(f"\n[IMPORT] Question: {question}")
        print(f"         Outcome: {outcome_str}  |  Entry Price (pm): {avg_price*100:.1f}%  |  Cost Basis: ${initial_val:,.2f}")

        pu_input = input("         Enter your subjective probability bounds % (e.g., 60-70 or 65) or press enter to skip: ").strip()
        try:
            if pu_input:
                lower, upper = parse_user_input(pu_input)
            else:
                lower, upper = avg_price, avg_price
        except Exception:
            lower, upper = avg_price, avg_price

        pu_val = (lower + upper) / 2.0
        edge = pu_val - avg_price

        u_spread = upper - lower
        m_spread = 0.0
        u_conviction = base_ego * (1.0 - u_spread)
        m_conviction = (1.0 - base_ego) * (1.0 - m_spread)
        if (u_conviction + m_conviction) > 0:
            dynamic_ego = u_conviction / (u_conviction + m_conviction)
        else:
            dynamic_ego = base_ego

        _, apy = calculate_annualized_yield(edge, avg_price, days_until)

        pred_entry = {
            "question": question,
            "slug": slug,
            "date": pos.get("endDate", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
            "pu": pu_val,
            "pm": avg_price,
            "dynamic_ego": dynamic_ego,
            "kelly": kelly_val,
            "edge": edge,
            "apy": apy,
            "conditionId": cond_id
        }

        if pos.get("redeemable"):
            pred_entry["outcome"] = determine_redeemable_outcome(pos, m)
            if market_id not in history["resolved"]:
                history["resolved"][market_id] = []
            history["resolved"][market_id].append(pred_entry)
            imported_resolved += 1
        else:
            if market_id not in history["predicted"]:
                history["predicted"][market_id] = []
            history["predicted"][market_id].append(pred_entry)
            imported_predicted += 1

    print(f"\n[+] Sync complete: {imported_predicted} open imported, {imported_resolved} resolved imported, {updated_existing} existing updated.")
    return history

def load_history():
    global BANKROLL
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f: 
                data = json.load(f)
                if isinstance(data, list) or "seen_market_ids" in data:
                    return {"predicted": {}, "skipped": {}, "resolved": {}}
                if "resolved" not in data:
                    data["resolved"] = {}
                if "wallet_summary" in data and isinstance(data["wallet_summary"], dict):
                    saved_br = data["wallet_summary"].get("bankroll")
                    if saved_br and saved_br > 0:
                        BANKROLL = float(saved_br)
                return data
        except json.JSONDecodeError:
            pass
    return {"predicted": {}, "skipped": {}, "resolved": {}}

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
    predictions = list(history.get("predicted", {}).keys())
    
    if predictions:
        print(f"[*] Auditing {len(predictions)} open predictions for resolution...")
        
    for market_id in predictions:
        try:
            resp = api.get(f"{GAMMA_API}/markets/{market_id}", timeout=10).json()
            
            is_closed = resp.get("closed", False)
            uma_resolved = resp.get("umaResolutionStatus") == "resolved"
            
            if is_closed or uma_resolved:
                prices = parse_outcome_prices(resp)
                if len(prices) < 2:
                    prices = [0.5, 0.5]
                
                if prices[0] == 1.0: outcome = 1.0
                elif prices[1] == 1.0: outcome = 0.0
                else: outcome = 0.5 
                
                pred_list = ensure_list(history["predicted"][market_id])
                
                if market_id not in history["resolved"]:
                    history["resolved"][market_id] = []
                    
                for pred_data in pred_list:
                    history["resolved"][market_id].append({
                        "question": pred_data.get("question", "unknown"),
                        "slug": pred_data.get("slug", "unknown"),
                        "date": pred_data.get("date", "unknown"),
                        "pu": pred_data.get("pu", 0.5),
                        "pm": pred_data.get("pm", 0.5),
                        "kelly": pred_data.get("kelly", 0.0), # Save the weight
                        "outcome": outcome
                    })
                    
                del history["predicted"][market_id]
                resolved_count += 1
        except Exception:
            continue
            
    if resolved_count > 0:
        print(f"[+] BACKGROUND SYSTEM: Auto-resolved and scored {resolved_count} closed markets.")
    return history

def print_review_table(history, sub_mode):
    TABLE_CONFIGS = {
        "predicted": {
            "headers": ["Date", "Question", "Kelly", "pu", "pm"],
            "widths": [16, 60, 8, 6, 6],
            "aligns": ["left", "left", "right", "right", "right"],
            "q_idx": 1
        },
        "skipped": {
            "headers": ["Date", "Question"],
            "widths": [16, 80],
            "aligns": ["left", "left"],
            "q_idx": 1
        },
        "predicted_skipped": {
            "headers": ["Mode", "Date", "Question", "Kelly", "pu", "pm"],
            "widths": [10, 16, 60, 8, 6, 6],
            "aligns": ["left", "left", "left", "right", "right", "right"],
            "q_idx": 2
        },
        "resolved": {
            "headers": ["Date", "Question", "Kelly", "pu", "pm", "Outcome"],
            "widths": [16, 60, 8, 6, 6, 8],
            "aligns": ["left", "left", "right", "right", "right", "left"],
            "q_idx": 1
        },
        "all": {
            "headers": ["Mode", "Date", "Question", "Kelly", "pu", "pm", "Outcome"],
            "widths": [10, 16, 60, 8, 6, 6, 8],
            "aligns": ["left", "left", "left", "right", "right", "right", "left"],
            "q_idx": 2
        }
    }

    # 1. Gather all items according to sub_mode
    rows = []
    
    # Gather predicted
    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        for market_id, value in history.get("predicted", {}).items():
            preds = ensure_list(value)
            for pred in preds:
                rows.append({
                    "Mode": "predicted",
                    "Market ID": market_id,
                    "Date": pred.get("date", "unknown"),
                    "Question": pred.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{pred.get('slug', 'unknown')}",
                    "Kelly": pred.get("kelly", 0.0),
                    "APY": pred.get("apy", 0.0),
                    "Edge": pred.get("edge", 0.0),
                    "pu": pred.get("pu", 0.5),
                    "pm": pred.get("pm", 0.5),
                    "Outcome": ""
                })
                
    # Gather skipped
    if sub_mode in ["skipped", "predicted_skipped", "all"]:
        for market_id, value in history.get("skipped", {}).items():
            rows.append({
                "Mode": "skipped",
                "Market ID": market_id,
                "Date": value.get("date", "unknown"),
                "Question": value.get("question", "unknown"),
                "Web Link": f"https://polymarket.com/event/{value.get('slug', 'unknown')}"
            })
            
    # Gather resolved
    if sub_mode in ["resolved", "all"]:
        for market_id, value in history.get("resolved", {}).items():
            resolutions = ensure_list(value)
            for res in resolutions:
                rows.append({
                    "Mode": "resolved",
                    "Market ID": market_id,
                    "Date": res.get("date", "unknown"),
                    "Question": res.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{res.get('slug', 'unknown')}",
                    "Kelly": res.get("kelly", 0.0),
                    "pu": res.get("pu", 0.5),
                    "pm": res.get("pm", 0.5),
                    "Outcome": str(res.get("outcome", 0.5))
                })

    if not rows:
        print(f"\n[!] No markets found in history for sub-mode: '{sub_mode}'")
        return

    # 2. Get Sorting Option
    sort_by = "date"
    if sub_mode in ["predicted", "predicted_skipped", "resolved", "all"]:
        print(f"\nSort '{sub_mode}' by:")
        print(" 1: Date (default)")
        print(" 2: Kelly")
        sort_choice = input("> ").strip()
        sort_by = {"2": "kelly"}.get(sort_choice, "date")

    # 3. Sort rows
    if sort_by == "date":
        rows.sort(key=lambda x: x.get("Date", "unknown") if x.get("Date", "unknown") != "unknown" else "", reverse=True)
    elif sort_by == "kelly":
        rows.sort(
            key=lambda x: (
                0 if x.get("Kelly") is None else 1,
                x.get("Kelly") if x.get("Kelly") is not None else -float("inf")
            ),
            reverse=True
        )

    # 4. Format and Print Table manually (with Question hyperlinked to Web Link)
    config = TABLE_CONFIGS.get(sub_mode, TABLE_CONFIGS["all"])
    headers = config["headers"]
    widths = config["widths"]
    aligns = config["aligns"]
    q_idx = config["q_idx"]

    def format_cell(text, width, align):
        if align == "right":
            return str(text).rjust(width)[:width]
        return str(text).ljust(width)[:width]

    header_parts = []
    for h, w, a in zip(headers, widths, aligns):
        header_parts.append(format_cell(h, w, a))
    header_str = " | ".join(header_parts)
    separator_str = "-+-".join(["-" * w for w in widths])
    
    print("\n" + "=" * len(header_str))
    print(f" [HISTORY TABLE: {sub_mode.upper()} - sorted by {sort_by.upper()}] ".center(len(header_str), "="))
    print("=" * len(header_str))
    print(header_str)
    print(separator_str)

    for r in rows:
        kelly_val = r.get("Kelly")
        apy_val = r.get("APY")
        edge_val = r.get("Edge")
        pu_val = r.get("pu")
        pm_val = r.get("pm")
        
        kelly_str = f"{kelly_val*100:.2f}%" if kelly_val is not None else "-"
        apy_str = f"{apy_val*100:.2f}%" if apy_val is not None else "-"
        edge_str = f"{edge_val*100:.2f}%" if edge_val is not None else "-"
        pu_str = f"{pu_val*100:.1f}%" if pu_val is not None else "-"
        pm_str = f"{pm_val*100:.1f}%" if pm_val is not None else "-"
        
        outcome_val = r.get("Outcome")
        if outcome_val == "1.0": outcome_str = "YES"
        elif outcome_val == "0.0": outcome_str = "NO"
        elif outcome_val == "0.5": outcome_str = "HALF"
        else: outcome_str = "-"
        
        # Populate row values based on sub_mode
        if sub_mode == "predicted":
            vals = [r["Date"], r["Question"], kelly_str, pu_str, pm_str]
        elif sub_mode == "skipped":
            vals = [r["Date"], r["Question"]]
        elif sub_mode == "predicted_skipped":
            if r["Mode"] == "predicted":
                vals = [r["Mode"].upper(), r["Date"], r["Question"], kelly_str, pu_str, pm_str]
            else:  # skipped rows have no kelly/pu/pm
                vals = [r["Mode"].upper(), r["Date"], r["Question"], "-", "-", "-"]
        elif sub_mode == "resolved":
            vals = [r["Date"], r["Question"], kelly_str, pu_str, pm_str, outcome_str]
        else: # "all"
            vals = [r["Mode"].upper(), r["Date"], r["Question"], kelly_str, pu_str, pm_str, outcome_str]
            
        row_parts = []
        for i, (v, w, a) in enumerate(zip(vals, widths, aligns)):
            if i == q_idx:
                if len(v) > w:
                    q_visible = v[:w-3] + "..."
                else:
                    q_visible = v
                q_padded = q_visible.ljust(w)
                # Hyperlink visible text to polymarket event page
                row_parts.append(f"\033]8;;{r['Web Link']}\033\\{q_padded}\033]8;;\033\\")
            else:
                row_parts.append(format_cell(v, w, a))
                
        print(" | ".join(row_parts))
    print("=" * len(header_str) + "\n")
    if sub_mode in ["predicted", "predicted_skipped", "resolved", "all"]:
        print_prediction_stats(history, sub_mode=sub_mode)

def calculate_prediction_stats(history, sub_mode="predicted", base_ego=None):
    if base_ego is None:
        resolved = history.get("resolved", {})
        bs_u_total, bs_m_total, total_w = 0.0, 0.0, 0.0
        for market_preds in resolved.values():
            for data in ensure_list(market_preds):
                w = data.get("kelly", 0.0)
                if w <= 0:
                    continue
                outcome = data.get("outcome", 0)
                bs_u_total += w * ((data.get("pu", 0.5) - outcome) ** 2)
                bs_m_total += w * ((data.get("pm", 0.5) - outcome) ** 2)
                total_w += w
        if total_w > 0 and (bs_u_total + bs_m_total) > 0:
            base_ego = bs_m_total / (bs_u_total + bs_m_total)
        else:
            base_ego = 0.50

    preds_list = []
    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        for m_preds in history.get("predicted", {}).values():
            preds_list.extend(ensure_list(m_preds))
    if sub_mode in ["resolved", "all"]:
        for m_res in history.get("resolved", {}).values():
            preds_list.extend(ensure_list(m_res))

    if not preds_list:
        return None

    eps = 0.01
    items = []
    for p in preds_list:
        pu = float(p.get("pu", 0.5))
        pm = float(p.get("pm", 0.5))
        kelly = max(0.0, float(p.get("kelly", 0.0)))
        ego = float(p.get("dynamic_ego", base_ego))

        pu_clip = float(np.clip(pu, eps, 1.0 - eps))
        pm_clip = float(np.clip(pm, eps, 1.0 - eps))
        logit_pu = math.log(pu_clip / (1.0 - pu_clip))
        logit_pm = math.log(pm_clip / (1.0 - pm_clip))
        logit_norm = (ego * logit_pu) + ((1.0 - ego) * logit_pm)
        pu_norm = 1.0 / (1.0 + math.exp(-logit_norm))

        if pu >= pm:
            win_prob_bal = pu_norm
            win_prob_u = pu
            win_prob_m = pm
        else:
            win_prob_bal = 1.0 - pu_norm
            win_prob_u = 1.0 - pu
            win_prob_m = 1.0 - pm

        var_i = win_prob_bal * (1.0 - win_prob_bal)
        alloc_i = kelly * BANKROLL
        items.append({
            "kelly": kelly,
            "alloc": alloc_i,
            "win_prob_bal": win_prob_bal,
            "win_prob_u": win_prob_u,
            "win_prob_m": win_prob_m,
            "var_i": var_i
        })

    total_kelly = sum(x["kelly"] for x in items)
    total_alloc = sum(x["alloc"] for x in items)
    active_items = [x for x in items if x["kelly"] > 0]

    if total_kelly > 0 and active_items:
        weights = [x["kelly"] / total_kelly for x in active_items]
        used_items = active_items
    else:
        n_all = len(items)
        weights = [1.0 / n_all for _ in items]
        used_items = items

    exp_win_bal = sum(w * x["win_prob_bal"] for w, x in zip(weights, used_items))
    exp_win_u = sum(w * x["win_prob_u"] for w, x in zip(weights, used_items))
    exp_win_m = sum(w * x["win_prob_m"] for w, x in zip(weights, used_items))

    norm_variance = sum((w ** 2) * x["var_i"] for w, x in zip(weights, used_items))
    norm_std_dev = math.sqrt(norm_variance)
    sum_w_sq = sum(w ** 2 for w in weights)
    effective_n = (1.0 / sum_w_sq) if sum_w_sq > 0 else 0.0

    return {
        "total_count": len(items),
        "active_count": len(active_items),
        "total_kelly": total_kelly,
        "total_alloc": total_alloc,
        "bankroll": BANKROLL,
        "exp_win_bal": exp_win_bal,
        "exp_win_u": exp_win_u,
        "exp_win_m": exp_win_m,
        "norm_variance": norm_variance,
        "norm_std_dev": norm_std_dev,
        "effective_n": effective_n
    }

def print_prediction_stats(history, sub_mode="predicted", base_ego=None):
    stats = calculate_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego)
    if not stats:
        return
    print("--- PREDICTION BALANCING & VARIANCE STATS ---")
    print(f"Predictions      : {stats['active_count']} active ({stats['total_count']} total) | Effective N: {stats['effective_n']:.2f}")
    w_sum = history.get("wallet_summary")
    if isinstance(w_sum, dict) and "cash" in w_sum and "portfolio" in w_sum:
        print(f"Wallet Bankroll  : ${stats['bankroll']:,.2f} (Cash: ${w_sum['cash']:,.2f} | Positions: ${w_sum['portfolio']:,.2f})")
    print(f"Allocated Capital: ${stats['total_alloc']:,.2f} / ${stats['bankroll']:,.2f} ({stats['total_kelly']*100:.1f}% Kelly)")
    print(f"Expected Win Rate: {stats['exp_win_bal']*100:.2f}% Balanced (User: {stats['exp_win_u']*100:.2f}% | Market: {stats['exp_win_m']*100:.2f}%)")
    print(f"Prob. Variance   : {stats['norm_variance']:.4f} (Std Dev: +/-{stats['norm_std_dev']*100:.2f}%)\n")

def calculate_base_ego(history):
    resolved = history.get("resolved", {})
    if not resolved: return 0.50 
        
    bs_u_total, bs_m_total = 0.0, 0.0
    total_weight = 0.0
    
    for market_preds in resolved.values():
        market_preds = ensure_list(market_preds)
        for data in market_preds:
            outcome = data.get("outcome", 0)
            
            # Fetch the Kelly fraction used
            weight = data.get("kelly", 0.0)
            
            if weight <= 0: 
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
        
    print(f"[*] Current Brier Score - User: {bs_u_total:.4f} | Market: {bs_m_total:.4f}")
    
    if bs_u_total == 0 and bs_m_total == 0: return 0.50 
    return bs_m_total / (bs_u_total + bs_m_total)

def calculate_annualized_yield(edge, true_price, days_until):
    # Prevent division by zero for expired markets or 0 prices
    if days_until <= 0.0 or true_price <= 0:
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
        return numbers[0] / 100.0, numbers[0] / 100.0
    else:
        return min(numbers[0], numbers[1]) / 100.0, max(numbers[0], numbers[1]) / 100.0

def calculate_allocation(lower_bound, upper_bound, pm_bid, pm_ask, fee_rate, bankroll, base_ego, custom_min_edge=None):
    if custom_min_edge is None:
        custom_min_edge = MIN_EDGE
    eps = 0.01
    
    # 1. Midpoints and Uncertainty Spreads
    pu_mid = (lower_bound + upper_bound) / 2.0
    u_spread = upper_bound - lower_bound
    m_spread = pm_ask - pm_bid
    pm_mid_clip = np.clip((pm_bid + pm_ask) / 2.0, eps, 1-eps)
    
    # 2. Bidirectional Conviction Weighting
    # We reduce the weight of whoever is more uncertain (wider spread)
    u_conviction = base_ego * (1.0 - u_spread)
    m_conviction = (1.0 - base_ego) * (1.0 - m_spread)
    
    # 3. Calculate Dynamic Ego (Normalized)
    if (u_conviction + m_conviction) > 0:
        dynamic_ego = u_conviction / (u_conviction + m_conviction)
    else:
        # Fallback to historical baseline if both user and market are at maximum uncertainty
        dynamic_ego = base_ego 
        
    pu_clip = np.clip(pu_mid, eps, 1-eps)
    
    # 4. Bayesian Logit Pooling
    logit_pu, logit_pm = np.log(pu_clip / (1-pu_clip)), np.log(pm_mid_clip / (1-pm_mid_clip))
    logit_norm = (dynamic_ego * logit_pu) + ((1 - dynamic_ego) * logit_pm)
    pu_norm = 1 / (1 + np.exp(-logit_norm)) 
    
    # 5. Pricing and Action Mapping
    cost_yes, cost_no = pm_ask, 1.0 - pm_bid 
    
    if pu_norm > cost_yes:
        action, base_price, win_prob = "YES", cost_yes, pu_norm
    elif (1.0 - pu_norm) > cost_no:
        action, base_price, win_prob = "NO", cost_no, 1.0 - pu_norm
    else:
        return "NONE", 0.0, 0.0, 0.0, dynamic_ego, 0.0
    
    # 6. Edge and Fee Evaluation
    true_price = base_price + (fee_rate * base_price * (1 - base_price))
    edge = win_prob - true_price

    # 7. Uncapped Fractional Kelly Allocation
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
        if market_id in history.get("predicted", {}) or market_id in history.get("skipped", {}):
            return False, "Already predicted or skipped"
    elif mode == "target":
        if exclude_mode in ["predicted", "both"] and market_id in history.get("predicted", {}):
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
    active_wallet = POLYGON_WALLET or history.get("wallet_address")
    if active_wallet:
        history = import_polymarket_positions(active_wallet, history, auto_sync=True)
            
    history = update_resolutions(history)
    save_history(stringify_overflow(history))
    
    if mode == "review":
        print_review_table(history, sub_mode)
        
        active_market_count = 0
        if sub_mode in ["predicted", "predicted_skipped", "all"]:
            active_market_count += len(history.get("predicted", {}))
        if sub_mode in ["skipped", "predicted_skipped", "all"]:
            active_market_count += len(history.get("skipped", {}))
        has_active_markets = active_market_count > 0
            
        if not has_active_markets:
            print("[*] No active/open markets to predict in this sub-mode. Returning to main menu.")
            return None
            
        review_choice = input(f"Do you want to run a prediction session on the {active_market_count} active markets in this selection? (y/n): ").strip().lower()
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
            
            # 2. Append the new independent prediction tranche
            history["predicted"][market_id].append({
                "question": question,
                "slug": event_slug,
                "date": today_str,
                "pu": (lower + upper) / 2.0,
                "pm": pm_mid,
                "dynamic_ego": dynamic_ego,
                "kelly": raw_kelly, # Required for new Weighted Brier calculation
                "edge": adjusted_edge,
                "apy": adjusted_apy,
                "conditionId": m.get("conditionId")
            })
            
            history["skipped"].pop(market_id, None)
            
            print(f"\n--- MARKET ANALYSIS ---")
            print(f"Market Spread: Bid {pm_bid*100:.1f}% | Ask {pm_ask*100:.1f}% (Spread: {(pm_ask - pm_bid)*100:.1f}%)")
            print(f"User Bounds  : {lower*100:.1f}% to {upper*100:.1f}% (Spread: {(upper-lower)*100:.1f}%)")
            print(f"Dynamic Ego  : {dynamic_ego:.2f} (Base {base_ego:.2f})")
            
            if action in ["YES", "NO"]:
                cumulative_exposure += final_alloc / BANKROLL
                session_trades[market_id] = final_alloc
                
                print(f"ACTION       : BUY {action} @ {true_price*100:.1f}%")
                if slippage > 0.0001:
                    print(f"VWAP PRICE   : {vwap_true*100:.2f}% (Slippage: +{slippage*100:.2f}%)")
                    print(f"ADJUSTED EDGE: {adjusted_edge*100:.2f}% (Target: {edge*100:.2f}%)")
                    print(f"ADJUSTED APY : {adjusted_apy*100:.2f}% (Target: {apy*100:.2f}%)")
                else:
                    print(f"USER EDGE    : {adjusted_edge*100:.2f}% (After Fees & Spread)")
                    print(f"APY          : {apy*100:.2f}% (Compounded)")
                    
                print(f"KELLY %      : {raw_kelly*100:.2f}% of bankroll")
                print(f"ALLOCATION   : ${final_alloc:,.2f}")
                
                portfolio_data.append({
                    "Question": question[:50] + "..",
                    "Action": action,
                    "Ego": f"{dynamic_ego:.2f}",
                    "Price": f"{vwap_true*100:.1f}%",
                    "Kelly": f"{raw_kelly*100:.2f}%",
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
    skip_count = len(history.get("skipped", {}))
    res_count = len(history.get("resolved", {}))
    
    print("\nSelect Category to Delete From:")
    print(f" 1: 'Predicted' Markets ({pred_count})")
    print(f" 2: 'Skipped' Markets ({skip_count})")
    print(f" 3: 'Resolved' Markets ({res_count})")
    print(f" 4: All Categories ({pred_count + skip_count + res_count})")
    print(" 5: Cancel")
    
    cat_sel = input("> ").strip()
    if cat_sel == "1":
        categories = ["predicted"]
    elif cat_sel == "2":
        categories = ["skipped"]
    elif cat_sel == "3":
        categories = ["resolved"]
    elif cat_sel == "4":
        categories = ["predicted", "skipped", "resolved"]
    else:
        print("[!] Deletion cancelled.")
        return

    entries = []
    for cat in categories:
        cat_data = history.get(cat, {})
        for market_id, val in cat_data.items():
            if cat in ["predicted", "resolved"]:
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
        # Check if direct match for Market ID
        by_id = [e for e in entries if e["market_id"].lower() == user_input.lower()]
        if by_id:
            to_delete = by_id
        else:
            # Try parsing as numbers / ranges
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
                # Try substring search across question / slug
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

    # Group deletions by (category, market_id)
    by_market = {}
    for e in to_delete:
        key = (e["category"], e["market_id"])
        if key not in by_market:
            by_market[key] = []
        by_market[key].append(e["pred_idx"])

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
    skip_count = len(history.get("skipped", {}))
    res_count = len(history.get("resolved", {}))
    
    print("\nSelect Category to Clear:")
    print(f" 1: 'Predicted' Markets ({pred_count})")
    print(f" 2: 'Skipped' Markets ({skip_count})")
    print(f" 3: 'Resolved' Markets ({res_count})")
    print(" 4: Cancel")
    
    cat_sel = input("> ").strip()
    cat_map = {"1": "predicted", "2": "skipped", "3": "resolved"}
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
    skip_count = len(history.get("skipped", {}))
    res_count = len(history.get("resolved", {}))
    total_markets = pred_count + skip_count + res_count

    print("\n--- DELETE DATA ---")
    print(" 1: Delete all data (Full reset)")
    print(f" 2: Selectively delete specific markets ({total_markets} in history)")
    print(" 3: Clear specific category (Predicted, Skipped, or Resolved)")
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
        print(" 1: Discover Markets")
        print(" 2: Review Markets")
        print(" 3: Target Specific URLs/Slugs")
        print(" 4: Delete Data (All or Selective)")
        print(" 5: Import Polymarket Wallet Positions")
        print(" 6: Custom Market Calculator")
        print(" 7: Manually Resolve Open Markets")
        print(" 8: Settings / Filters")
        print(" 9: Exit")
        
        choice = input("> ").strip()
        
        targets = None
        category = "All"
        session_max_offset = 0
        target_exclude_mode = "none" # Default initialization
        discovery_order = "volume"
        
        if choice == "9":
            print("Exiting program.")
            break
            
        elif choice == "8":
            print("\n--- SETTINGS / FILTERS ---")
            print(f" 1: Min Edge       = {MIN_EDGE*100:.2f}%  (skip bets below this edge)")
            print(f" 2: Max Days       = {MAX_DAYS}  (ignore markets beyond this horizon)")
            print(f" 3: Min Days       = {MIN_DAYS}  (ignore markets resolving sooner than this)")
            print(f" 4: Extreme Odds   = {EXTREME_ODDS*100:.2f}%  (tail-risk filter — skip near 0% or 100%)")
            print(f" 5: Bankroll       = ${BANKROLL:,.2f}  (manual override; overwritten by wallet import)")
            print(f"\nEnter a setting number to edit, or hit enter to cancel.")
            s_choice = input("> ").strip()
            setting_map = {"1": "MIN_EDGE", "2": "MAX_DAYS", "3": "MIN_DAYS", "4": "EXTREME_ODDS", "5": "BANKROLL"}
            if s_choice in setting_map:
                param = setting_map[s_choice]
                current_val = {"MIN_EDGE": MIN_EDGE, "MAX_DAYS": MAX_DAYS, "MIN_DAYS": MIN_DAYS,
                               "EXTREME_ODDS": EXTREME_ODDS, "BANKROLL": BANKROLL}[param]
                hint = {
                    "MIN_EDGE": "e.g. 0.02 for 2% — set 0 to take any positive edge",
                    "MAX_DAYS": "e.g. 90 — markets resolving beyond this are ignored",
                    "MIN_DAYS": "e.g. 1 — skip markets resolving within 1 day (0 = no floor)",
                    "EXTREME_ODDS": "e.g. 0.02 for 2% — skip markets priced below 2% or above 98%",
                    "BANKROLL": "e.g. 100.00 — your total betting capital in USD"
                }[param]
                new_raw = input(f"New value for {param} (currently {current_val}) [{hint}]: ").strip()
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
                        print("[!] Invalid value — setting unchanged.")
                else:
                    print("[!] No change made.")
            elif s_choice != "":
                print("[!] Invalid selection.")
            continue
            
        elif choice == "7":
            history = load_history()
            open_preds_dict = history.get("predicted", {})
            if not open_preds_dict:
                print("[!] No open predictions to resolve.")
                continue
                
            print("\n--- MANUALLY RESOLVE MARKETS ---")
            for m_id, preds in open_preds_dict.items():
                preds_list = ensure_list(preds)
                q = preds_list[0].get("question", "Unknown")
                print(f" ID: {m_id} | {q}")
                
            sel = input("\nEnter the Market ID or Custom ID to resolve (or hit enter to cancel): ").strip()
            if not sel: continue
            
            if sel not in open_preds_dict:
                print(f"[!] ID '{sel}' not found in open predictions.")
                continue
                
            target_id = sel
            preds = open_preds_dict[target_id]
            outcome = None
            while True:
                print(f"\nResolving: {ensure_list(preds)[0].get('question')} \nSelect outcome or hit enter to cancel")
                print(" 1: YES")
                print(" 2: NO")
                print(" 3: HALF")
                out_sel = input("> ").strip()
                if out_sel == "": break
                if out_sel == "1": outcome = 1.0
                elif out_sel == "2": outcome = 0.0
                elif out_sel == "3": outcome = 0.5
                else:
                    print("[!] Invalid selection. Please select from 1, 2, or 3")
                    continue
                break
                
            if outcome is None:
                print("[!] Resolve cancelled.")
                continue
                
            if "resolved" not in history:
                history["resolved"] = {}
            if target_id not in history["resolved"]:
                history["resolved"][target_id] = []
                
            for p in ensure_list(preds):
                p["outcome"] = outcome
                history["resolved"][target_id].append(p)
                
            del history["predicted"][target_id]
            save_history(stringify_overflow(history))
            print("[+] Market manually resolved and saved.")
            continue
            
        elif choice == "6":
            print("\n--- CUSTOM MARKET CALCULATOR ---")
            history = load_history()
            
            ego_mode = input("Custom Base Ego (e.g., 0.5) or leave blank for Historical Ego: ").strip().upper()
            if not ego_mode:
                base_ego = calculate_base_ego(history)
            else:
                try:
                    base_ego = float(ego_mode)
                    if not (0.0 <= base_ego <= 1.0):
                        print("[!] Ego must be between 0.0 and 1.0. Defaulting to Historical.")
                        base_ego = calculate_base_ego(history)
                except ValueError:
                    print("[!] Invalid ego value. Defaulting to Historical.")
                    base_ego = calculate_base_ego(history)
            
            try:
                m_input = input("Market Probability Bounds % (e.g., 42-52 or 47): ").strip()
                m_lower, m_upper = parse_user_input(m_input)
                
                u_input = input("User Probability Bounds % (e.g., 16-42 or 29): ").strip()
                lower, upper = parse_user_input(u_input)
                
                fee_input = input("Fee Rate % (default 3.0 Polymarket Sports): ").strip()
                if fee_input:
                    fee_rate = float(fee_input) / 100.0
                else:
                    fee_rate = 0.03
                    
                days_input = input("Days until resolution (default 0 for no APY): ").strip()
                days_until = float(days_input) if days_input else 0.0
                
                bankroll_input = input(f"Bankroll (default ${BANKROLL:,.2f}): ").strip()
                if bankroll_input:
                    calc_bankroll = float(bankroll_input)
                else:
                    calc_bankroll = BANKROLL
                    
                min_edge_input = input(f"Minimum Edge % (default {MIN_EDGE*100:.1f}): ").strip()
                if min_edge_input:
                    calc_min_edge = float(min_edge_input) / 100.0
                else:
                    calc_min_edge = MIN_EDGE
                    
                action, true_price, raw_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
                    lower, upper, m_lower, m_upper, fee_rate, calc_bankroll, base_ego, custom_min_edge=calc_min_edge
                )
                
                _, apy = calculate_annualized_yield(edge, true_price, days_until)
                
                
                print(f"\n--- CALCULATION RESULTS ---")
                if action in ["YES", "NO"]:
                    print(f"ACTION       : BUY {action} @ {true_price*100:.1f}%")
                    print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
                    print(f"DYNAMIC EGO  : {dynamic_ego:.2f} (Base {base_ego:.2f})")
                    if days_until > 0: print(f"APY          : {apy*100:.2f}%")
                    print(f"KELLY %      : {raw_kelly*100:.2f}% of bankroll")
                    print(f"ALLOCATION   : ${final_alloc:,.2f}")
                else:
                    if action == "THIN_EDGE": reason = f"Edge below {calc_min_edge*100:.1f}% threshold"
                    elif action == "NONE": reason = "Trapped inside bid-ask spread"
                    else: reason = "Unknown"
                    print(f"ACTION       : $0 Allocation ({reason})")
                    print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
                    print(f"DYNAMIC EGO  : {dynamic_ego:.2f} (Base {base_ego:.2f})")
                    
                save_q = input("\nSave this custom market prediction to history? (y/n): ").strip().lower()
                if save_q == 'y':
                    q_title = input("Enter a title for this market: ").strip()
                    if not q_title: q_title = "Custom Market"
                    
                    custom_id = f"custom_{int(datetime.now().timestamp())}"
                    if "predicted" not in history:
                        history["predicted"] = {}
                    if custom_id not in history["predicted"]:
                        history["predicted"][custom_id] = []
                        
                    history["predicted"][custom_id].append({
                        "question": f"[CUSTOM] {q_title}",
                        "slug": "custom",
                        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "pu": (lower + upper) / 2.0,
                        "pm": (m_lower + m_upper) / 2.0,
                        "dynamic_ego": dynamic_ego,
                        "kelly": raw_kelly,
                        "edge": edge,
                        "apy": apy,
                        "conditionId": "custom"
                    })
                    save_history(stringify_overflow(history))
                    print("[+] Saved to predicted history.")
                
            except ValueError as e:
                print(f"\n[!] Error: {e}. Please try again.\n")
            continue
            
        elif choice == "5":
            history = import_polymarket_positions(POLYGON_WALLET, load_history())
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
                print("[!] No valid inputs parsed. Returning to main menu.\n")
                continue
            else:
                print(f"Parsed Targets ({len(targets)}): {', '.join(targets)}")
                
            # --- TARGET MODE EXCLUSION MENU ---
            print("\nTarget Mode - Select Exclusion Filter:")
            print(" 1: None (Target all markets)")
            print(" 2: Exclude Already Predicted")
            print(" 3: Exclude Already Skipped")
            print(" 4: Exclude Both (Predicted + Skipped)")
            
            ex_choice = input("> ").strip()
            target_exclude_mode = {"2": "predicted", "3": "skipped", "4": "both"}.get(ex_choice, "none")
                
            op_mode, sub_mode = "target", "all"
            
        elif choice == "1":
            category_input = input("\nEnter Category Tag (e.g., 'Politics', 'Crypto') or hit enter for 'All': ").strip()
            category = category_input if category_input else "All"
            
            # Select Discovery Vector (Default = Volume/Normal)
            print("\nSelect Discovery Vector:")
            print(" 1: Volume (Default - Top volume markets)")
            print(" 2: 24h Volume (Recent trending/breakout markets)")
            print(" 3: Liquidity (Deepest orderbooks)")
            print(" 4: Ending Soonest (Short time horizon)")
            print(" 5: Competitive (Contested 50/50 odds)")
            print(" 6: Low Volume / Niche (Untapped / long-tail markets)")
            print(" 7: Newest Listed (Freshly created markets)")
            print(" 8: Multi-Vector (Full spectrum rotation across all vectors)")
            vec_choice = input("> ").strip()
            vec_map = {
                "1": "volume",
                "2": "volume24hr",
                "3": "liquidity",
                "4": "endDate",
                "5": "competitive",
                "6": "volume_asc",
                "7": "startDate",
                "8": "multi"
            }
            discovery_order = vec_map.get(vec_choice, "volume")
            
            # Fetch Brink to establish dynamic boundary checks
            platform_max_offset = find_platform_brink()
            max_pages = (platform_max_offset // 100) + 1
            
            while True:
                market_mode = input(f"\nDiscover Mode - Select Page Limit:\n 1: Max Reach ({max_pages} pages / ~{max_pages*100} events for selected vector)\n 2: Custom Top Pages (1 to {max_pages})\n> ").strip()
                if market_mode in ["1", "2"]:
                    op_mode = "discover"
                    sub_mode = "all" if market_mode == "1" else "custom"
                    
                    if sub_mode == "custom":
                        try:
                            custom_limit = int(input(f"Enter number of top pages to search (1 to {max_pages}): ").strip())
                            if custom_limit < 1 or custom_limit > max_pages:
                                print(f"[!] Out of bounds. Please enter a number between 1 and {max_pages}.")
                                continue
                            session_max_offset = (custom_limit - 1) * 100
                        except ValueError:
                            print("[!] Invalid input. Defaulting to 1 page.")
                            session_max_offset = 0
                    else:
                        session_max_offset = platform_max_offset
                        
                    break
                print("[!] Invalid selection. Please choose 1 or 2.")
                
        elif choice == "2":
            while True:
                review_mode = input("\nReview Mode - Select Sub-Mode:\n 1: 'Predicted' Markets\n 2: 'Skipped' Markets\n 3: 'Resolved' Markets\n 4: 'Predicted' + 'Skipped' Markets\n 5: All (Predicted + Skipped + Resolved)\n> ").strip()
                if review_mode in ["1", "2", "3", "4", "5"]:
                    op_mode = "review"
                    sub_mode = {"1": "predicted", "2": "skipped", "3": "resolved", "4": "predicted_skipped", "5": "all"}[review_mode]
                    break
                print("[!] Invalid selection. Please choose 1, 2, 3, 4, or 5.")
                
        else:
            print("[!] Invalid selection. Please choose 1-9.")
            continue
            
        portfolio = run_prediction_session(mode=op_mode, sub_mode=sub_mode, target_slugs=targets, category=category, max_offset=session_max_offset, exclude_mode=target_exclude_mode, discovery_order=discovery_order)
        
        if portfolio is not None and not portfolio.empty:
            print("\n--- Final Session Allocations ---")
            pd.set_option('display.max_colwidth', None) 
            print(portfolio.to_string(index=False))
        else:
            print("\nNo allocations made.")