import math
import requests
import json
import os
import random
import re
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
HISTORY_FILE = "prediction_history.json"
BANKROLL = 42.00
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
        "https://polygon-rpc.com",
        "https://polygon.llamarpc.com",
        "https://rpc.ankr.com/polygon"
    ]
    usdc_contract = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"
    addr_clean = wallet_address.lower().replace('0x', '').zfill(64)
    data = '0x70a08231' + addr_clean
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": usdc_contract, "data": data}, "latest"],
        "id": 1
    }
    for rpc in rpcs:
        try:
            resp = api.post(rpc, json=payload, timeout=3).json()
            result_hex = resp.get("result", "0x0")
            return int(result_hex, 16) / 10**6
        except Exception:
            continue
    return None

def calculate_vwap(asks, target_usdc):
    if not asks or target_usdc <= 0:
        return 0.0, 0.0
    total_shares = 0.0
    remaining_usdc = target_usdc
    total_spent = 0.0
    for ask in asks:
        try:
            price = float(ask['price'])
            size = float(ask['size'])
        except Exception:
            continue
        if price <= 0:
            continue
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

def import_polymarket_positions(wallet_address, history):
    global BANKROLL
    if not wallet_address:
        wallet_address = input("Enter Polymarket wallet address (0x...): ").strip()
    if not wallet_address.startswith("0x"):
        print("[!] Invalid address format.")
        return history
        
    base_ego = calculate_base_ego(history)
        
    print(f"[*] Fetching positions from Polymarket for {wallet_address}...")
    try:
        url = f"https://data-api.polymarket.com/positions?user={wallet_address}"
        positions = api.get(url, timeout=10).json()
    except Exception as e:
        print(f"[!] Error fetching positions: {e}")
        return history
        
    if not positions:
        print("[*] No positions found on Polymarket for this user.")
        return history
        
    print("[*] Calculating Live Portfolio Value...")
    cash_balance = get_wallet_usdc_balance(wallet_address) or 0.0
    portfolio_value = 0.0
    
    for pos in positions:
        size = float(pos.get("size", 0.0))
        avg_price = float(pos.get("avgPrice", 0.0))
        # Polymarket data API often provides currentValue, otherwise fallback to cost basis (initialValue)
        value = float(pos.get("currentValue", pos.get("initialValue", size * avg_price)))
        portfolio_value += value
        
    BANKROLL = cash_balance + portfolio_value
    print(f"[+] Live Bankroll Updated: ${BANKROLL:,.2f} (Cash: ${cash_balance:,.2f} | Portfolio: ${portfolio_value:,.2f})")
    
    print(f"[+] Found {len(positions)} positions. Importing new ones...")
    imported_predicted = 0
    imported_resolved = 0
    
    for pos in positions:
        cond_id = pos.get("conditionId")
        if not cond_id:
            continue
            
        # Check if already in history (under predicted or resolved keys)
        already_stored = False
        for m_id, preds in history.get("predicted", {}).items():
            preds_list = ensure_list(preds)
            if any(p.get("conditionId") == cond_id for p in preds_list):
                already_stored = True
                break
        if not already_stored:
            for m_id, res in history.get("resolved", {}).items():
                res_list = ensure_list(res)
                if any(r.get("conditionId") == cond_id for r in res_list):
                    already_stored = True
                    break
                    
        if already_stored:
            continue
            
        # Query Gamma API to resolve metadata
        try:
            resp = api.get(f"{GAMMA_API}/markets?conditionId={cond_id}", timeout=5).json()
            if not resp or not isinstance(resp, list):
                continue
            m = resp[0]
        except Exception:
            continue
            
        market_id = m.get("id")
        question = m.get("question")
        slug = m.get("slug")
        
        # Calculate attributes
        avg_price = float(pos.get("avgPrice", 0.5))
        size = float(pos.get("size", 0.0))
        initial_val = float(pos.get("initialValue", size * avg_price))
        
        # Avoid zero division
        kelly_val = initial_val / BANKROLL if BANKROLL > 0 else 0.0
        
        outcome_str = pos.get("outcome", "Yes")
        
        print(f"\n[IMPORT] Question: {question}")
        print(f"         Outcome: {outcome_str}  |  Entry Price (pm): {avg_price*100:.1f}%")
        
        pu_input = input("         Enter your subjective probability bounds % (e.g., 60-70 or 65) or press enter to skip: ").strip()
        try:
            if pu_input:
                lower, upper = parse_user_input(pu_input)
            else:
                lower, upper = avg_price, avg_price
        except:
            lower, upper = avg_price, avg_price
            
        pu_val = (lower + upper) / 2.0
        edge = pu_val - avg_price
        
        # Reverse-calculate dynamic_ego using the user bounds
        u_spread = upper - lower
        m_spread = 0.0 # Unknown historic market spread, assuming fully tight
        
        u_conviction = base_ego * (1.0 - u_spread)
        m_conviction = (1.0 - base_ego) * (1.0 - m_spread)
        if (u_conviction + m_conviction) > 0:
            dynamic_ego = u_conviction / (u_conviction + m_conviction)
        else:
            dynamic_ego = base_ego
        
        if outcome_str == "No":
            # For 'No', the true subjective probability is 1 - pu. Wait, the user is buying 'No' shares at avgPrice.
            # Usually, pu for 'No' would be interpreted as the probability of 'No' occurring.
            # But in the database, `pu` is strictly the probability of `Yes`.
            # If the user enters the probability of their outcome (No), we should normalize it.
            # Let's just store the pu as entered if we assume the user understands pu is the probability of Yes.
            pass
        
        # Calculate APY
        try:
            res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            days_until = seconds_left / 86400.0
        except:
            days_until = 0.0
            
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
        
        # Put in resolved or predicted based on redeemable status
        if pos.get("redeemable"):
            outcome_val = 1.0 if outcome_str == "Yes" else 0.0 if outcome_str == "No" else 0.5
            pred_entry["outcome"] = outcome_val
            if market_id not in history["resolved"]:
                history["resolved"][market_id] = []
            history["resolved"][market_id].append(pred_entry)
            imported_resolved += 1
        else:
            if market_id not in history["predicted"]:
                history["predicted"][market_id] = []
            history["predicted"][market_id].append(pred_entry)
            imported_predicted += 1
            
    print(f"\n[+] Import complete: {imported_predicted} open predictions and {imported_resolved} resolved predictions imported.")
    return history

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f: 
                data = json.load(f)
                if isinstance(data, list) or "seen_market_ids" in data:
                    return {"predicted": {}, "skipped": {}, "resolved": {}}
                if "resolved" not in data:
                    data["resolved"] = {}
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
            "headers": ["Date", "Question", "Kelly", "APY", "Edge", "pu", "pm"],
            "widths": [16, 60, 8, 8, 8, 6, 6],
            "aligns": ["left", "left", "right", "right", "right", "right", "right"],
            "q_idx": 1
        },
        "skipped": {
            "headers": ["Date", "Question"],
            "widths": [16, 80],
            "aligns": ["left", "left"],
            "q_idx": 1
        },
        "resolved": {
            "headers": ["Date", "Question", "Kelly", "pu", "pm", "Outcome"],
            "widths": [16, 60, 8, 6, 6, 8],
            "aligns": ["left", "left", "right", "right", "right", "left"],
            "q_idx": 1
        },
        "all": {
            "headers": ["Mode", "Date", "Question", "Kelly", "Edge", "Outcome"],
            "widths": [10, 16, 60, 8, 8, 8],
            "aligns": ["left", "left", "left", "right", "right", "left"],
            "q_idx": 2
        }
    }

    # 1. Gather all items according to sub_mode
    rows = []
    
    # Gather predicted
    if sub_mode in ["predicted", "all"]:
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
    if sub_mode in ["skipped", "all"]:
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
    if sub_mode == "predicted":
        print("\nSort 'predicted' by:")
        print(" 1: Date (default)")
        print(" 2: Kelly")
        print(" 3: APY")
        print(" 4: Edge")
        sort_choice = input("> ").strip()
        sort_by = {"2": "kelly", "3": "apy", "4": "edge"}.get(sort_choice, "date")
    elif sub_mode == "resolved":
        print("\nSort 'resolved' by:")
        print(" 1: Date (default)")
        print(" 2: Kelly")
        sort_choice = input("> ").strip()
        sort_by = {"2": "kelly"}.get(sort_choice, "date")

    # 3. Sort rows
    if sort_by == "date":
        rows.sort(key=lambda x: x.get("Date", "unknown") if x.get("Date", "unknown") != "unknown" else "", reverse=True)
    elif sort_by == "kelly":
        rows.sort(key=lambda x: x.get("Kelly", 0.0), reverse=True)
    elif sort_by == "apy":
        rows.sort(key=lambda x: x.get("APY", 0.0), reverse=True)
    elif sort_by == "edge":
        rows.sort(key=lambda x: x.get("Edge", 0.0), reverse=True)

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
            vals = [r["Date"], r["Question"], kelly_str, apy_str, edge_str, pu_str, pm_str]
        elif sub_mode == "skipped":
            vals = [r["Date"], r["Question"]]
        elif sub_mode == "resolved":
            vals = [r["Date"], r["Question"], kelly_str, pu_str, pm_str, outcome_str]
        else: # "all"
            vals = [r["Mode"].upper(), r["Date"], r["Question"], kelly_str, edge_str, outcome_str]
            
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
        apy = ((1 + roi) ** (365.0 / days_until)) - 1
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

def calculate_allocation(lower_bound, upper_bound, pm_bid, pm_ask, fee_rate, bankroll, base_ego):
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
    raw_kelly = max(0, edge / (1 - true_price))
    final_allocation = raw_kelly * bankroll
    
    if edge < MIN_EDGE:
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

        if seconds_left > 0:
            days_until = seconds_left / 86400.0
            if days_until < MIN_DAYS or days_until > MAX_DAYS:
                return False, f"Time horizon ({days_until:.1f} days) outside bounds"
    except:
        return False, "Invalid end date format"
            
    return True, "Valid"

def generate_market_stream(mode, sub_mode, target_slugs, history, category="All", max_offset=0, exclude_mode="none"):
    if mode == "discover":
        seen_categories = set()
        while True:
            offset = random.randint(0, max_offset // 100) * 100
            print(f"[*] Fetching market...",  end="\r")
            try:
                events = api.get(f"{GAMMA_API}/events?active=true&closed=false&limit=100&offset={offset}").json()
            except Exception:
                continue
            
            if not events: continue
            random.shuffle(events)
            
            for event in events:
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

                    # Print all seen Categories
                    # if seen_categories:
                    #     print(f"Current Matched Categories: {', '.join(sorted(seen_categories))}")

                        
                valid_markets = []
                for m in event.get('markets', []):
                    m['parent_slug'] = event.get('slug', 'unknown')
                    m['parent_name'] = event.get('title', 'unknown')
                    
                    is_valid, reason = validate_market(m, history, mode, exclude_mode)
                    if is_valid: 
                        valid_markets.append(m)
                    elif reason.startswith("Non-binary"):
                        print(f"[-] Ignored Categorical Market: '{m.get('question')}' | Reason: {reason}")
                        print(f"URL: https://polymarket.com/market/{m.get('id')}\n")
                
                if valid_markets: yield random.choice(valid_markets) 
                    
    elif mode == "target":
        if not target_slugs:
            print("[!] Error: No valid URL/slug provided.")
            return
            
        for slug in target_slugs:
            try:
                events = api.get(f"{GAMMA_API}/events?slug={slug}").json()
                if not events:
                    print(f"[-] Invalid or empty data returned for slug: {slug}")
                    continue
                    
                for event in events:
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
        if sub_mode in ["predicted", "all"]: target_ids.extend(list(history.get("predicted", {}).keys()))
        if sub_mode in ["skipped", "all"]: target_ids.extend(list(history.get("skipped", {}).keys()))
            
        if not target_ids:
            print("\n[!] History is empty. No markets available to review.")
            return
            
        random.shuffle(target_ids)
        for market_id in target_ids:
            try:
                m = api.get(f"{GAMMA_API}/markets/{market_id}").json()
                if m:
                    m['parent_slug'] = m.get('slug', 'unknown') 
                    m['parent_name'] = m.get('question', 'unknown')
                    
                    is_valid, reason = validate_market(m, history, mode, exclude_mode)
                    if is_valid: 
                        yield m
                    else:
                        if market_id in history.get("skipped", {}) and ("closed" in reason or "paused" in reason):
                            print(f"[+] Auto-Purged Dead Skipped Market: '{m.get('question')}'")
                            del history["skipped"][market_id]
                        elif market_id in history.get("predicted", {}) and ("paused" in reason):
                            print(f"[-] Review Paused (Awaiting UMA Resolution): '{m.get('question')}'")
                        else:
                            print(f"[-] Ignored Review Market: '{m.get('question')}' | Reason: {reason}")
            except Exception:
                continue

def run_prediction_session(mode="discover", sub_mode="all", target_slugs=None, category="All", max_offset=0, exclude_mode="none"):
    global BANKROLL
    
    # Live wallet bankroll update
    if POLYGON_WALLET:
        print(f"[*] Fetching live USDC balance on Polygon for wallet {POLYGON_WALLET}...")
        balance = get_wallet_usdc_balance(POLYGON_WALLET)
        if balance is not None:
            BANKROLL = balance
            print(f"[+] Wallet USDC Balance successfully loaded: ${BANKROLL:,.2f}")
        else:
            print(f"[!] Failed to fetch live wallet balance. Falling back to default: ${BANKROLL:,.2f}")
            
    history = update_resolutions(load_history())
    save_history(stringify_overflow(history))
    
    if mode == "review":
        print_review_table(history, sub_mode)
        
        has_active_markets = False
        if sub_mode in ["predicted", "all"] and history.get("predicted"):
            has_active_markets = True
        if sub_mode in ["skipped", "all"] and history.get("skipped"):
            has_active_markets = True
            
        if not has_active_markets:
            print("[*] No active/open markets to review in this sub-mode. Returning to main menu.")
            return None
            
        review_choice = input("Do you want to run a prediction review session on the active markets in this selection? (y/n): ").strip().lower()
        if review_choice != 'y':
            return None
            
    base_ego = calculate_base_ego(history)
    
    print(f"\n--- Starting Session [{mode.upper()} - {sub_mode.upper()}] ---")
    if mode == "discover": print(f"Category Filter: {category.upper()}")
    print(f"Bankroll       : ${BANKROLL:,.2f}")
    print(f"Base Ego       : {base_ego:.3f} (Brier Score Ratio)")
    
    market_stream = generate_market_stream(mode, sub_mode, target_slugs, history, category, max_offset, exclude_mode)

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
 
        user_input = input("Enter % bounds ('16-51' or '42'), 's' (skip), 'r' (redo), 'q' (quit): ").strip()
 
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
            fee_rate = m.get('feeSchedule', {}).get('rate', 0.05)
            
            action, true_price, raw_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
                lower, upper, pm_bid, pm_ask, fee_rate, BANKROLL, base_ego
            )
            
            res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            days_until = seconds_left / 86400.0

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
                        # Index 0 is YES, Index 1 is NO
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
                        slippage = vwap_true - true_price
                        
                        # Re-calculate edge and APY using VWAP price
                        win_prob = true_price + edge
                        adjusted_edge = win_prob - vwap_true
                        _, adjusted_apy = calculate_annualized_yield(adjusted_edge, vwap_true, days_until)

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
                if slippage > 0:
                    print(f"VWAP PRICE   : {vwap_true*100:.2f}% (Slippage: +{slippage*100:.2f}%)")
                    print(f"ADJUSTED EDGE: {adjusted_edge*100:.2f}% (Target: {edge*100:.2f}%)")
                    print(f"ADJUSTED APY : {adjusted_apy*100:.2f}% (Target: {apy*100:.2f}%)")
                    if adjusted_edge < MIN_EDGE:
                        print(f"[WARNING] Slippage reduces edge below MIN_EDGE ({MIN_EDGE*100:.1f}%)!")
                else:
                    print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
                    print(f"APY          : {apy*100:.2f}% (Compounded)")
                    
                print(f"KELLY %      : {raw_kelly*100:.2f}% of bankroll")
                print(f"ALLOCATION   : ${final_alloc:,.2f}")
                
                portfolio_data.append({
                    "Question": question[:50] + "..",
                    "Action": action,
                    "Ego": f"{dynamic_ego:.2f}",
                    "Price": f"{vwap_true*100:.1f}%",
                    "Alloc": f"${final_alloc:,.2f}"
                })
            else:
                if action == "THIN_EDGE": reason = f"Edge below {MIN_EDGE*100}% threshold"
                elif action == "NONE": reason = "Trapped inside bid-ask spread"
                else: reason = "Unknown"
                
                print(f"ACTION       : $0 Allocation ({reason})")
                print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
            
            print(f"EXPOSURE     : {cumulative_exposure*100:.1f}% of Bankroll Used")
            print(f"SESSION DATA : {len(session_trades)} predictions | Total Allocated: ${sum(session_trades.values()):,.2f}\n")

            current_index += 1 
                
        except ValueError as e: 
            print(f"\n[!] Error: {e}. Please try again.\n")
            
    save_history(stringify_overflow(history))
    return pd.DataFrame(portfolio_data)

if __name__ == "__main__":
    while True:
        print(" 1: Discover Markets")
        print(" 2: Review Markets")
        print(" 3: Target Specific URLs/Slugs")
        print(" 4: Delete all data")
        print(" 5: Import Polymarket Wallet Positions")
        print(" 6: Custom Market Calculator")
        print(" 7: Manually Resolve Open Markets")
        print(" 8: Exit")
        
        choice = input("> ").strip()
        
        targets = None
        category = "All"
        session_max_offset = 0
        target_exclude_mode = "none" # Default initialization
        
        if choice == "8":
            print("Exiting program.")
            break
            
        elif choice == "7":
            history = load_history()
            open_preds_dict = history.get("predicted", {})
            if not open_preds_dict:
                print("No open predictions to resolve.")
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
            print(f"\nResolving: {ensure_list(preds)[0].get('question')}")
            print(" 1: YES")
            print(" 2: NO")
            print(" 3: HALF")
            out_sel = input("> ").strip()
            if out_sel == "1": outcome = 1.0
            elif out_sel == "2": outcome = 0.0
            elif out_sel == "3": outcome = 0.5
            else:
                print("Cancelled.")
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
                    
                action, true_price, raw_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
                    lower, upper, m_lower, m_upper, fee_rate, calc_bankroll, base_ego
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
                    
                    save_q = input("\nSave this custom market to history? (y/n): ").strip().lower()
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
                        
                else:
                    if action == "THIN_EDGE": reason = f"Edge below {MIN_EDGE*100}% threshold"
                    elif action == "NONE": reason = "Trapped inside bid-ask spread"
                    else: reason = "Unknown"
                    print(f"ACTION       : $0 Allocation ({reason})")
                    print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
                    print(f"DYNAMIC EGO  : {dynamic_ego:.2f} (Base {base_ego:.2f})")
                
            except ValueError as e:
                print(f"\n[!] Error: {e}. Please try again.\n")
            continue
            
        elif choice == "5":
            history = import_polymarket_positions(POLYGON_WALLET, load_history())
            save_history(stringify_overflow(history))
            continue
            
        elif choice == "4":
            confirm = input("WARNING: Type 'CONFIRM' to delete all local data: ")
            if confirm == "CONFIRM":
                if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
                print("All data deleted.")
            else:
                print("Data deletion cancelled.")
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
                print("No valid inputs parsed. Returning to main menu.\n")
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
            
            # Fetch Brink to establish dynamic boundary checks
            platform_max_offset = find_platform_brink()
            max_pages = (platform_max_offset // 100) + 1
            
            while True:
                market_mode = input(f"\nDiscover Mode - Select Page Limit:\n 1: All Active Markets (Full Platform, {max_pages} pages)\n 2: Custom Top Pages (1 to {max_pages})\n> ").strip()
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
                print("Invalid selection. Please choose 1 or 2.")
                
        elif choice == "2":
            while True:
                review_mode = input("\nReview Mode - Select Sub-Mode:\n 1: 'Predicted' Markets\n 2: 'Skipped' Markets\n 3: 'Resolved' Markets\n 4: All (Predicted + Skipped + Resolved)\n> ").strip()
                if review_mode in ["1", "2", "3", "4"]:
                    op_mode = "review"
                    sub_mode = {"1": "predicted", "2": "skipped", "3": "resolved", "4": "all"}[review_mode]
                    break
                print("Invalid selection. Please choose 1, 2, 3, or 4.")
                
        else:
            print("Invalid selection. Please choose 1-6.")
            continue
            
        portfolio = run_prediction_session(mode=op_mode, sub_mode=sub_mode, target_slugs=targets, category=category, max_offset=session_max_offset, exclude_mode=target_exclude_mode)
        
        if portfolio is not None and not portfolio.empty:
            print("\n--- Final Session Allocations ---")
            pd.set_option('display.max_colwidth', None) 
            print(portfolio.to_string(index=False))
        else:
            print("\nNo allocations made.")