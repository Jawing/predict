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

# --- FILTERING PARAMETERS ---
MIN_EDGE = 0.01      # X% minimum mathematical edge to bother executing
MAX_DAYS = 420       # Ignore markets locking up capital for more than X days
MIN_DAYS = 0         # Ignore markets resolving within X days (e.g., 1 day = 24h, 0.5 days = 12h)
EXTREME_ODDS = 0.01  # Ignore tail-odds markets below X% (e.g., 0.01 = 1%)

api = requests.Session()

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
    try:
        prices = m.get('outcomePrices')
        if isinstance(prices, str): pm_mid = float(json.loads(prices)[0])
        elif isinstance(prices, list): pm_mid = float(prices[0])
        else: return None, None, None
    except: return None, None, None
        
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
                prices_raw = resp.get("outcomePrices")
                if isinstance(prices_raw, str): prices = [float(x) for x in json.loads(prices_raw)]
                elif isinstance(prices_raw, list): prices = [float(x) for x in prices_raw]
                else: prices = [0.5, 0.5]
                
                if len(prices) >= 2:
                    if prices[0] == 1.0: outcome = 1.0
                    elif prices[1] == 1.0: outcome = 0.0
                    else: outcome = 0.5 
                else: outcome = 0.5
                
                pred_data = history["predicted"][market_id]
                history["resolved"][market_id] = {
                    "question": pred_data.get("question", "unknown"),
                    "slug": pred_data.get("slug", "unknown"),
                    "pu": pred_data.get("pu", 0.5),
                    "pm": pred_data.get("pm", 0.5),
                    "outcome": outcome
                }
                del history["predicted"][market_id]
                resolved_count += 1
        except Exception:
            continue
            
    if resolved_count > 0:
        print(f"[+] BACKGROUND SYSTEM: Auto-resolved and scored {resolved_count} closed markets.")
    return history

def calculate_base_ego(history):
    resolved = history.get("resolved", {})
    if not resolved: return 0.50 
        
    bs_u_total, bs_m_total = 0.0, 0.0
    for data in resolved.values():
        outcome = data.get("outcome", 0)
        bs_u_total += (data.get("pu", 0.5) - outcome) ** 2
        bs_m_total += (data.get("pm", 0.5) - outcome) ** 2
        
    print(f"[*] Current Brier Score - User: {bs_u_total:.4f} | Market: {bs_m_total:.4f}")
    # If both scores are zero (e.g., no resolved markets or perfect predictions), return neutral ego of 0.5
    if bs_u_total == 0 and bs_m_total == 0: return 0.50 
    return bs_m_total / (bs_u_total + bs_m_total)

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
    pu_mid = (lower_bound + upper_bound) / 2.0
    spread = upper_bound - lower_bound
    
    dynamic_ego = max(0.00, base_ego * (1.0 - spread))
    pu_clip = np.clip(pu_mid, eps, 1-eps)
    pm_mid_clip = np.clip((pm_bid + pm_ask) / 2.0, eps, 1-eps)
    
    logit_pu, logit_pm = np.log(pu_clip / (1-pu_clip)), np.log(pm_mid_clip / (1-pm_mid_clip))
    logit_norm = (dynamic_ego * logit_pu) + ((1 - dynamic_ego) * logit_pm)
    pu_norm = 1 / (1 + np.exp(-logit_norm)) 
    
    cost_yes, cost_no = pm_ask, 1.0 - pm_bid 
    
    if pu_norm > cost_yes:
        action, base_price, win_prob = "YES", cost_yes, pu_norm
    elif (1.0 - pu_norm) > cost_no:
        action, base_price, win_prob = "NO", cost_no, 1.0 - pu_norm
    else:
        return "NONE", 0.0, 0.0, 0.0, dynamic_ego, 0.0
    
    true_price = base_price + (fee_rate * base_price * (1 - base_price))
    edge = win_prob - true_price
    
    if edge < MIN_EDGE:
        return "THIN_EDGE", true_price, 0.0, 0.0, dynamic_ego, edge
    
    raw_kelly = max(0, edge / (1 - true_price))
    final_allocation = raw_kelly * bankroll
    
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

def validate_market(m, history, mode):
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
    if mode == "discover" and (market_id in history.get("predicted", {}) or market_id in history.get("skipped", {})):
        return False, "Already predicted or skipped"
        
    _, _, pm_mid = extract_prices(m)
    if pm_mid is None:
        return False, "Broken or missing price data from API"
    if pm_mid < EXTREME_ODDS or pm_mid > (1.0 - EXTREME_ODDS):
        return False, f"Odds ({pm_mid*100:.1f}%) hit extreme tail-risk filter"
        
    try:
        res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
        seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()

        if seconds_left <= 0:
            return False, "Market has already expired"
            
        days_until = seconds_left / 86400.0
        if days_until < MIN_DAYS or days_until > MAX_DAYS:
            return False, f"Time horizon ({days_until:.1f} days) outside bounds"
    except:
        return False, "Invalid end date format"
            
    return True, "Valid"

def generate_market_stream(mode, sub_mode, target_slugs, history, category="All", max_offset=0):
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
                    
                    is_valid, reason = validate_market(m, history, mode)
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
                        is_valid, reason = validate_market(m, history, mode)
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
                    
                    is_valid, reason = validate_market(m, history, mode)
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

def run_prediction_session(mode="discover", sub_mode="all", target_slugs=None, category="All", max_offset=0):
    history = update_resolutions(load_history())
    base_ego = calculate_base_ego(history)
    
    print(f"\n--- Starting Session [{mode.upper()} - {sub_mode.upper()}] ---")
    if mode == "discover": print(f"Category Filter: {category.upper()}")
    print(f"Bankroll       : ${BANKROLL:,.2f}")
    print(f"Base Ego       : {base_ego:.3f} (Brier Score Ratio)\n")
    
    market_stream = generate_market_stream(mode, sub_mode, target_slugs, history, category, max_offset)

    portfolio_data = []
    session_trades = {}
    cumulative_exposure = 0.0 
    session_markets = [] 
    current_index = 0
    
    while True:
        if cumulative_exposure >= 1.0:
            print(f"\n[!] Maximum capital exposure reached. Session ending.")
            break
            
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
        market_id = m.get('id')
        question = m.get('question', 'unknown')
        event_slug = m.get('parent_slug', 'unknown')
        event_url = f"https://polymarket.com/event/{event_slug}"
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        
        try:
            res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
            exact_date_str = res_dt.strftime("%B %d, %Y")
            
            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            if seconds_left < 86400:
                h, rem = divmod(seconds_left, 3600)
                mn, s = divmod(rem, 60)
                days_str = f"{int(h)}h {int(mn)}m {int(s)}s"
            else:
                days_str = f"{int(seconds_left // 86400)} Days"
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
            
            history["predicted"][market_id] = {
                "question": question,
                "slug": event_slug,
                "date": today_str,
                "pu": (lower + upper) / 2.0,
                "pm": pm_mid
            }
            history["skipped"].pop(market_id, None)
            
            print(f"\n--- MARKET ANALYSIS ---")
            print(f"Market Spread: Bid {pm_bid*100:.1f}% | Ask {pm_ask*100:.1f}% (Spread: {(pm_ask - pm_bid)*100:.1f}%)")
            print(f"User Bounds  : {lower*100:.1f}% to {upper*100:.1f}% (Spread: {(upper-lower)*100:.1f}%)")
            print(f"Dynamic Ego  : {dynamic_ego:.2f} (Base {base_ego:.2f})")
            
            if final_alloc > 0.01:
                cumulative_exposure += final_alloc / BANKROLL
                session_trades[market_id] = final_alloc
                
                print(f"ACTION       : BUY {action} @ {true_price*100:.1f}%")
                print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
                print(f"KELLY %      : {raw_kelly*100:.2f}% of bankroll")
                print(f"ALLOCATION   : ${final_alloc:,.2f}")
                
                portfolio_data.append({
                    "Question": question[:50] + "..",
                    "Action": action,
                    "Ego": f"{dynamic_ego:.2f}",
                    "Price": f"{true_price*100:.1f}%",
                    "Alloc": f"${final_alloc:,.2f}"
                })
            else:
                if action == "THIN_EDGE": reason = f"Edge below {MIN_EDGE*100}% threshold"
                elif action == "NONE": reason = "Trapped inside bid-ask spread"
                else: reason = "No mathematical edge"
                
                print(f"ACTION       : $0 Allocation ({reason})")
                print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
            
            print(f"EXPOSURE     : {cumulative_exposure*100:.1f}% of Bankroll Used")
            print(f"SESSION DATA : {len(session_trades)} predictions | Total Allocated: ${sum(session_trades.values()):,.2f}\n")

            current_index += 1 
                
        except ValueError as e: 
            print(f"\n[!] Error: {e}. Please try again.\n")
            
    save_history(history)
    return pd.DataFrame(portfolio_data)

if __name__ == "__main__":
    while True:
        print(" 1: Discover Markets")
        print(" 2: Review Markets")
        print(" 3: Target Specific URLs/Slugs")
        print(" 4: Delete all data")
        print(" 5: Exit")
        
        choice = input("> ").strip()
        
        targets = None
        category = "All"
        session_max_offset = 0 
        
        if choice == "5":
            print("Exiting program.")
            break
            
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
                review_mode = input("\nReview Mode - Select Sub-Mode:\n 1: 'Predicted' Markets\n 2: 'Skipped' Markets\n 3: All (Predicted + Skipped)\n> ").strip()
                if review_mode in ["1", "2", "3"]:
                    op_mode = "review"
                    sub_mode = {"1": "predicted", "2": "skipped", "3": "all"}[review_mode]
                    break
                print("Invalid selection. Please choose 1, 2, or 3.")
                
        else:
            print("Invalid selection. Please choose 1-5.")
            continue
            
        portfolio = run_prediction_session(mode=op_mode, sub_mode=sub_mode, target_slugs=targets, category=category, max_offset=session_max_offset)
        
        if portfolio is not None and not portfolio.empty:
            print("\n--- Final Session Allocations ---")
            pd.set_option('display.max_colwidth', None) 
            print(portfolio.to_string(index=False))
        else:
            print("\nNo allocations made.")