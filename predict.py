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
KELLY_FRACTION = 1.0
MAX_VOLUME_IMPACT = 0.01 

# --- MICROSTRUCTURE DEFENSES ---
MIN_EDGE = 0.02          # 2% minimum mathematical edge to bother executing
MAX_DAYS = 90            # Ignore markets locking up capital for more than 3 months
MAX_SPREAD = 0.15        # Ignore markets with bid-ask spreads wider than 15%
EXTREME_ODDS = 0.03      # Ignore tail-risk markets below 3% or above 97%

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

def update_resolutions(history):
    resolved_count = 0
    for market_id in list(history.get("predicted", {}).keys()):
        try:
            resp = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}").json()
            if resp.get("closed") and not resp.get("active"):
                prices = json.loads(resp.get("outcomePrices", '["0.5", "0.5"]'))
                if prices[0] in ["1", "1.0"]: outcome = 1.0
                elif prices[1] in ["1", "1.0"]: outcome = 0.0
                else: continue 
                
                pred_data = history["predicted"][market_id]
                history["resolved"][market_id] = {
                    "question": pred_data.get("question", "Unknown"),
                    "pu": pred_data.get("pu", 0.5),
                    "pm": pred_data.get("pm", 0.5),
                    "outcome": outcome,
                    "date_resolved": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                }
                del history["predicted"][market_id]
                resolved_count += 1
        except Exception:
            continue
            
    if resolved_count > 0:
        print(f"\n[*] BACKGROUND SYSTEM: Auto-resolved and scored {resolved_count} closed markets.")
    return history

def calculate_base_ego(history):
    resolved = history.get("resolved", {})
    if not resolved:
        return 0.50 
        
    bs_u_total, bs_m_total = 0.0, 0.0
    
    for data in resolved.values():
        outcome = data.get("outcome", 0)
        bs_u_total += (data.get("pu", 0.5) - outcome) ** 2
        bs_m_total += (data.get("pm", 0.5) - outcome) ** 2
        
    if bs_u_total == 0 and bs_m_total == 0: 
        return 0.50 
        
    return bs_m_total / (bs_u_total + bs_m_total)

def parse_user_input(user_input):
    numbers = [float(x) for x in re.findall(r'\d+\.?\d*', user_input)]
    if len(numbers) == 1:
        return numbers[0] / 100.0, numbers[0] / 100.0
    elif len(numbers) >= 2:
        return min(numbers[0], numbers[1]) / 100.0, max(numbers[0], numbers[1]) / 100.0
    else:
        raise ValueError("No numbers found")

def calculate_allocation(lower_bound, upper_bound, pm_bid, pm_ask, fee_rate, volume, bankroll, base_ego, kelly, max_vol):
    eps = 0.01
    pu_mid = (lower_bound + upper_bound) / 2.0
    spread = upper_bound - lower_bound
    dynamic_ego = max(0.00, base_ego * (1.0 - spread))
    
    pu_clip, pm_mid_clip = np.clip(pu_mid, eps, 1-eps), np.clip((pm_bid + pm_ask) / 2.0, eps, 1-eps)
    
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
    
    # Explicitly calculate the mathematical edge
    edge = win_prob - true_price
    
    # MICROSTRUCTURE: Min Edge Filter uses actual Edge
    if edge < MIN_EDGE:
        return "THIN_EDGE", true_price, 0.0, 0.0, dynamic_ego, edge
    
    # Kelly fraction based on the edge
    raw_kelly = max(0, edge / (1 - true_price))
    adj_kelly = raw_kelly * kelly
    
    final_allocation = min(adj_kelly * bankroll, volume * max_vol)
    return action, true_price, adj_kelly, final_allocation, dynamic_ego, edge

def run_prediction_session(mode="discover", sub_mode="all", target_slugs=None):
    history = update_resolutions(load_history())
    live_base_ego = calculate_base_ego(history)
    
    print(f"\n--- Starting Session [{mode.upper()} - {sub_mode.upper()}] ---")
    print(f"Bankroll       : ${BANKROLL:,.2f}")
    print(f"Base Ego       : {live_base_ego:.3f} (Historical Accuracy Weight)\n")
    
    events = []
    if target_slugs:
        # Loop through multiple slugs if provided
        for slug in target_slugs:
            try:
                resp = requests.get(f"https://gamma-api.polymarket.com/events?slug={slug}").json()
                events.extend(resp)
            except Exception as e:
                print(f"API Error fetching slug '{slug}': {e}")
    else:
        try:
            events = requests.get("https://gamma-api.polymarket.com/events?active=true&closed=false&limit=100").json()
        except Exception as e:
            print(f"API Error: {e}")
            return pd.DataFrame()
            
    all_active_markets = []
    for event in events:
        for m in event.get('markets', []):
            if not m.get('active') or m.get('closed') or m.get('umaResolutionStatus') == 'resolved':
                continue
            
            try: pm_mid = float(json.loads(m.get('outcomePrices', '["0.5"]'))[0])
            except: pm_mid = 0.50
            
            # MICROSTRUCTURE: Extreme Odds Filter (Bypassed if user specifically targeted URLs)
            if not target_slugs:
                if pm_mid < EXTREME_ODDS or pm_mid > (1.0 - EXTREME_ODDS):
                    continue
                    
                try:
                    res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
                    days_until = (res_dt - datetime.now(timezone.utc)).days
                    if days_until < 0 or days_until > MAX_DAYS:
                        continue 
                except Exception:
                    continue 
            
            market_id = m.get('id')
            
            if mode == "discover" and not target_slugs:
                if market_id in history["predicted"] or market_id in history["skipped"]:
                    continue
            elif mode == "review":
                if sub_mode == "predicted" and market_id not in history["predicted"]: continue
                if sub_mode == "skipped" and market_id not in history["skipped"]: continue
                if sub_mode == "all" and (market_id not in history["predicted"] and market_id not in history["skipped"]): continue

            m['parent_slug'] = event.get('slug', 'unknown-event')
            m['parent_name'] = event.get('title', 'Unknown Event')
            all_active_markets.append(m)

    random.shuffle(all_active_markets) 
    
    if not all_active_markets:
        print("\nNo valid markets found for the selected mode/filters.")
        return pd.DataFrame()

    portfolio_data = []
    session_trades = {}
    cumulative_exposure = 0.0 
    i = 0
    
    while i < len(all_active_markets):
        if cumulative_exposure >= 1.0:
            print("\n[!] Maximum capital exposure reached (100%). Session ending.")
            break
            
        m = all_active_markets[i]
        market_id = m.get('id')
        question = m.get('question', 'Unknown Question')
        event_url = f"https://polymarket.com/event/{m.get('parent_slug')}"
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        event_title = m.get('parent_name')
        
        try:
            res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
            exact_date_str = res_dt.strftime("%B %d, %Y")
            days_str = f"{(res_dt - datetime.now(timezone.utc)).days} Days"
        except: exact_date_str, days_str = "Unknown", "Unknown"

        try: pm_mid = float(json.loads(m.get('outcomePrices', '["0.5"]'))[0])
        except: pm_mid = 0.50
        try: pm_bid, pm_ask = float(m.get('bestBid', pm_mid)), float(m.get('bestAsk', pm_mid))
        except: pm_bid, pm_ask = pm_mid, pm_mid
        
        # Bypassed Spread cap if user specifically requested targeted URLs
        if not target_slugs and (pm_ask - pm_bid) > MAX_SPREAD:
            i += 1 
            continue

        print(f"==================================================")
        print(f"Event    : {event_title}")
        print(f"Market   : {question}")
        print(f"Resolves : {exact_date_str} ({days_str})")
        print(f"Exposure : {cumulative_exposure*100:.1f}% deployed")
        
        user_input = input("Enter % value (bounds '16-51' or '42'), 's' (skip), 'r' (redo), 'q' (quit): ").strip()
        
        if user_input.lower() == 'q':
            print("\nSaving and exiting session...")
            break
            
        elif user_input.lower() == 'r':
            if i > 0:
                print("\n[!] Rolling back to previous market...")
                i -= 1 
                prev_market_id = all_active_markets[i].get('id')
                if prev_market_id in session_trades:
                    refund_amount = session_trades.pop(prev_market_id)
                    cumulative_exposure -= (refund_amount / BANKROLL)
                
                history["predicted"].pop(prev_market_id, None)
                history["skipped"].pop(prev_market_id, None)
            else:
                print("\n[!] Cannot redo. This is the first market of the session.\n")
            continue

        elif user_input.lower() == 's': 
            print("Status: Skipped.\n")
            if market_id not in history["predicted"]:
                history["skipped"][market_id] = {"date": today_str}
            i += 1 
            continue
        
        try:
            lower, upper = parse_user_input(user_input)
            volume = float(m.get('volumeNum', 0))
            fee_rate = m.get('feeSchedule', {}).get('rate', 0.05)
            
            action, true_price, adj_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
                lower, upper, pm_bid, pm_ask, fee_rate, volume, BANKROLL, live_base_ego, KELLY_FRACTION, MAX_VOLUME_IMPACT
            )
            
            history["predicted"][market_id] = {
                "question": question,
                "date": today_str,
                "bounds": f"{(lower*100):.0f}% - {(upper*100):.0f}%",
                "pu": (lower + upper) / 2.0,
                "pm": pm_mid,
                "theoretical_kelly": round(adj_kelly, 4), 
                "allocation": round(final_alloc, 2)
            }
            history["skipped"].pop(market_id, None)
            
            print(f"\n--- MARKET ANALYSIS ---")
            print(f"Market Spread: Bid {pm_bid*100:.1f}% | Ask {pm_ask*100:.1f}% (Spread: {(pm_ask - pm_bid)*100:.1f}%)")
            print(f"User Bounds  : {lower*100:.1f}% to {upper*100:.1f}% (Spread: {(upper-lower)*100:.1f}%)")
            print(f"Dynamic Ego  : {dynamic_ego:.2f} (Base {live_base_ego:.2f})")
            
            if final_alloc > 0:
                cumulative_exposure += final_alloc / BANKROLL
                session_trades[market_id] = final_alloc
                
                print(f"ACTION       : BUY {action} @ {true_price*100:.1f}%")
                print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
                print(f"EXPOSURE     : {cumulative_exposure*100:.1f}% of Bankroll Used")
                print(f"FEE RATE     : {fee_rate*100:.1f}%")
                print(f"VOLUME       : ${volume:,.0f} available (Max impact cap: ${volume*MAX_VOLUME_IMPACT:,.2f})")
                print(f"KELLY ALLOC %: {adj_kelly*100:.2f}% of bankroll")
                print(f"FINAL ALLOC %: {final_alloc/BANKROLL*100:.1f}% of bankroll")
                print(f"ALLOCATION   : ${final_alloc:,.2f}")
                print(f"SESSION DATA : {len(session_trades)} predictions | Total Allocated: ${sum(session_trades.values()):,.2f}\n")
                print(f"LINK         : {event_url}\n")
                
                portfolio_data.append({
                    "Question": question[:50] + "..",
                    "Action": action,
                    "Ego": f"{dynamic_ego:.2f}",
                    "Price": f"{true_price*100:.1f}%",
                    "Alloc": f"${final_alloc:,.0f}"
                })
            else:
                if action == "THIN_EDGE": reason = f"Edge below {MIN_EDGE*100}% minimum threshold"
                elif action == "NONE": reason = "Trapped inside bid-ask spread"
                else: reason = "No mathematical edge"
                
                print(f"ACTION       : $0 Allocation ({reason})")
                print(f"USER EDGE    : {edge*100:.2f}% (After Fees & Spread)")
                print(f"EXPOSURE     : {cumulative_exposure*100:.1f}% of Bankroll Used")
                print(f"SESSION DATA : {len(session_trades)} trades | Total Allocated: ${sum(session_trades.values()):,.2f}\n")
                print(f"LINK         : {event_url}\n")
                
            i += 1 
                
        except ValueError: 
            print("\n[!] Error: Invalid input format. Please try again.\n")
            
    save_history(history)
    return pd.DataFrame(portfolio_data)

if __name__ == "__main__":
    print("Select Operation Mode:")
    print(" 1: Discover New Markets (Random)")
    print(" 2: Review 'Predicted' Markets")
    print(" 3: Review 'Skipped' Markets")
    print(" 4: Review All (Predicted + Skipped)")
    print(" 5: Target Specific URLs (Sniper Mode)")
    print(" 6: Complete Reset (Delete History file)")
    
    choice = input("> ").strip()
    
    targets = None
    if choice == "6":
        confirm = input("WARNING: Type 'CONFIRM' to delete all mathematical history and Brier scores: ")
        if confirm == "CONFIRM":
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            print("Reset complete. Exiting.")
        else:
            print("Reset aborted. Exiting.")
        exit()
    elif choice == "5":
        urls_input = input("Paste Polymarket URLs (comma or space separated):\n> ").replace(',', ' ').split()
        targets = []
        for url in urls_input:
            try:
                # Extracts the slug robustly from standard polymarket.com/event/[slug] formats
                slug = url.split("/event/")[1].split("/")[0].split("?")[0]
                targets.append(slug)
            except IndexError:
                print(f"Invalid URL format skipped: {url}")
        
        if not targets:
            print("No valid URLs parsed. Exiting.")
            exit()
        op_mode, sub_mode = "sniper", "single"
    else:
        mode_map = {
            "1": ("discover", "all"), 
            "2": ("review", "predicted"), 
            "3": ("review", "skipped"),
            "4": ("review", "all")
        }
        op_mode, sub_mode = mode_map.get(choice, ("discover", "all"))
    
    portfolio = run_prediction_session(mode=op_mode, sub_mode=sub_mode, target_slugs=targets)
    
    if not portfolio.empty:
        print("\n--- Final Session Allocations ---")
        pd.set_option('display.max_colwidth', None) 
        print(portfolio.to_string(index=False))
    else:
        print("\nNo allocations made.")