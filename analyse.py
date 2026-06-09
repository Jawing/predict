import requests
import json
import os
from datetime import datetime, timezone

# --- CONFIGURATION ---
GAMMA_API = "https://gamma-api.polymarket.com"
PAGE_SIZE = 40  # Rows per terminal page

api = requests.Session()

def extract_market_metrics(m):
    """Safely extracts base metrics, calculating spreads and velocities."""
    # Base Odds (Strict parsing to handle API strings vs lists)
    try: 
        prices = m.get('outcomePrices')
        if isinstance(prices, str):
            odds = float(json.loads(prices)[0])
        elif isinstance(prices, list):
            odds = float(prices[0])
        else:
            odds = 0.50
    except: 
        odds = 0.50
        
    # Liquidity Spreads
    try:
        pm_bid = float(m.get('bestBid', odds))
        pm_ask = float(m.get('bestAsk', odds))
        spread = pm_ask - pm_bid
    except:
        pm_bid, pm_ask, spread = odds, odds, 0.0
        
    # Time Horizon
    try:
        res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
        time_delta = res_dt - datetime.now(timezone.utc)
        seconds_left = time_delta.total_seconds()
    except: 
        seconds_left = None
        
    # Standard Volumes & Orderbook Metrics
    m_vol = float(m.get('volumeNum', 0))
    m_24h = float(m.get('volume24hr', 0))
    m_liq = float(m.get('liquidity', 0))
    
    # --- SYNTHESIZED METRICS ---
    velocity = (m_24h / m_vol) if m_vol > 0 else 0.0
        
    return odds, spread, seconds_left, m_vol, m_24h, velocity, m_liq

def fetch_global_overview(fetch_depth=2000, top_x=20):
    """Sweeps the platform to build a macro view of categories and liquidity distribution."""
    print(f"\n[*] Sweeping up to {fetch_depth} events for global category stats...")
    
    offset = 0
    total_events = 0
    total_markets = 0
    total_volume = 0
    total_24h_volume = 0
    total_liquidity = 0
    total_oi = 0
    category_stats = {}
    
    while offset < fetch_depth:
        limit = min(100, fetch_depth - offset)
        url = f"{GAMMA_API}/events?active=true&closed=false&limit={limit}&offset={offset}"
        
        try:
            response = api.get(url, timeout=10)
            if response.status_code != 200: break
            events = response.json()
        except Exception as e:
            print(f"[!] API Error: {e}")
            break
            
        if not events: break
            
        total_events += len(events)
        
        for event in events:
            event_vol = 0
            event_24h_vol = 0
            event_liq = 0
            
            # Extract Open Interest ONCE per Event
            event_oi = float(event.get('openInterest', event.get('openInterestNum', 0)))
            
            for m in event.get('markets', []):
                if m.get('active') and not m.get('closed') and m.get('umaResolutionStatus') != 'resolved':
                    # Only tally events that haven't officially expired
                    try:
                        res_dt = datetime.fromisoformat(m.get('endDate').replace('Z', '+00:00'))
                        if (res_dt - datetime.now(timezone.utc)).total_seconds() <= 0:
                            continue
                    except:
                        pass

                    total_markets += 1
                    event_vol += float(m.get('volumeNum', 0))
                    event_24h_vol += float(m.get('volume24hr', 0))
                    event_liq += float(m.get('liquidity', 0))
                    
            total_volume += event_vol
            total_24h_volume += event_24h_vol
            total_liquidity += event_liq
            
            # Add OI to the global total ONLY if the event actually has active, valid markets
            if event_vol > 0 or event_liq > 0:
                total_oi += event_oi
            
            tags = event.get('tags', [])
            for t in tags:
                if isinstance(t, dict):
                    label = t.get('label', 'Unknown').strip().title()
                elif isinstance(t, str):
                    label = t.strip().title()
                else:
                    continue
                    
                if not label: continue
                
                if label not in category_stats:
                    category_stats[label] = {'events': 0, 'volume': 0.0, '24h_vol': 0.0, 'liquidity': 0.0, 'oi': 0.0}
                    
                category_stats[label]['events'] += 1
                category_stats[label]['volume'] += event_vol
                category_stats[label]['24h_vol'] += event_24h_vol
                category_stats[label]['liquidity'] += event_liq
                category_stats[label]['oi'] += event_oi
                
        offset += limit
        print(f"    ...scanned {offset} events", end="\r")
        
    print("\n\n" + "="*100)
    print(" 🌍 GLOBAL OVERVIEW ".center(100, "="))
    print("="*100)
    print(f" 📊 Total Events Scanned   : {total_events:,}")
    print(f" 📈 Total Active Markets   : {total_markets:,}")
    print(f" 💰 Total Volume           : ${total_volume:,.0f}")
    print(f" ⏱️ Total 24h Volume       : ${total_24h_volume:,.0f}")
    print(f" 💧 Total Liquidity (TVL)  : ${total_liquidity:,.0f}")
    print(f" 📜 Total Open Interest    : ${total_oi:,.0f}")
    print("-" * 100)
    print(f" 🏷️  TOP {top_x} CATEGORIES (Sorted by Event Count) ".center(100))
    print("-" * 100)
    print(f" {'Rank':<5} | {'Category Name':<25} | {'Events':<6} | {'Total Vol':<12} | {'24h Vol':<10} | {'Liquidity':<12}")
    print("-" * 100)
    
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]['events'], reverse=True)
    
    for i, (tag, stats) in enumerate(sorted_cats[:top_x], 1):
        vol_str = f"${stats['volume']:,.0f}"
        v24_str = f"${stats['24h_vol']:,.0f}"
        liq_str = f"${stats['liquidity']:,.0f}"
        print(f" {i:<5} | {tag:<25} | {stats['events']:<6} | {vol_str:<12} | {v24_str:<10} | {liq_str:<12}")
        
    print("="*100 + "\n")

def fetch_and_compile_markets(fetch_depth=500, category_filter=None):
    """Fetches events, flattens sub-markets, and securely calculates OI."""
    print(f"\n[*] Fetching top {fetch_depth} active events from Polymarket...")
    
    all_markets = []
    offset = 0
    total_filtered_oi = 0
    seen_event_slugs = set()
    
    while offset < fetch_depth:
        limit = min(100, fetch_depth - offset)
        url = f"{GAMMA_API}/events?active=true&closed=false&limit={limit}&offset={offset}"
        
        try:
            events = api.get(url, timeout=10).json()
        except Exception as e:
            print(f"[!] API Error: {e}")
            break
            
        if not events: break
            
        for event in events:
            if category_filter and category_filter.lower() != "all":
                tags = []
                for t in event.get('tags', []):
                    if isinstance(t, dict):
                        tags.append(t.get('label', '').lower())
                    elif isinstance(t, str):
                        tags.append(t.lower())
                        
                if not any(category_filter.lower() in t for t in tags):
                    continue

            event_title = event.get('title', 'Unknown Event')
            parent_slug = event.get('slug', '')
            
            # Grab OI at the event level
            event_oi = float(event.get('openInterest', event.get('openInterestNum', 0)))
            has_valid_market = False
            
            for m in event.get('markets', []):
                if not m.get('active') or m.get('closed') or m.get('umaResolutionStatus') == 'resolved':
                    continue
                    
                odds, spread, seconds_left, m_vol, m_24h, velocity, m_liq = extract_market_metrics(m)
                
                # STRICT PIPELINE FILTER: Discard null dates or expired markets
                if seconds_left is None or seconds_left <= 0:
                    continue
                
                has_valid_market = True
                question = m.get('question', event_title)
                
                all_markets.append({
                    "question": question[:45] + ".." if len(question) > 45 else question,
                    "target_slug": parent_slug,
                    "m_vol": m_vol,
                    "m_24h": m_24h,
                    "m_liq": m_liq,
                    "velocity": velocity,
                    "odds": odds,
                    "spread": spread,
                    "seconds": seconds_left
                })
                
            # Prevent Double Counting: Only add Event OI once if the event had surviving markets
            if has_valid_market and parent_slug not in seen_event_slugs:
                total_filtered_oi += event_oi
                seen_event_slugs.add(parent_slug)
                
        offset += limit
        
    return all_markets, total_filtered_oi

def print_global_summary(markets, total_oi):
    """Aggregates and prints statistics for the filtered dataset."""
    if not markets: return
    
    total_vol = sum(m['m_vol'] for m in markets)
    total_24h = sum(m['m_24h'] for m in markets)
    total_liq = sum(m['m_liq'] for m in markets)
    
    avg_velocity = (total_24h / total_vol) if total_vol > 0 else 0
    avg_spread = sum(m['spread'] for m in markets) / len(markets)

    print("\n" + "="*70)
    print(" 📊 FILTERED SUMMARY ".center(70, "="))
    print("="*70)
    print(f" Total Markets Found : {len(markets):,}")
    print(f" Total Volume        : ${total_vol:,.0f}")
    print(f" Total 24h Volume    : ${total_24h:,.0f}")
    print(f" Total Liquidity     : ${total_liq:,.0f}")
    print(f" Total Open Interest : ${total_oi:,.0f}")
    print(f" Avg Market Velocity : {avg_velocity*100:.1f}%")
    print(f" Avg Platform Spread : {avg_spread*100:.2f}%")
    print("="*70 + "\n")

def sort_markets(markets, sort_mode, reverse_sort):
    """Sorts the flattened list based on selected metric and direction."""
    sort_map = {
        "1": ("m_vol", True),    # Total Volume
        "2": ("m_24h", True),    # 24h Volume
        "3": ("m_liq", True),    # Liquidity
        "4": ("velocity", True), # Velocity
        "5": ("spread", False),  # Tightest Spread 
        "6": ("seconds", False)  # Resolving Soonest
    }
    
    key, default_rev = sort_map.get(sort_mode, ("m_vol", True))
    final_direction = not default_rev if reverse_sort else default_rev
    
    return sorted(markets, key=lambda x: x[key], reverse=final_direction)

def display_pager(markets, show_odds):
    """Pages through the locally sorted dataset."""
    if not markets: return
    total_markets = len(markets)
    current_idx = 0
    
    while current_idx < total_markets:
        chunk = markets[current_idx : current_idx + PAGE_SIZE]
        
        print("=" * 165)
        header = f"{'Mkt Vol':<9} | {'24h Vol':<8} | {'Liq':<8} | {'Vel %':<5} | {'Spread':<6} | {'Ends In':<12} | "
        if show_odds: header += f"{'Odds':<5} | "
        header += f"{'Target URL Slug':<35} | Market Question"
        print(header)
        print("=" * 165)
        
        for m in chunk:
            m_vol_str = f"${m['m_vol']:,.0f}"
            h24_str = f"${m['m_24h']:,.0f}"
            liq_str = f"${m['m_liq']:,.0f}"
            
            vel_str = f"{m['velocity']*100:.0f}%"
            spr_str = f"{m['spread']*100:.1f}%"
            
            # --- Exact Seconds Formatting Logic ---
            secs = m['seconds']
            d = int(secs // 86400)
            rem = secs % 86400
            h = int(rem // 3600)
            rem %= 3600
            mins = int(rem // 60)
            s = int(rem % 60)
            
            if d > 0:
                time_str = f"{d}d {h:02d}h"
            else:
                time_str = f"{h:02d}:{mins:02d}:{s:02d}"

            odds_str = f"{m['odds']*100:.1f}%"
            
            row = f"{m_vol_str:<9} | {h24_str:<8} | {liq_str:<8} | {vel_str:<5} | {spr_str:<6} | {time_str:<12} | "
            if show_odds: row += f"{odds_str:<5} | "
            row += f"{m['target_slug']:<35} | {m['question']}"
            
            print(row)
            
        print("=" * 165)
        
        current_idx += PAGE_SIZE
        if current_idx >= total_markets:
            print("\n[!] End of dataset reached.")
            break
            
        user_choice = input(f"\n[Showing {current_idx}/{total_markets}] 'n' for Next, or 'q' to Main Menu: ").strip().lower()
        if user_choice == 'q': break

def run_interactive_analyzer():
    category, sort_mode, show_odds, reverse_sort = "All", "1", True, False
    
    while True:
        print(f"\n1. Category Filter : [{category}]")
        print(f"2. Sort Metric     : [{'Vol' if sort_mode=='1' else '24h Vol' if sort_mode=='2' else 'Liquidity' if sort_mode=='3' else 'Velocity' if sort_mode=='4' else 'Spread' if sort_mode=='5' else 'Time Left'}]")
        print(f"3. Reverse Sort    : [{'ON' if reverse_sort else 'OFF'}]")
        print(f"4. View Odds       : [{'ON' if show_odds else 'OFF'}]")
        print("5. Execute Scan & View Data")
        print("6. Global Stats & Category Overview")
        print("7. Exit")
        
        choice = input("\n> ").strip()
        if choice == "7": break
        elif choice == "1":
            category = input("Enter Category (e.g., 'Politics', 'Crypto') or 'All': ").strip().title() or "All"
        elif choice == "2":
            print("Sort by: [1] Vol [2] 24h Vol [3] Liquidity [4] Velocity [5] Spread [6] Soonest")
            s_choice = input("> ").strip()
            if s_choice in ["1", "2", "3", "4", "5", "6"]: sort_mode = s_choice
        elif choice == "3":
            reverse_sort = not reverse_sort
        elif choice == "4":
            show_odds = not show_odds
        elif choice == "5":
            try: depth = int(input("Search Depth limit (e.g., 500, 1000): ").strip())
            except ValueError: depth = 500
                
            markets, total_oi = fetch_and_compile_markets(fetch_depth=depth, category_filter=category)
            if markets:
                print_global_summary(markets, total_oi)
                display_pager(sort_markets(markets, sort_mode, reverse_sort), show_odds)
            else:
                print("[-] No markets found matching criteria.")
        elif choice == "6":
            try: depth = int(input("Search Depth limit for Overview (e.g., 2000, 5000): ").strip())
            except ValueError: depth = 2000
            
            try: top_x = int(input("How many top categories to display? (e.g., 15): ").strip())
            except ValueError: top_x = 15
            
            fetch_global_overview(fetch_depth=depth, top_x=top_x)

if __name__ == "__main__":
    run_interactive_analyzer()