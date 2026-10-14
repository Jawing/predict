import requests
import json
import os
from datetime import datetime, timezone

# --- CONFIGURATION ---
GAMMA_API = "https://gamma-api.polymarket.com"
PAGE_SIZE = 40  # Rows per terminal page

api = requests.Session()

from utils import parse_outcome_prices, format_time_remaining



def extract_market_metrics(m):
    """Safely extracts base metrics, calculating spreads and velocities."""
    # Base Odds (Strict parsing to handle API strings vs lists)
    prices = parse_outcome_prices(m)
    odds = prices[0] if len(prices) > 0 else 0.50
        
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

def fetch_global_overview(fetch_depth=2000, top_x=20, vector_mode="1"):
    """Sweeps the platform across selected or aggregated vectors to build a macro view."""
    VECTOR_CONFIGS = {
        "1": [("volume", "false")],
        "2": [("volume24hr", "false")],
        "3": [("liquidity", "false")],
        "4": [("endDate", "true")],
        "5": [("competitive", "false")],
        "6": [("startDate", "false")],
        "7": [
            ("volume", "false"),        # High volume
            ("volume", "true"),         # Low volume / niche
            ("volume24hr", "false"),    # High 24h volume
            ("liquidity", "false"),     # Deep liquidity
            ("endDate", "true"),        # Ending soonest
            ("endDate", "false"),       # Far horizon
            ("competitive", "false"),   # Close 50/50 odds
            ("startDate", "false")      # Freshly listed
        ]
    }
    VECTOR_LABELS = {
        "1": "Volume (Macro Dollars)",
        "2": "24h Volume (Trending)",
        "3": "Liquidity (TVL)",
        "4": "Ending Soonest",
        "5": "Competitive (50/50 Odds)",
        "6": "Newest Listed (Start Date)",
        "7": "Multi-Vector Aggregated (Deduplicated, 8-Vectors)"
    }
    
    sweep_plan = VECTOR_CONFIGS.get(vector_mode, VECTOR_CONFIGS["1"])
    overview_label = VECTOR_LABELS.get(vector_mode, "Volume (Macro Dollars)")
    
    print(f"\n[*] Sweeping events for global overview via [{overview_label}]...")
    
    seen_event_ids = set()
    total_events = 0
    total_markets = 0
    total_volume = 0
    total_24h_volume = 0
    total_liquidity = 0
    total_oi = 0
    category_stats = {}
    
    depth_per_vector = fetch_depth if len(sweep_plan) == 1 else min(2000, max(200, fetch_depth // len(sweep_plan)))
    
    for order_param, ascending_param in sweep_plan:
        offset = 0
        while offset < depth_per_vector:
            limit = min(100, depth_per_vector - offset)
            url = f"{GAMMA_API}/events?active=true&closed=false&limit={limit}&offset={offset}&order={order_param}&ascending={ascending_param}"
            
            try:
                response = api.get(url, timeout=10)
                if response.status_code != 200:
                    if response.status_code == 422:
                        pass
                    break
                events = response.json()
            except Exception as e:
                print(f"[!] API Error: {e}")
                break
                
            if not events or not isinstance(events, list): break
            
            for event in events:
                if not isinstance(event, dict): continue
                event_id = event.get('id')
                if event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event_id)
                total_events += 1
                
                event_vol = 0
                event_24h_vol = 0
                event_liq = 0
                
                # Extract Open Interest ONCE per Event
                event_oi = float(event.get('openInterest', event.get('openInterestNum', 0)))
                
                for m in event.get('markets', []):
                    if m.get('active') and not m.get('closed') and m.get('umaResolutionStatus') != 'resolved':
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
            print(f"    ...scanned {total_events} unique events", end="\r")
        
    print("\n\n" + "="*100)
    print(f" [GLOBAL OVERVIEW: {overview_label.upper()}] ".center(100, "="))
    print("="*100)
    print(f" Total Unique Events Scanned : {total_events:,}")
    print(f" Total Active Markets        : {total_markets:,}")
    print(f" Total Volume                : ${total_volume:,.0f}")
    print(f" Total 24h Volume            : ${total_24h_volume:,.0f}")
    print(f" Total Liquidity (TVL)       : ${total_liquidity:,.0f}")
    print(f" Total Open Interest         : ${total_oi:,.0f}")
    print("-" * 100)
    print(f" TOP {top_x} CATEGORIES (Sorted by Event Count) ".center(100, "-"))
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

def fetch_and_compile_markets(fetch_depth=500, category_filter=None, sort_mode="1", reverse_sort=False, exclude_expired=False):
    """Fetches events, flattens sub-markets, and securely calculates OI."""
    print(f"\n[*] Fetching top {fetch_depth} active events from Polymarket...")
    
    # Map selected sort mode to Gamma API order/ascending parameters
    api_order_map = {
        "1": "volume",
        "2": "volume24hr",
        "3": "liquidity",
        "6": "endDate",
        "7": "competitive",
        "8": "startDate"
    }
    order_param = api_order_map.get(sort_mode, "volume")
    ascending_param = "true" if sort_mode == "6" else "false"
    if reverse_sort:
        ascending_param = "false" if ascending_param == "true" else "true"
        
    all_markets = []
    offset = 0
    total_filtered_oi = 0
    seen_event_slugs = set()
    
    while offset < fetch_depth:
        limit = min(100, fetch_depth - offset)
        url = f"{GAMMA_API}/events?active=true&closed=false&limit={limit}&offset={offset}&order={order_param}&ascending={ascending_param}"
        
        try:
            response = api.get(url, timeout=10)
            if response.status_code != 200:
                if response.status_code == 422:
                    print(f"[*] Reached Polymarket API offset limit at {offset} events.")
                else:
                    print(f"[!] API Error: HTTP {response.status_code}")
                break
            events = response.json()
        except Exception as e:
            print(f"[!] API Error: {e}")
            break
            
        if not events or not isinstance(events, list): break
            
        for event in events:
            if not isinstance(event, dict):
                continue
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
            event_comp = float(event.get('competitive', 0))
            
            # Grab OI at the event level
            event_oi = float(event.get('openInterest', event.get('openInterestNum', 0)))
            has_valid_market = False
            
            for m in event.get('markets', []):
                if not m.get('active') or m.get('closed') or m.get('umaResolutionStatus') == 'resolved':
                    continue
                    
                odds, spread, seconds_left, m_vol, m_24h, velocity, m_liq = extract_market_metrics(m)
                
                # STRICT PIPELINE FILTER: Discard null dates (and expired markets if excluded)
                if seconds_left is None:
                    continue
                
                if exclude_expired and seconds_left <= 0:
                    continue
                
                has_valid_market = True
                question = m.get('question', event_title)
                start_date = m.get('startDate') or event.get('startDate') or ""
                
                all_markets.append({
                    "question": question[:45] + ".." if len(question) > 45 else question,
                    "target_slug": parent_slug,
                    "m_vol": m_vol,
                    "m_24h": m_24h,
                    "m_liq": m_liq,
                    "velocity": velocity,
                    "odds": odds,
                    "spread": spread,
                    "seconds": seconds_left,
                    "competitive": event_comp,
                    "start_date": start_date
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
    print(" [FILTERED SUMMARY] ".center(70, "="))
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
    if sort_mode == "9":
        return markets[::-1] if reverse_sort else markets
        
    sort_map = {
        "1": ("m_vol", True),       # Total Volume
        "2": ("m_24h", True),       # 24h Volume
        "3": ("m_liq", True),       # Liquidity
        "4": ("velocity", True),    # Velocity
        "5": ("spread", False),     # Tightest Spread 
        "6": ("seconds", False),    # Resolving Soonest
        "7": ("competitive", True), # Competitive
        "8": ("start_date", True)   # Newest Listed (startDate)
    }
    
    key, default_rev = sort_map.get(sort_mode, ("m_vol", True))
    final_direction = not default_rev if reverse_sort else default_rev
    
    return sorted(markets, key=lambda x: x[key], reverse=final_direction)

def get_navigation_key(prompt):
    """Reads a navigation key (arrow keys, n/p, q, Enter/Space) for display_pager."""
    print(prompt, end="", flush=True)
    try:
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            if ch == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            if ch in ('\x00', '\xe0'):
                code = msvcrt.getwch()
                if code in ('M', 'P'):  # Right arrow or Down arrow
                    print()
                    return 'next'
                elif code in ('K', 'H'):  # Left arrow or Up arrow
                    print()
                    return 'prev'
            elif ch in ('\r', '\n', ' '):  # Enter or Space
                print()
                return 'next'
            elif ch in ('n', 'N'):
                print()
                return 'next'
            elif ch in ('p', 'P', 'b', 'B'):
                print()
                return 'prev'
            elif ch in ('q', 'Q', '\x1b'):  # 'q' or Esc
                print()
                return 'quit'
    except Exception:
        pass

    try:
        raw = input().strip().lower()
        if raw in ('q', 'quit', 'exit'):
            return 'quit'
        elif raw in ('p', 'prev', 'previous', 'b', 'back', 'left', '<-', '<'):
            return 'prev'
        else:
            return 'next'
    except (EOFError, KeyboardInterrupt):
        return 'quit'

def display_pager(markets, show_odds):
    """Pages through the locally sorted dataset with forward and backward navigation."""
    if not markets: return
    total_markets = len(markets)
    current_idx = 0
    
    while True:
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
            time_str = format_time_remaining(m['seconds'])

            odds_str = f"{m['odds']*100:.1f}%"
            
            row = f"{m_vol_str:<9} | {h24_str:<8} | {liq_str:<8} | {vel_str:<5} | {spr_str:<6} | {time_str:<12} | "
            if show_odds: row += f"{odds_str:<5} | "
            row += f"{m['target_slug']:<35} | {m['question']}"
            
            print(row)
            
        print("=" * 165)
        
        end_idx = min(current_idx + len(chunk), total_markets)
        prompt = f"\n[Showing {end_idx}/{total_markets}] [Right/Down/n] Next | [Left/Up/p] Prev | [q/Esc] Main Menu: "
        
        action = get_navigation_key(prompt)
        
        if action == 'quit':
            break
        elif action == 'next':
            if current_idx + PAGE_SIZE < total_markets:
                current_idx += PAGE_SIZE
            else:
                print("\n[!] Already on the last page.")
        elif action == 'prev':
            if current_idx > 0:
                current_idx = max(0, current_idx - PAGE_SIZE)
            else:
                print("\n[!] Already on the first page.")

def run_interactive_analyzer():
    category, sort_mode, show_odds, reverse_sort, exclude_expired = "All", "1", True, False, False
    
    SORT_LABELS = {
        "1": "Vol",
        "2": "24h Vol",
        "3": "Liquidity",
        "4": "Velocity",
        "5": "Spread",
        "6": "Time Left",
        "7": "Competitive",
        "8": "Newest",
        "9": "None"
    }
    
    while True:
        print(f"\n1. Category Filter : [{category}]")
        print(f"2. Sort Metric     : [{SORT_LABELS.get(sort_mode, 'Vol')}]")
        print(f"3. Reverse Sort    : [{'ON' if reverse_sort else 'OFF'}]")
        print(f"4. View Odds       : [{'ON' if show_odds else 'OFF'}]")
        print(f"5. Exclude Expired : [{'ON' if exclude_expired else 'OFF'}]")
        print("6. Execute Scan & View Data")
        print("7. Global Stats & Category Overview")
        print("8. Exit")
        
        choice = input("\n> ").strip()
        if choice == "8": break
        elif choice == "1":
            category = input("Enter Category (e.g., 'Politics', 'Crypto') or 'All': ").strip().title() or "All"
        elif choice == "2":
            print("Sort by: [1] Vol [2] 24h Vol [3] Liquidity [4] Velocity [5] Spread [6] Soonest [7] Competitive [8] Newest (Start Date) [9] No Sorting")
            s_choice = input("> ").strip()
            if s_choice in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]: sort_mode = s_choice
        elif choice == "3":
            reverse_sort = not reverse_sort
        elif choice == "4":
            show_odds = not show_odds
        elif choice == "5":
            exclude_expired = not exclude_expired
        elif choice == "6":
            try: depth = int(input("Search Depth limit (e.g., 500, 1000): ").strip())
            except ValueError: depth = 500
                
            markets, total_oi = fetch_and_compile_markets(
                fetch_depth=depth,
                category_filter=category,
                sort_mode=sort_mode,
                reverse_sort=reverse_sort,
                exclude_expired=exclude_expired
            )
            if markets:
                print_global_summary(markets, total_oi)
                display_pager(sort_markets(markets, sort_mode, reverse_sort), show_odds)
            else:
                print("[-] No markets found matching criteria.")
        elif choice == "7":
            print("\nSelect Overview Vector:")
            print(" 1: Volume (Default - Macro Dollars & TVL)")
            print(" 2: 24h Volume (Active Daily Velocity)")
            print(" 3: Liquidity (Deepest Orderbooks)")
            print(" 4: Ending Soonest (Near-Term Resolutions)")
            print(" 5: Competitive (Contested 50/50 Odds)")
            print(" 6: Newest Listed (Freshly Created Markets)")
            print(" 7: Multi-Vector Aggregated (Full 8-Vector Sweep, Deduplicated)")
            v_choice = input("> ").strip()
            if v_choice not in ["1", "2", "3", "4", "5", "6", "7"]:
                v_choice = "1"
                
            try: depth = int(input("Search Depth limit for Overview (e.g., 2000, 5000): ").strip())
            except ValueError: depth = 2000
            
            try: top_x = int(input("How many top categories to display? (e.g., 15): ").strip())
            except ValueError: top_x = 15
            
            fetch_global_overview(fetch_depth=depth, top_x=top_x, vector_mode=v_choice)

if __name__ == "__main__":
    run_interactive_analyzer()