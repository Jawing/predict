import requests
import json
import os
from datetime import datetime, timezone

# --- CONFIGURATION ---
GAMMA_API = "https://gamma-api.polymarket.com"
PAGE_SIZE = 40  # Rows per terminal page

api = requests.Session()

def extract_market_metrics(m):
    """Safely extracts base metrics, calculating spreads, velocities, and annualized yields."""
    # Base Odds
    try: 
        odds = float(json.loads(m.get('outcomePrices', '["0.5"]'))[0])
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
        days_left = max(0, (res_dt - datetime.now(timezone.utc)).days)
    except: 
        days_left = 999 
        
    # Volumes
    m_vol = float(m.get('volumeNum', 0))
    m_24h = float(m.get('volume24hr', 0))
    
    # --- SYNTHESIZED METRICS ---
    velocity = (m_24h / m_vol) if m_vol > 0 else 0.0
    
    if days_left > 0 and 0.01 < odds < 0.99:
        ann_yield = ((1.0 / odds) - 1.0) * (365.0 / days_left)
    else:
        ann_yield = 0.0
        
    return odds, spread, days_left, m_vol, m_24h, velocity, ann_yield

def fetch_and_compile_markets(fetch_depth=500, category_filter=None):
    """Fetches events, flattens sub-markets, and links their internal target slugs."""
    print(f"\n[*] Fetching top {fetch_depth} active events from Polymarket...")
    
    all_markets = []
    offset = 0
    
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
                tags = [t.get('label', '').lower() for t in event.get('tags', []) if isinstance(t, dict)]
                if not any(category_filter.lower() in t for t in tags):
                    continue

            event_title = event.get('title', 'Unknown Event')
            parent_slug = event.get('slug', '')
            
            for m in event.get('markets', []):
                if not m.get('active') or m.get('closed') or m.get('umaResolutionStatus') == 'resolved':
                    continue
                    
                odds, spread, days_left, m_vol, m_24h, velocity, ann_yield = extract_market_metrics(m)
                question = m.get('question', event_title)
                
                all_markets.append({
                    "question": question[:45] + ".." if len(question) > 45 else question,
                    "target_slug": parent_slug,
                    "m_vol": m_vol,
                    "m_24h": m_24h,
                    "velocity": velocity,
                    "yield": ann_yield,
                    "odds": odds,
                    "spread": spread,
                    "days": days_left
                })
                
        offset += limit
        
    return all_markets

def print_global_summary(markets):
    """Aggregates and prints macro statistics for the filtered dataset."""
    if not markets: return
    
    total_vol = sum(m['m_vol'] for m in markets)
    total_24h = sum(m['m_24h'] for m in markets)
    avg_velocity = (total_24h / total_vol) if total_vol > 0 else 0
    avg_spread = sum(m['spread'] for m in markets) / len(markets)
    
    highest_yield_mkt = max(markets, key=lambda x: x['yield'], default=None)
    highest_vel_mkt = max(markets, key=lambda x: x['velocity'], default=None)

    print("\n" + "="*70)
    print(" 📊 GLOBAL MACRO SUMMARY ".center(70, "="))
    print("="*70)
    print(f" Total Markets Found : {len(markets):,}")
    print(f" Captured Liquidity  : ${total_vol:,.0f}")
    print(f" Captured 24h Volume : ${total_24h:,.0f}")
    print(f" Avg Market Velocity : {avg_velocity*100:.1f}%")
    print(f" Avg Platform Spread : {avg_spread*100:.2f}%")
    print("-" * 70)
    if highest_vel_mkt and highest_vel_mkt['velocity'] > 0:
        print(f" ⚡ Highest Velocity : {highest_vel_mkt['velocity']*100:.1f}% -> {highest_vel_mkt['question']}")
    if highest_yield_mkt and highest_yield_mkt['yield'] > 0:
        print(f" 📈 Highest Ann. Yield: {highest_yield_mkt['yield']*100:,.0f}% -> {highest_yield_mkt['question']}")
    print("="*70 + "\n")

def sort_markets(markets, sort_mode):
    """Sorts the flattened list based on selected metric."""
    sort_map = {
        "1": ("m_vol", True),    # Total Volume
        "2": ("m_24h", True),    # 24h Volume
        "3": ("velocity", True), # Velocity
        "4": ("yield", True),    # Annualized Yield
        "5": ("spread", False),  # Tightest Spread (Ascending)
        "6": ("days", False)     # Resolving Soonest (Ascending)
    }
    key, reverse = sort_map.get(sort_mode, ("m_vol", True))
    return sorted(markets, key=lambda x: x[key], reverse=reverse)

def display_pager(markets, show_odds):
    """Pages through the locally sorted dataset, handling view toggles and links."""
    if not markets: return
    total_markets = len(markets)
    current_idx = 0
    
    while current_idx < total_markets:
        chunk = markets[current_idx : current_idx + PAGE_SIZE]
        
        print("=" * 150)
        header = f"{'Mkt Vol':<10} | {'24h Vol':<8} | {'Vel %':<6} | {'Spread':<6} | {'Ann. Yld':<9} | {'Ends':<5} | "
        if show_odds: header += f"{'Odds':<5} | "
        header += f"{'Target URL Snipe Slug':<35} | Market Question"
        print(header)
        print("=" * 150)
        
        for m in chunk:
            m_vol_str = f"${m['m_vol']:,.0f}"
            h24_str = f"${m['m_24h']:,.0f}"
            vel_str = f"{m['velocity']*100:.1f}%"
            spr_str = f"{m['spread']*100:.1f}%"
            
            yld = m['yield'] * 100
            yld_str = f"{yld:,.0f}%" if yld < 10000 else ">10k%"
            days_str = f"{m['days']}d" if m['days'] != 999 else "N/A"
            odds_str = f"{m['odds']*100:.0f}%"
            
            row = f"{m_vol_str:<10} | {h24_str:<8} | {vel_str:<6} | {spr_str:<6} | {yld_str:<9} | {days_str:<5} | "
            if show_odds: row += f"{odds_str:<5} | "
            row += f"{m['target_slug']:<35} | {m['question']}"
            
            print(row)
            
        print("=" * 150)
        
        current_idx += PAGE_SIZE
        if current_idx >= total_markets:
            print("\n[!] End of dataset reached.")
            break
            
        user_choice = input(f"\n[Showing {current_idx}/{total_markets}] 'n' for Next, or 'q' to Main Menu: ").strip().lower()
        if user_choice == 'q': break

def run_interactive_analyzer():
    category, sort_mode, show_odds = "All", "1", True
    
    while True:
        print(f"1. Category Filter : [{category}]")
        print(f"2. Sort Metric     : [{'Vol' if sort_mode=='1' else '24h Vol' if sort_mode=='2' else 'Velocity' if sort_mode=='3' else 'Yield' if sort_mode=='4' else 'Spread' if sort_mode=='5' else 'Days'}]")
        print(f"3. View Odds       : [{'ON' if show_odds else 'OFF'}]")
        print("4. Execute Scan & View Data")
        print("5. Exit")
        
        choice = input("> ").strip()
        if choice == "5": break
        elif choice == "1":
            category = input("Enter Category (e.g., 'Politics', 'Crypto') or 'All': ").strip().title() or "All"
        elif choice == "2":
            print("Sort by: [1] Vol [2] 24h Vol [3] Velocity % [4] Ann. Yield % [5] Tightest Spread [6] Soonest")
            s_choice = input("> ").strip()
            if s_choice in ["1", "2", "3", "4", "5", "6"]: sort_mode = s_choice
        elif choice == "3":
            show_odds = not show_odds
        elif choice == "4":
            try: depth = int(input("Search Depth limit (e.g., 500, 1000): ").strip())
            except ValueError: depth = 500
                
            markets = fetch_and_compile_markets(fetch_depth=depth, category_filter=category)
            if markets:
                print_global_summary(markets)
                display_pager(sort_markets(markets, sort_mode), show_odds)
            else:
                print("[-] No markets found matching criteria.")

if __name__ == "__main__":
    run_interactive_analyzer()