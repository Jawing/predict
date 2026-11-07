"""analyse.py - Standalone market scanning, aggregation, and interactive paging."""
import sys
import math
from datetime import datetime, timezone
import config as cfg
from utils import parse_outcome_prices, format_time_remaining, parse_market_date, ansi_link

PAGE_SIZE = 40  # Rows per terminal page
GAMMA_API = cfg.GAMMA_API
api = cfg.api


def extract_market_metrics(m):
    """Safely extracts base metrics, calculating spreads and velocities."""
    prices = parse_outcome_prices(m)
    odds = prices[0] if len(prices) > 0 else 0.50

    # Liquidity Spreads
    try:
        pm_bid = float(m.get('bestBid', odds))
        pm_ask = float(m.get('bestAsk', odds))
        spread = pm_ask - pm_bid
    except Exception:
        pm_bid, pm_ask, spread = odds, odds, 0.0

    # Time Horizon
    _, seconds_left, days_until = parse_market_date(m.get('endDate'))

    # Standard Volumes & Orderbook Metrics
    m_vol = float(m.get('volumeNum', 0))
    m_24h = float(m.get('volume24hr', 0))
    m_liq = float(m.get('liquidity', 0))

    # Synthesized Metrics
    velocity = (m_24h / m_vol) if m_vol > 0 else 0.0

    # Pure Epistemic Uncertainty (Efficiency), Epistemic Impact Index (EII), & Compounding Yield (CEY)
    eff = 4.0 * odds * (1.0 - odds)
    eii = eff * m_vol
    days_left = max(0.1, days_until) if days_until is not None else (max(0.1, seconds_left / 86400.0) if seconds_left and seconds_left > 0 else 30.0)
    cey = eii / math.sqrt(days_left + 7.0)

    return odds, spread, seconds_left, m_vol, m_24h, velocity, m_liq, eff, eii, cey


def safe_clean_str(text, max_len=23):
    """Sanitizes strings for safe terminal rendering across any Windows codepage/encoding."""
    import sys
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        t = str(text)
        t.encode(encoding)
    except Exception:
        t = str(text).encode('ascii', 'ignore').decode('ascii')
    t = ' '.join(t.split())
    return (t[:max_len - 2] + "..") if len(t) > max_len else t


def get_category_sort_config(sort_mode, vector_mode=None):
    """Maps sort_mode and vector_mode to a sorting key function, default direction, and human-readable label."""
    effective_mode = str(sort_mode).lower() if sort_mode is not None else "1"

    def _safe_vel(s):
        return (s['24h_vol'] / s['volume']) if s['volume'] > 0 else 0.0

    def _safe_spr(s):
        return (s['spread_sum'] / s['spread_count']) if s.get('spread_count', 0) > 0 else 999.0

    def _safe_comp(s):
        return (s['comp_sum'] / s['events']) if s.get('events', 0) > 0 else 0.0

    SORT_DEFS = {
        "1": (lambda x: x[1].get('cey', 0.0), True, "Compounding Epistemic Yield (CEY)"),
        "2": (lambda x: x[1]['eii'], True, "Epistemic Impact (EII)"),
        "3": (lambda x: x[1]['volume'], True, "Total Volume"),
        "4": (lambda x: x[1]['24h_vol'], True, "24h Volume"),
        "5": (lambda x: x[1]['liquidity'], True, "Liquidity (TVL)"),
        "6": (lambda x: _safe_vel(x[1]), True, "Velocity (24h/Vol)"),
        "7": (lambda x: _safe_spr(x[1]), False, "Avg Spread"),
        "8": (lambda x: x[1]['events'], True, "Event Count"),
        "9": (lambda x: _safe_comp(x[1]), True, "Competitiveness"),
        "10": (lambda x: x[1]['events'], True, "Event Count"),
        "11": (lambda x: x[1]['oi'], True, "Open Interest"),
        "12": (lambda x: (x[1]['eii'] / x[1]['volume']) if x[1]['volume'] > 0 else 0.0, True, "Avg Efficiency (50/50)"),
        "13": (lambda x: x[1]['markets'], True, "Market Count"),
        "cey": (lambda x: x[1].get('cey', 0.0), True, "Compounding Epistemic Yield (CEY)"),
        "eii": (lambda x: x[1]['eii'], True, "Epistemic Impact (EII)"),
        "vol": (lambda x: x[1]['volume'], True, "Total Volume"),
        "volume": (lambda x: x[1]['volume'], True, "Total Volume"),
        "24h": (lambda x: x[1]['24h_vol'], True, "24h Volume"),
        "volume24hr": (lambda x: x[1]['24h_vol'], True, "24h Volume"),
        "liq": (lambda x: x[1]['liquidity'], True, "Liquidity (TVL)"),
        "liquidity": (lambda x: x[1]['liquidity'], True, "Liquidity (TVL)"),
        "vel": (lambda x: _safe_vel(x[1]), True, "Velocity (24h/Vol)"),
        "velocity": (lambda x: _safe_vel(x[1]), True, "Velocity (24h/Vol)"),
        "spread": (lambda x: _safe_spr(x[1]), False, "Avg Spread"),
        "oi": (lambda x: x[1]['oi'], True, "Open Interest"),
        "events": (lambda x: x[1]['events'], True, "Event Count"),
        "markets": (lambda x: x[1]['markets'], True, "Market Count"),
    }
    return SORT_DEFS.get(effective_mode, (lambda x: x[1]['eii'], True, "Epistemic Impact (EII)"))


def fetch_global_overview(fetch_depth=2000, top_x=None, vector_mode="1", sort_mode="1", reverse_sort=False):
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
    total_eii = 0.0
    total_cey = 0.0
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
                    break
                events = response.json()
            except Exception as e:
                print(f"[!] API Error: {e}")
                break

            if not events or not isinstance(events, list):
                break

            for event in events:
                if not isinstance(event, dict):
                    continue
                event_id = event.get('id')
                if event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event_id)
                total_events += 1

                event_vol = 0
                event_24h_vol = 0
                event_liq = 0
                event_markets = 0
                event_spread_sum = 0.0
                event_spread_count = 0
                event_eii = 0.0
                event_cey = 0.0

                event_oi = float(event.get('openInterest', event.get('openInterestNum', 0)))
                event_comp = float(event.get('competitive', 0))

                for m in event.get('markets', []):
                    if m.get('active') and not m.get('closed') and m.get('umaResolutionStatus') != 'resolved':
                        total_markets += 1
                        event_markets += 1
                        m_v = float(m.get('volumeNum', 0))
                        m_24 = float(m.get('volume24hr', 0))
                        m_l = float(m.get('liquidity', 0))
                        event_vol += m_v
                        event_24h_vol += m_24
                        event_liq += m_l

                        prices = parse_outcome_prices(m)
                        m_odds = prices[0] if len(prices) > 0 else 0.50
                        m_eff = 4.0 * m_odds * (1.0 - m_odds)
                        m_eii = m_eff * m_v
                        event_eii += m_eii
                        _, sec_left, days_u = parse_market_date(m.get('endDate'))
                        days_c = max(0.1, days_u) if days_u is not None else 30.0
                        event_cey += m_eii / math.sqrt(days_c + 7.0)

                        try:
                            pm_bid = float(m.get('bestBid', 0))
                            pm_ask = float(m.get('bestAsk', 0))
                            if pm_ask > pm_bid > 0:
                                event_spread_sum += (pm_ask - pm_bid)
                                event_spread_count += 1
                        except Exception:
                            pass

                total_volume += event_vol
                total_24h_volume += event_24h_vol
                total_liquidity += event_liq
                total_eii += event_eii
                total_cey += event_cey

                if event_vol > 0 or event_liq > 0:
                    total_oi += event_oi

                tags = event.get('tags', [])
                if not tags:
                    tags = [{'label': 'Uncategorized'}]

                seen_event_labels = set()
                for t in tags:
                    if isinstance(t, dict):
                        label = t.get('label', 'Unknown').strip().title()
                    elif isinstance(t, str):
                        label = t.strip().title()
                    else:
                        continue

                    if not label or label.lower() == 'all':
                        continue

                    if label in seen_event_labels:
                        continue
                    seen_event_labels.add(label)

                    if label not in category_stats:
                        category_stats[label] = {
                            'events': 0,
                            'markets': 0,
                            'volume': 0.0,
                            '24h_vol': 0.0,
                            'liquidity': 0.0,
                            'oi': 0.0,
                            'eii': 0.0,
                            'cey': 0.0,
                            'spread_sum': 0.0,
                            'spread_count': 0,
                            'comp_sum': 0.0
                        }

                    category_stats[label]['events'] += 1
                    category_stats[label]['markets'] += event_markets
                    category_stats[label]['volume'] += event_vol
                    category_stats[label]['24h_vol'] += event_24h_vol
                    category_stats[label]['liquidity'] += event_liq
                    category_stats[label]['eii'] += event_eii
                    category_stats[label]['cey'] += event_cey
                    if event_vol > 0 or event_liq > 0:
                        category_stats[label]['oi'] += event_oi
                    category_stats[label]['spread_sum'] += event_spread_sum
                    category_stats[label]['spread_count'] += event_spread_count
                    category_stats[label]['comp_sum'] += event_comp

            offset += limit
            print(f"    ...scanned {total_events} unique events", end="\r")

    print("\n\n" + "=" * 146)
    print(f" [GLOBAL OVERVIEW: {overview_label.upper()}] ".center(146, "="))
    print("=" * 146)
    print(f" Total Unique Events Scanned : {total_events:,}")
    print(f" Total Active Markets        : {total_markets:,}")
    print(f" Total Volume                : ${total_volume:,.0f}")
    print(f" Total Epistemic Impact (EII): ${total_eii:,.0f}")
    print(f" Total 24h Volume            : ${total_24h_volume:,.0f}")
    print(f" Total Liquidity (TVL)       : ${total_liquidity:,.0f}")
    print(f" Total Open Interest         : ${total_oi:,.0f}")
    if total_volume > 0:
        print(f" Avg Market Velocity         : {(total_24h_volume / total_volume * 100):.2f}%")
        print(f" Platform Epistemic Ratio    : {(total_eii / total_volume * 100):.1f}%")

    if not category_stats:
        print("[-] No categories found.")
        print("=" * 146 + "\n")
        return

    sort_key, default_rev, sort_label = get_category_sort_config(sort_mode, vector_mode)
    final_rev = not default_rev if reverse_sort else default_rev
    sorted_cats = sorted(category_stats.items(), key=sort_key, reverse=final_rev)

    is_all = (
        top_x is None
        or top_x == 0
        or (isinstance(top_x, str) and str(top_x).lower() in ("all", "0", "-1", "*"))
    )

    if is_all:
        display_cats = sorted_cats
        cat_header_title = f" ALL {len(display_cats)} CATEGORIES (Sorted by {sort_label}{' [DESC]' if final_rev else ' [ASC]'}) "
    else:
        try:
            display_limit = max(1, int(top_x))
            display_cats = sorted_cats[:display_limit]
        except (ValueError, TypeError):
            display_cats = sorted_cats
        cat_header_title = f" TOP {len(display_cats)} CATEGORIES (Sorted by {sort_label}{' [DESC]' if final_rev else ' [ASC]'}) "

    print("-" * 146)
    print(cat_header_title.center(146, "-"))
    print("-" * 146)

    header = (
        f" {'Rank':<5} | {'Category Name':<22} | {'Markets':<7} | {'Events':<6} | "
        f"{'Total Vol':<13} | {'EII ($)':<12} | {'24h Vol':<11} | {'Liquidity':<12} | {'Open Interest':<13} | "
        f"{'Vel %':<6} | {'Share %':<7} | {'Avg Spr':<7}"
    )
    print(header)
    print("-" * 146)

    current_idx = 0
    total_cats = len(display_cats)

    while current_idx < total_cats:
        chunk = display_cats[current_idx : current_idx + PAGE_SIZE]
        for idx_offset, (tag, stats) in enumerate(chunk):
            rank = current_idx + idx_offset + 1
            vol_str = f"${stats['volume']:,.0f}"
            eii_str = f"${stats.get('eii', 0.0):,.0f}"
            v24_str = f"${stats['24h_vol']:,.0f}"
            liq_str = f"${stats['liquidity']:,.0f}"
            oi_str = f"${stats['oi']:,.0f}"

            vel = (stats['24h_vol'] / stats['volume'] * 100) if stats['volume'] > 0 else 0.0
            vel_str = f"{vel:.1f}%"

            share = (stats['volume'] / total_volume * 100) if total_volume > 0 else 0.0
            share_str = f"{share:.1f}%"

            spr = (stats['spread_sum'] / stats['spread_count'] * 100) if stats.get('spread_count', 0) > 0 else 0.0
            spr_str = f"{spr:.1f}%"

            cat_display = safe_clean_str(tag, 22)

            row = (
                f" {rank:<5} | {cat_display:<22} | {stats['markets']:<7,} | {stats['events']:<6,} | "
                f"{vol_str:<13} | {eii_str:<12} | {v24_str:<11} | {liq_str:<12} | {oi_str:<13} | "
                f"{vel_str:<6} | {share_str:<7} | {spr_str:<7}"
            )
            print(row)

        print("-" * 146)

        end_idx = min(current_idx + len(chunk), total_cats)
        if end_idx >= total_cats:
            print("=" * 146 + "\n")
            break

        prompt = f"\n[Showing {end_idx}/{total_cats}] [Right/Down/n/Enter] Next | [Left/Up/p] Prev | [a] Dump All | [q/Esc] Main Menu: "
        action = get_navigation_key(prompt)

        if action == 'quit':
            print("=" * 146 + "\n")
            break
        elif action in ('all', 'dump'):
            current_idx += len(chunk)
            while current_idx < total_cats:
                rem_chunk = display_cats[current_idx : current_idx + PAGE_SIZE]
                for idx_offset, (tag, stats) in enumerate(rem_chunk):
                    rank = current_idx + idx_offset + 1
                    vol_str = f"${stats['volume']:,.0f}"
                    eii_str = f"${stats.get('eii', 0.0):,.0f}"
                    v24_str = f"${stats['24h_vol']:,.0f}"
                    liq_str = f"${stats['liquidity']:,.0f}"
                    oi_str = f"${stats['oi']:,.0f}"
                    vel = (stats['24h_vol'] / stats['volume'] * 100) if stats['volume'] > 0 else 0.0
                    vel_str = f"{vel:.1f}%"
                    share = (stats['volume'] / total_volume * 100) if total_volume > 0 else 0.0
                    share_str = f"{share:.1f}%"
                    spr = (stats['spread_sum'] / stats['spread_count'] * 100) if stats.get('spread_count', 0) > 0 else 0.0
                    spr_str = f"{spr:.1f}%"
                    cat_display = safe_clean_str(tag, 22)
                    row = (
                        f" {rank:<5} | {cat_display:<22} | {stats['markets']:<7,} | {stats['events']:<6,} | "
                        f"{vol_str:<13} | {eii_str:<12} | {v24_str:<11} | {liq_str:<12} | {oi_str:<13} | "
                        f"{vel_str:<6} | {share_str:<7} | {spr_str:<7}"
                    )
                    print(row)
                current_idx += len(rem_chunk)
            print("=" * 146 + "\n")
            break
        elif action == 'next':
            current_idx += PAGE_SIZE
            print("-" * 146)
            print(header)
            print("-" * 146)
        elif action == 'prev':
            current_idx = max(0, current_idx - PAGE_SIZE)
            print("-" * 146)
            print(header)
            print("-" * 146)


def fetch_and_compile_markets(fetch_depth=500, category_filter=None, sort_mode="1", reverse_sort=False, exclude_expired=False, open_only=False):
    """Fetches events, flattens sub-markets, and securely calculates OI and Epistemic Impact Index (EII)."""
    print(f"\n[*] Fetching top {fetch_depth} active events from Polymarket...")

    pred_ids = set()
    try:
        from storage import load_history
        hist = load_history()
        pred_ids = set(str(k) for k in hist.get('predicted', {}).keys())
    except Exception:
        pass

    api_order_map = {
        "1": "volume",     # EII = Eff * Vol -> fetch highest volume candidates
        "2": "volume",
        "3": "volume24hr",
        "4": "liquidity",
        "7": "endDate",
        "8": "competitive",
        "9": "startDate"
    }
    order_param = api_order_map.get(sort_mode, "volume")
    ascending_param = "true" if sort_mode == "7" else "false"
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

        if not events or not isinstance(events, list):
            break

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

            event_oi = float(event.get('openInterest', event.get('openInterestNum', 0)))
            has_valid_market = False

            for m in event.get('markets', []):
                if not m.get('active') or m.get('closed') or m.get('umaResolutionStatus') == 'resolved':
                    continue

                m_id = str(m.get('id', ''))
                is_pred = m_id in pred_ids
                if open_only and is_pred:
                    continue

                odds, spread, seconds_left, m_vol, m_24h, velocity, m_liq, eff, eii, cey = extract_market_metrics(m)

                if seconds_left is None:
                    continue

                if exclude_expired and seconds_left <= 0:
                    continue

                has_valid_market = True
                question = m.get('question', event_title)
                start_date = m.get('startDate') or event.get('startDate') or ""
                status_str = "PRED" if is_pred else "OPEN"

                all_markets.append({
                    "id": m_id,
                    "status": status_str,
                    "question": question[:45] + ".." if len(question) > 45 else question,
                    "target_slug": parent_slug,
                    "m_vol": m_vol,
                    "m_24h": m_24h,
                    "m_liq": m_liq,
                    "velocity": velocity,
                    "odds": odds,
                    "eff": eff,
                    "eii": eii,
                    "cey": cey,
                    "spread": spread,
                    "seconds": seconds_left,
                    "competitive": event_comp,
                    "start_date": start_date,
                    "oi": event_oi
                })

            if has_valid_market and parent_slug not in seen_event_slugs:
                total_filtered_oi += event_oi
                seen_event_slugs.add(parent_slug)

        offset += limit

    return all_markets, total_filtered_oi


def print_global_summary(markets, total_oi):
    """Aggregates and prints statistics for the filtered dataset."""
    if not markets:
        return

    total_vol = sum(m['m_vol'] for m in markets)
    total_24h = sum(m['m_24h'] for m in markets)
    total_liq = sum(m['m_liq'] for m in markets)
    total_eii = sum(m.get('eii', 0.0) for m in markets)

    avg_velocity = (total_24h / total_vol) if total_vol > 0 else 0
    avg_spread = sum(m['spread'] for m in markets) / len(markets)
    epistemic_ratio = (total_eii / total_vol * 100) if total_vol > 0 else 0.0

    print("\n" + "=" * 70)
    print(" [FILTERED SUMMARY] ".center(70, "="))
    print("=" * 70)
    print(f" Total Markets Found : {len(markets):,}")
    print(f" Total Volume        : ${total_vol:,.0f}")
    print(f" Epistemic Impact EII: ${total_eii:,.0f} ({epistemic_ratio:.1f}% open uncertainty)")
    print(f" Total 24h Volume    : ${total_24h:,.0f}")
    print(f" Total Liquidity     : ${total_liq:,.0f}")
    print(f" Total Open Interest : ${total_oi:,.0f}")
    print(f" Avg Market Velocity : {avg_velocity*100:.1f}%")
    print(f" Avg Platform Spread : {avg_spread*100:.2f}%")
    print("=" * 70 + "\n")


def sort_markets(markets, sort_mode, reverse_sort):
    """Sorts the flattened list based on selected metric and direction."""
    if sort_mode in ("13", "none"):
        return markets[::-1] if reverse_sort else markets

    sort_map = {
        "1": ("cey", True),
        "2": ("eii", True),
        "3": ("m_vol", True),
        "4": ("m_24h", True),
        "5": ("m_liq", True),
        "6": ("velocity", True),
        "7": ("spread", False),
        "8": ("seconds", False),
        "9": ("competitive", True),
        "10": ("start_date", True),
        "11": ("oi", True),
        "12": ("eff", True),
        "cey": ("cey", True),
        "eii": ("eii", True),
        "vol": ("m_vol", True),
    }

    key, default_rev = sort_map.get(str(sort_mode).lower(), ("cey", True))
    final_direction = not default_rev if reverse_sort else default_rev

    return sorted(markets, key=lambda x: x.get(key, 0) or 0, reverse=final_direction)


def get_navigation_key(prompt):
    """Reads a navigation key (arrow keys, n/p, q, Enter/Space, a for all, or # to predict) for display_pager."""
    print(prompt, end="", flush=True)
    if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
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
                elif ch in ('\r', '\n', ' '):
                    print()
                    return 'next'
                elif ch in ('n', 'N'):
                    print()
                    return 'next'
                elif ch in ('p', 'P', 'b', 'B'):
                    print()
                    return 'prev'
                elif ch in ('a', 'A'):
                    print()
                    return 'all'
                elif ch in ('q', 'Q', '\x1b'):
                    print()
                    return 'quit'
                elif ch.isdigit() or ch in ('#', ':'):
                    buf = ch
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                    while True:
                        c2 = msvcrt.getwch()
                        if c2 in ('\r', '\n'):
                            print()
                            clean = buf.lstrip('#:').strip()
                            return f"predict:{clean}"
                        elif c2 == '\x08':
                            buf = buf[:-1]
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()
                            if not buf:
                                break
                        elif c2.isdigit():
                            buf += c2
                            sys.stdout.write(c2)
                            sys.stdout.flush()
        except Exception:
            pass

    try:
        raw = input().strip().lower()
        if raw in ('q', 'quit', 'exit'):
            return 'quit'
        elif raw in ('p', 'prev', 'previous', 'b', 'back', 'left', '<-', '<'):
            return 'prev'
        elif raw in ('a', 'all', 'dump'):
            return 'all'
        elif raw.startswith('#') or raw.startswith('predict ') or raw.isdigit():
            clean = raw.replace('predict', '').replace('#', '').strip()
            if clean.isdigit():
                return f"predict:{clean}"
        else:
            return 'next'
    except (EOFError, KeyboardInterrupt):
        return 'quit'


def format_market_row(m, show_odds, rank=None):
    """Formats a single market dictionary into an aligned terminal row."""
    status_str = m.get('status', 'OPEN')
    cey_str = f"{m.get('cey', 0.0):,.0f}"
    eii_str = f"${m.get('eii', 0.0):,.0f}"
    m_vol_str = f"${m['m_vol']:,.0f}"
    h24_str = f"${m['m_24h']:,.0f}"
    liq_str = f"${m['m_liq']:,.0f}"

    vel_str = f"{m['velocity']*100:.0f}%"
    spr_str = f"{m['spread']*100:.1f}%"
    time_str = format_time_remaining(m['seconds'])

    odds_str = f"{m['odds']*100:.1f}%"
    eff_val = m.get('eff', 4.0 * m['odds'] * (1.0 - m['odds']))
    eff_str = f"{eff_val * 100:.1f}%"
    url = f"https://polymarket.com/event/{m['target_slug']}"
    slug_link = ansi_link(url, f"{m['target_slug']:<26}")
    q_link = ansi_link(url, m['question'])

    idx_str = f"{rank:3d} | " if rank is not None else ""
    row = (
        f"{idx_str}{status_str:<6} | {cey_str:<9} | {eii_str:<11} | {m_vol_str:<10} | {h24_str:<9} | {liq_str:<9} | "
        f"{vel_str:<5} | {spr_str:<6} | {time_str:<12} | "
    )
    if show_odds:
        row += f"{odds_str:<5} | {eff_str:<5} | "
    row += f"{slug_link} | {q_link}"
    return row


def display_pager(markets, show_odds):
    """Pages through the locally sorted dataset with forward, backward, and direct predict navigation."""
    if not markets:
        return
    total_markets = len(markets)
    current_idx = 0

    header = (
        f"{'#':<3} | {'Status':<6} | {'CEY':<9} | {'EII ($)':<11} | {'Mkt Vol':<10} | {'24h Vol':<9} | {'Liq':<9} | "
        f"{'Vel %':<5} | {'Spread':<6} | {'Ends In':<12} | "
    )
    if show_odds:
        header += f"{'Odds':<5} | {'Eff%':<5} | "
    header += f"{'Target URL Slug':<26} | Market Question"

    while True:
        chunk = markets[current_idx : current_idx + PAGE_SIZE]

        print("=" * 195)
        print(header)
        print("=" * 195)

        for i, m in enumerate(chunk):
            print(format_market_row(m, show_odds, rank=current_idx + i + 1))

        print("=" * 195)

        end_idx = min(current_idx + len(chunk), total_markets)
        prompt = f"\n[Showing {end_idx}/{total_markets}] [Enter/n] Next | [p] Prev | [#] Predict | [a] Dump All | [q/Esc] Menu: "

        action = get_navigation_key(prompt)

        if action == 'quit':
            break
        elif str(action).startswith("predict:"):
            try:
                pick_num = int(action.split(":")[1])
                target_idx = pick_num - 1
                if 0 <= target_idx < total_markets:
                    target_m = markets[target_idx]
                    print(f"\n[*] Launching Prediction Session for Market #{pick_num}:")
                    print(f"    Question : {target_m['question']}")
                    print(f"    Event    : {target_m['target_slug']}\n")
                    try:
                        from session import run_prediction_session
                        run_prediction_session(
                            mode="target",
                            target_slugs=[target_m['target_slug']],
                            target_market_id=target_m['id']
                        )
                    except Exception as e:
                        print(f"[!] Prediction session error: {e}")
                else:
                    print(f"\n[!] Invalid market #{pick_num}. Please pick between 1 and {total_markets}.")
            except Exception:
                pass
            continue
        elif action in ('all', 'dump'):
            current_idx += len(chunk)
            while current_idx < total_markets:
                rem_chunk = markets[current_idx : current_idx + PAGE_SIZE]
                for i, m in enumerate(rem_chunk):
                    print(format_market_row(m, show_odds, rank=current_idx + i + 1))
                current_idx += len(rem_chunk)
            print("=" * 195)
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
    category, sort_mode, show_odds, reverse_sort, exclude_expired = "All", "1", True, False, True
    open_only = False

    SORT_LABELS = {
        "1": "CEY (Compounding Epistemic Yield: EII / sqrt(Time))",
        "2": "EII (Epistemic Impact: Eff * Vol)",
        "3": "Vol",
        "4": "24h Vol",
        "5": "Liquidity",
        "6": "Velocity",
        "7": "Spread",
        "8": "Soonest",
        "9": "Competitive",
        "10": "Newest",
        "11": "OI",
        "12": "Efficiency (Odds 50/50)",
        "13": "None"
    }

    while True:
        print(f"\n1. Category Filter : [{category}]")
        print(f"2. Sort Metric     : [{SORT_LABELS.get(sort_mode, 'CEY')}]")
        print(f"3. Reverse Sort    : [{'ON' if reverse_sort else 'OFF'}]")
        print(f"4. View Odds       : [{'ON' if show_odds else 'OFF'}]")
        print(f"5. Exclude Expired : [{'ON' if exclude_expired else 'OFF'}]")
        print(f"6. Open Only (New) : [{'ON' if open_only else 'OFF'}]")
        print("7. Execute Scan & View Data")
        print("8. Global Stats & Category Overview")
        print("9. Exit")

        choice = input("\n> ").strip()
        if choice == "9":
            break
        elif choice == "1":
            category = input("Enter Category (e.g., 'Politics', 'Crypto') or 'All': ").strip().title() or "All"
        elif choice == "2":
            print("Sort by: [1] CEY (Compounding Yield) [2] EII (Epistemic Impact) [3] Vol [4] 24h Vol [5] Liquidity [6] Velocity [7] Spread [8] Soonest [9] Competitive [10] Newest [11] OI [12] Efficiency (50/50) [13] No Sorting")
            s_choice = input("> ").strip()
            if s_choice in [str(i) for i in range(1, 14)]:
                sort_mode = s_choice
        elif choice == "3":
            reverse_sort = not reverse_sort
        elif choice == "4":
            show_odds = not show_odds
        elif choice == "5":
            exclude_expired = not exclude_expired
        elif choice == "6":
            open_only = not open_only
        elif choice == "7":
            try:
                depth = int(input("Search Depth limit (e.g., 500, 1000): ").strip())
            except ValueError:
                depth = 500

            markets, total_oi = fetch_and_compile_markets(
                fetch_depth=depth,
                category_filter=category,
                sort_mode=sort_mode,
                reverse_sort=reverse_sort,
                exclude_expired=exclude_expired,
                open_only=open_only
            )
            if markets:
                print_global_summary(markets, total_oi)
                display_pager(sort_markets(markets, sort_mode, reverse_sort), show_odds)
            else:
                print("[-] No markets found matching criteria.")
        elif choice == "8":
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

            try:
                depth = int(input("Search Depth limit for Overview (e.g., 2000, 5000): ").strip())
            except ValueError:
                depth = 2000

            top_raw = input("How many top categories to display? (e.g., 15, or 'all'): ").strip().lower()
            if not top_raw or top_raw in ("all", "a", "0", "-1", "*"):
                top_x = None
            else:
                try:
                    top_x = max(1, int(top_raw))
                except ValueError:
                    top_x = None

            fetch_global_overview(
                fetch_depth=depth,
                top_x=top_x,
                vector_mode=v_choice,
                sort_mode=sort_mode,
                reverse_sort=reverse_sort
            )


if __name__ == "__main__":
    run_interactive_analyzer()
