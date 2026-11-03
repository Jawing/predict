"""Polymarket API interactions: order books, CLOB streams, Gamma markets, and on-chain RPC balances."""
import json
import random
import threading
import time
from datetime import datetime, timezone, timedelta
import config as cfg
from config import api
from utils import parse_outcome_prices, format_time_remaining, parse_market_date


def get_wallet_usdc_balance(wallet_address):
    """Fetches combined USDC/pUSD balance on Polygon across collateral contracts."""
    if not wallet_address or not wallet_address.startswith("0x"):
        return None
    rpcs = [
        "https://polygon-bor-rpc.publicnode.com",
        "https://1rpc.io/matic",
        "https://polygon.drpc.org"
    ]
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
    """Retrieves current value of open positions from Polymarket Data API (v2)."""
    if not wallet_address or not wallet_address.startswith("0x"):
        return 0.0
    try:
        val_resp = api.get(f"https://data-api.polymarket.com/v2/value?user={wallet_address}", timeout=5).json()
        if isinstance(val_resp, dict):
            data_val = val_resp.get("data", {})
            if isinstance(data_val, dict) and "value" in data_val:
                return float(data_val["value"] or 0.0)
            elif "value" in val_resp:
                return float(val_resp["value"] or 0.0)
    except Exception:
        pass
    if positions:
        total = 0.0
        for pos in positions:
            size = float(pos.get("current_size") if pos.get("current_size") is not None else pos.get("size", 0.0) or 0.0)
            avg_price = float(pos.get("avg_price") if pos.get("avg_price") is not None else pos.get("avgPrice", 0.0) or 0.0)
            value = float(pos.get("current_value") if pos.get("current_value") is not None else pos.get("currentValue", pos.get("initialValue", size * avg_price)) or (size * avg_price))
            total += value
        return total
    return 0.0


def get_market_clob_tokens(m):
    """Returns YES and NO CLOB token IDs for a given market dict."""
    if not isinstance(m, dict):
        return None, None
    clob_ids = m.get('clobTokenIds')
    if not clob_ids:
        return None, None
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except Exception:
            return None, None
    if not isinstance(clob_ids, list) or not clob_ids:
        return None, None
    yes_token = clob_ids[0] if len(clob_ids) > 0 else None
    no_token = clob_ids[1] if len(clob_ids) > 1 else None
    return yes_token, no_token


def refresh_prices_from_clob(m):
    """Pulls best bid/ask and midpoint directly from the CLOB order book."""
    yes_token, _ = get_market_clob_tokens(m)
    if not yes_token:
        return m
    try:
        r = api.get(f"https://clob.polymarket.com/book?token_id={yes_token}", timeout=5)
        if r.status_code == 200:
            book = r.json()
            bids = [float(b['price']) for b in book.get('bids', []) if float(b.get('price', 0)) > 0]
            asks = [float(a['price']) for a in book.get('asks', []) if float(a.get('price', 0)) > 0]
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


def stream_clob_book(m, custom_token_id=None):
    """Live-streams bid/ask quotes and size from CLOB order book."""
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

                best_bid, bid_size = max(bids, key=lambda x: x[0]) if bids else (0.0, 0.0)
                best_ask, ask_size = min(asks, key=lambda x: x[0]) if asks else (1.0, 0.0)

                midpoint = (best_bid + best_ask) / 2.0
                spread = max(0.0, best_ask - best_bid)

                if target_token == yes_token:
                    m['bestBid'], m['bestAsk'] = best_bid, best_ask
                    m['outcomePrices'] = [midpoint, 1.0 - midpoint]
                    m['cached_bid'], m['cached_ask'], m['cached_mid'] = best_bid, best_ask, midpoint
                elif target_token == no_token:
                    yes_bid, yes_ask = 1.0 - best_ask, 1.0 - best_bid
                    yes_mid = (yes_bid + yes_ask) / 2.0
                    m['bestBid'], m['bestAsk'] = yes_bid, yes_ask
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
    """Resolves market question, slug, and end date by looking up Gamma events or markets."""
    if event_cache is None:
        event_cache = {}
    cond_id = pos.get("conditionId")
    event_id = pos.get("eventId")
    event_slug = pos.get("eventSlug")
    pos_slug = pos.get("slug")
    m = None

    # 1. Lookup via parent Event ID
    if event_id:
        event_id_str = f"id:{event_id}"
        if event_id_str not in event_cache:
            try:
                ev_resp = api.get(f"{cfg.GAMMA_API}/events/{event_id}", timeout=5).json()
                event_cache[event_id_str] = ev_resp if isinstance(ev_resp, dict) else {}
            except Exception:
                event_cache[event_id_str] = {}
        ev_data = event_cache.get(event_id_str, {})
        for cand in ev_data.get("markets", []):
            if cand.get("conditionId") == cond_id or (pos_slug and cand.get("slug") == pos_slug):
                m = cand
                break

    # 2. Lookup via parent Event Slug
    if not m and event_slug:
        ev_slug_key = f"slug:{event_slug}"
        if ev_slug_key not in event_cache:
            try:
                ev_list = api.get(f"{cfg.GAMMA_API}/events?slug={event_slug}", timeout=5).json()
                event_cache[ev_slug_key] = ev_list[0] if (isinstance(ev_list, list) and ev_list) else {}
            except Exception:
                event_cache[ev_slug_key] = {}
        ev_data = event_cache.get(ev_slug_key, {})
        for cand in ev_data.get("markets", []):
            if cand.get("conditionId") == cond_id or (pos_slug and cand.get("slug") == pos_slug):
                m = cand
                break

    # 3. Fallback: Lookup via market slug on Gamma API
    if not m and pos_slug:
        try:
            resp = api.get(f"{cfg.GAMMA_API}/markets?slug={pos_slug}", timeout=5).json()
            if isinstance(resp, list):
                for cand in resp:
                    if cand.get("conditionId") == cond_id or cand.get("slug") == pos_slug:
                        m = cand
                        break
        except Exception:
            pass

    market_id = str(m.get("id")) if (m and m.get("id")) else str(cond_id)
    question = (m.get("question") if m else None) or pos.get("title") or "unknown"
    slug = pos.get("eventSlug") or pos.get("slug") or (m.get("slug") if m else "unknown")
    end_date = (m.get("endDate") if m else None) or pos.get("endDate") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return m, market_id, question, slug, end_date


def get_event_sub_markets(m):
    """Returns list of sibling sub-markets for the parent event of market m."""
    sub_markets = m.get("event_markets")
    if sub_markets and len(sub_markets) > 1:
        return sub_markets

    slug = m.get("parent_slug") or m.get("slug")
    if slug and slug != "unknown":
        try:
            ev_list = api.get(f"{cfg.GAMMA_API}/events?slug={slug}", timeout=5).json()
            if isinstance(ev_list, list) and ev_list:
                ev_markets = ev_list[0].get("markets", [])
                if ev_markets:
                    m["event_markets"] = ev_markets
                    return ev_markets
        except Exception:
            pass
    return sub_markets or [m]


def determine_redeemable_outcome(pos, m=None):
    """Returns 1.0 if the held position won, 0.0 if it lost, 0.5 if split."""
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


def extract_prices(m):
    """Safely extracts bestBid, bestAsk, and midpoint prices from market dict."""
    prices = parse_outcome_prices(m)
    if len(prices) < 1:
        return None, None, None
    pm_mid = prices[0]
    try: 
        pm_bid = float(m.get('bestBid', pm_mid))
        pm_ask = float(m.get('bestAsk', pm_mid))
    except Exception: 
        pm_bid, pm_ask = pm_mid, pm_mid
    return pm_bid, pm_ask, pm_mid


def find_platform_brink(limit=100):
    """Probes the upper pagination limit of Polymarket events via binary search."""
    max_guess = 2**16
    low, high, max_valid = 0, max_guess, 0
    while low <= high:
        mid = ((low + high) // 2) // limit * limit
        try:
            res = api.get(f"{cfg.GAMMA_API}/events?active=true&closed=false&limit={limit}&offset={mid}").json()
            if isinstance(res, dict) and 'error' in res:
                high = mid - limit
            elif isinstance(res, list) and len(res) > 0:
                max_valid, low = mid, mid + limit
            else:
                high = mid - limit 
        except Exception:
            high = mid - limit
    return max_valid


def validate_market(m, history, mode, exclude_mode="none"):
    """Filters out closed, paused, non-binary, or parameter-violating markets."""
    if m.get('closed') or m.get('umaResolutionStatus') == 'resolved':
        return False, "Market closed or resolved"
    if not m.get('active'):
        return False, "Market betting paused (Awaiting UMA resolution)"
        
    try:
        outcomes = json.loads(m.get('outcomes', '["Yes", "No"]'))
        if len(outcomes) != 2:
            return False, f"Non-binary categorical market ({len(outcomes)} outcomes)"
    except Exception:
        return False, "Invalid outcomes structure"

    market_id = m.get('id')
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
    if pm_mid < cfg.EXTREME_ODDS or pm_mid > (1.0 - cfg.EXTREME_ODDS):
        return False, f"Odds ({pm_mid*100:.1f}%) hit extreme tail-risk filter"
        
    res_dt, seconds_left, days_until = parse_market_date(m.get('endDate'))
    if res_dt is None or seconds_left is None:
        return False, "Invalid end date format"
    if seconds_left <= 0:
        return False, "Market has expired"
    if days_until < cfg.MIN_DAYS or days_until > cfg.MAX_DAYS:
        return False, f"Time horizon ({days_until:.1f} days) outside bounds"
            
    return True, "Valid"


def generate_market_stream(mode, sub_mode, target_slugs, history, category="All", max_offset=0, exclude_mode="none", discovery_order="volume", target_market_id=None):
    """Yields validated market objects from discovery vectors or targeted slugs."""
    if mode == "discover":
        seen_categories = set()
        seen_market_ids = set()
        while True:
            offset = random.randint(0, max_offset // 100) * 100
            
            if discovery_order == "multi":
                order_param, ascending_param = random.choice([
                    ("volume", "false"), ("volume", "true"),
                    ("volume24hr", "false"), ("liquidity", "false"),
                    ("endDate", "true"), ("endDate", "false"),
                    ("competitive", "false"), ("startDate", "false")
                ])
            elif discovery_order == "volume_asc":
                order_param, ascending_param = "volume", "true"
            elif discovery_order == "startDate":
                order_param, ascending_param = "startDate", "false"
            elif discovery_order == "endDate_desc":
                order_param, ascending_param = "endDate", "false"
            else:
                order_param = discovery_order
                ascending_param = "true" if discovery_order == "endDate" else "false"
                
            try:
                _now = datetime.now(timezone.utc)
                _min_dt = (_now + timedelta(days=cfg.MIN_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
                _max_dt = (_now + timedelta(days=cfg.MAX_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
                events = api.get(
                    f"{cfg.GAMMA_API}/events?active=true&closed=false&limit=100&offset={offset}&order={order_param}&ascending={ascending_param}&end_date_min={_min_dt}&end_date_max={_max_dt}",
                    timeout=10
                ).json()
            except Exception:
                continue
            
            if not events or not isinstance(events, list):
                continue
            random.shuffle(events)
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                if category and category.lower() != "all":
                    tags = [t.get('label', '').lower() for t in event.get('tags', []) if isinstance(t, dict)]
                    matches = [t for t in tags if category.lower() in t]
                    if not matches:
                        continue
                    for match in matches:
                        full_name = match.title()
                        if full_name not in seen_categories:
                            print(f"[+] Category Matched: '{full_name}'")
                            seen_categories.add(full_name)

                raw_markets = event.get('markets', [])
                valid_markets = []
                for m in raw_markets:
                    m_id = m.get('id')
                    if m_id in seen_market_ids:
                        continue
                    m['parent_slug'] = event.get('slug', 'unknown')
                    m['parent_name'] = event.get('title', 'unknown')
                    m['event_markets'] = raw_markets
                    
                    is_valid, reason = validate_market(m, history, mode, exclude_mode)
                    if is_valid: 
                        valid_markets.append(m)
                
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
                events = api.get(f"{cfg.GAMMA_API}/events?slug={slug}", timeout=10).json()
                if not events or not isinstance(events, list):
                    print(f"[-] Invalid or empty data returned for slug: {slug}")
                    continue
                    
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    raw_markets = event.get('markets', [])
                    for m in raw_markets:
                        if target_market_id and str(m.get('id')) != str(target_market_id):
                            continue
                        m['parent_slug'] = event.get('slug', 'unknown')
                        m['parent_name'] = event.get('title', 'unknown')
                        m['event_markets'] = raw_markets
                        is_valid, reason = validate_market(m, history, mode, exclude_mode)
                        if is_valid:
                            yield m
                        else:
                            print(f"[-] Ignored Target Market: '{m.get('question')}' | Reason: {reason}")
            except Exception as e:
                print(f"[!] Error fetching slug {slug}: {e}")
