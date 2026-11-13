import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
"""Portfolio and history display tables, aggregate variance metrics, and top-of-menu dashboard."""
import math
from datetime import datetime, timezone
import config as cfg
from utils import (
    ensure_list,
    parse_market_date,
    format_time_remaining,
    ansi_link,
    strip_ansi,
    get_history_counts,
    calculate_position_holding,
    parse_outcome_prices,
)
from storage import save_history, get_active_profile, load_history
from model import (
    compute_dynamic_ego, calculate_logistic_metrics, calculate_base_ego, calculate_annualized_yield,
    calculate_dynamic_kelly_action, compute_trajectory_brier, calculate_trade_ego_metrics,
    calculate_trade_weight, restore_snapshot_to_prediction
)

def auto_prune_closed_markets(history, force=False):
    """Silently audits and prunes closed/resolved markets from skipped history in background."""
    now_ts = datetime.now(timezone.utc).timestamp()
    last_ts = history.get("_last_skipped_prune", 0.0)
    if not force and (now_ts - last_ts < 300):
        return 0
    try:
        from manage import prune_closed_skipped_markets
        pruned = prune_closed_skipped_markets(history, verbose=False)
        history["_last_skipped_prune"] = now_ts
        if pruned > 0:
            save_history(history)
        return pruned
    except Exception:
        return 0


def ensure_predicted_market_metadata(history, fetch_clob=True, base_ego=None):
    """Enriches predicted and exited (unresolved) markets with live prices, spread, endDate, and volume from Gamma API in bulk."""
    predicted_dict = history.get("predicted", {})
    exited_dict = history.get("exited", {})
    if not predicted_dict and not exited_dict:
        return
    if base_ego is None:
        base_ego = calculate_base_ego(history, verbose=False)

    m_ids = list({
        str(mid) for mid in list(predicted_dict.keys()) + list(exited_dict.keys())
        if not str(mid).startswith("custom_")
    })
    if not m_ids:
        return

    changed = False

    # Auto-consolidate any fragmented/duplicate predictions for the same market
    for d in (predicted_dict, exited_dict):
        for mid, p_val in list(d.items()):
            preds = ensure_list(p_val)
            if len(preds) > 1:
                wallet_items = [p for p in preds if p.get("source") == "wallet"]
                local_items = [p for p in preds if p.get("source") != "wallet"]
                if wallet_items:
                    canonical = wallet_items[0]
                    for other_w in wallet_items[1:]:
                        canonical["shares"] = round(float(canonical.get("shares", 0.0) or 0.0) + float(other_w.get("shares", 0.0) or 0.0), 4)
                        canonical["cost_basis"] = round(float(canonical.get("cost_basis", 0.0) or 0.0) + float(other_w.get("cost_basis", 0.0) or 0.0), 4)
                    pool = wallet_items + local_items
                    latest_p = sorted(pool, key=lambda x: str(x.get("last_pu_date") or x.get("date", "")), reverse=True)[0]
                    for k in ("pu", "dynamic_ego", "date", "edge", "kelly", "opt_buy_usd", "opt_buy_vwap", "opt_buy_edge", "snapshots", "last_pu_ts", "last_pu_date"):
                        if k in latest_p:
                            canonical[k] = latest_p[k]
                    d[mid] = [canonical]
                    changed = True
                elif local_items:
                    latest = sorted(local_items, key=lambda x: str(x.get("last_pu_date") or x.get("date", "")), reverse=True)[0]
                    d[mid] = [latest]
                    changed = True
                else:
                    d[mid] = [preds[0]]
                    changed = True

    # Resolve cross-bucket overlap: if market is in predicted, remove from exited
    for mid in list(predicted_dict.keys()):
        if mid in exited_dict:
            del exited_dict[mid]
            changed = True

    chunk_size = 40
    for i in range(0, len(m_ids), chunk_size):
        chunk = m_ids[i:i + chunk_size]
        try:
            url = f"{cfg.GAMMA_API}/markets?" + "&".join(f"id={mid}" for mid in chunk)
            resp = cfg.api.get(url, timeout=5).json()
            if isinstance(resp, list):
                resp_map = {str(item.get("id")): item for item in resp if item.get("id")}
                for mid in chunk:
                    if mid not in resp_map:
                        continue
                    m_data = resp_map[mid]
                    prices = parse_outcome_prices(m_data)
                    best_bid = float(m_data.get("bestBid") or 0.0) if m_data.get("bestBid") is not None else None
                    best_ask = float(m_data.get("bestAsk") or 0.0) if m_data.get("bestAsk") is not None else None
                    spread = float(m_data.get("spread") or 0.0) if m_data.get("spread") is not None else None
                    end_d = m_data.get("endDate")
                    vol = float(m_data.get("volume") or 0.0)
                    vol24 = float(m_data.get("volume24hr") or 0.0)
                    closed = bool(m_data.get("closed", False))

                    preds = ensure_list(predicted_dict.get(mid, [])) + ensure_list(exited_dict.get(mid, []))
                    for p in preds:
                        pos = p.get("position", "Yes")
                        is_no = (pos == "No")
                        if is_no:
                            live_p = prices[1] if len(prices) >= 2 else (1.0 - prices[0] if prices else None)
                            live_bid = round(1.0 - best_ask, 4) if best_ask is not None else None
                            live_ask = round(1.0 - best_bid, 4) if best_bid is not None else None
                        else:
                            live_p = prices[0] if prices else None
                            live_bid = best_bid
                            live_ask = best_ask

                        if live_p is not None and live_p != p.get("cur_pm"):
                            p["cur_pm"] = round(live_p, 6)
                            changed = True

                        if live_bid is not None and p.get("bestBid") != live_bid:
                            p["bestBid"] = live_bid
                            changed = True
                        if live_ask is not None and p.get("bestAsk") != live_ask:
                            p["bestAsk"] = live_ask
                            changed = True
                        if spread is not None and p.get("spread") != spread:
                            p["spread"] = spread
                            changed = True
                        if end_d and p.get("endDate") != end_d:
                            p["endDate"] = end_d
                            changed = True
                        if p.get("volume") != vol:
                            p["volume"] = vol
                            changed = True
                        if p.get("volume24hr") != vol24:
                            p["volume24hr"] = vol24
                            changed = True
                        clob_toks = m_data.get("clobTokenIds")
                        if clob_toks and p.get("clobTokenIds") != clob_toks:
                            p["clobTokenIds"] = clob_toks
                            changed = True
                        if closed and not p.get("closed"):
                            p["closed"] = True
                            changed = True
                        if "fee_rate" not in p or p.get("fee_rate") is None:
                            f_rate_m = (m_data.get("feeSchedule") or {}).get("rate")
                            if f_rate_m is None:
                                f_val = float(m_data.get("makerBaseFee") or 0.0)
                                f_rate_m = f_val / 1e18 if f_val > 1 else f_val
                            p["fee_rate"] = float(f_rate_m or 0.0)
                            changed = True
        except Exception:
            pass

    # Fetch live CLOB L2 books in parallel & calculate exact VWAP & optimal immediate market-buy for all active markets
    if fetch_clob and (predicted_dict or exited_dict):
        import concurrent.futures
        from model import calculate_vwap, calculate_optimal_market_buy, calculate_logistic_metrics, calculate_dynamic_kelly_action
        from market import get_market_clob_tokens

        ws_b = history.get("wallet_summary", {})
        live_bankroll = float(ws_b.get("bankroll") or cfg.BANKROLL) if isinstance(ws_b, dict) else cfg.BANKROLL
        min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))

        clob_tasks = []
        for bucket_d in (predicted_dict, exited_dict):
            for mid, preds_list in bucket_d.items():
                for p in ensure_list(preds_list):
                    clob_ids = p.get("clobTokenIds")
                    if not clob_ids:
                        continue
                    yes_tok, no_tok = get_market_clob_tokens({"clobTokenIds": clob_ids})
                    pos = p.get("position", "Yes")
                    target_token = no_tok if pos == "No" else yes_tok
                    if target_token:
                        clob_tasks.append((p, target_token))

        def _fetch_single_clob(item_tuple):
            p, target_token = item_tuple
            try:
                r = cfg.api.get(f"https://clob.polymarket.com/book?token_id={target_token}", timeout=4)
                if r.status_code == 200:
                    return p, r.json()
            except Exception:
                pass
            return p, None

        if clob_tasks:
            max_w = min(8, max(1, len(clob_tasks)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                clob_results = list(executor.map(_fetch_single_clob, clob_tasks))

            for p, book in clob_results:
                if not isinstance(book, dict):
                    continue
                try:
                    asks = book.get("asks", [])
                    bids = book.get("bids", [])
                    valid_asks = sorted([(float(a["price"]), float(a["size"])) for a in asks if float(a.get("price", 0)) > 0], key=lambda x: x[0])
                    valid_bids = sorted([(float(b["price"]), float(b["size"])) for b in bids if float(b.get("price", 0)) > 0], key=lambda x: x[0], reverse=True)
                    if valid_asks:
                        best_ask = valid_asks[0][0]
                        p["bestAsk"] = best_ask
                    if valid_bids:
                        best_bid = valid_bids[0][0]
                        p["bestBid"] = best_bid
                    if valid_asks and valid_bids:
                        p["spread"] = round(max(0.0, p["bestAsk"] - p["bestBid"]), 4)
                        midpoint = round((p["bestAsk"] + p["bestBid"]) / 2.0, 4)
                        p["cur_pm"] = midpoint

                    # Compute belief hurdle & optimal market buy capacity
                    pu_val = float(p.get("pu", 0.5))
                    pm_val = float(p.get("pm", p.get("cur_pm", 0.5)))
                    ego_val = compute_dynamic_ego(base_ego, float(p.get("u_spread", 0.0) or 0.0))
                    p["dynamic_ego"] = ego_val
                    k_val = float(p.get("entry_kelly") or p.get("effective_kelly", 0.0) or p.get("kelly", 0.0) or 0.0)
                    cost_basis = float(p.get("cost_basis", 0.0) or 0.0)
                    shares_val = float(p.get("shares", 0.0) or 0.0)
                    is_held = (not p.get("exited")) and (cost_basis > 0 or shares_val > 0 or p.get("source") == "wallet")

                    eval_px = p.get("cur_pm") if is_held else (p.get("bestAsk") or p.get("cur_pm") or pm_val)
                    fee_r = float(p.get("fee_rate", 0.0) or 0.0)
                    lm_tmp = calculate_logistic_metrics(pu_val, pm_val, ego_val, cur_pm=eval_px, fee_rate=(0.0 if is_held else fee_r))
                    p_norm = lm_tmp["win_prob_bal"]

                    if is_held:
                        cur_px_val = float(p.get("cur_pm", pm_val) or pm_val)
                        held_val, held_frac, _ = calculate_position_holding(p, cur_px_val, live_bankroll)
                        half_sp = max(0.0005, float(p.get("spread", 0.01) or 0.01) / 2.0)
                        dyn_act = calculate_dynamic_kelly_action(
                            win_prob_bal=p_norm,
                            cur_p=cur_px_val,
                            held_frac=held_frac,
                            bankroll=live_bankroll,
                            fee_rate=fee_r,
                            half_spread=half_sp,
                            entry_kelly=max(lm_tmp["eff_kelly"], k_val),
                            best_bid=p.get("bestBid"),
                            best_ask=p.get("bestAsk")
                        )
                        if dyn_act["signal_code"] == "ADD" and dyn_act["delta_usd"] > 0 and valid_asks:
                            opt_usd, opt_vwap, opt_edge = calculate_optimal_market_buy(
                                valid_asks, dyn_act["delta_usd"], p_norm=p_norm, fee_rate=fee_r, min_edge=0.001
                            )
                            p["opt_buy_usd"] = opt_usd
                            p["opt_buy_vwap"] = opt_vwap
                            p["opt_buy_edge"] = opt_edge
                            p["vwap"] = opt_vwap if opt_usd > 0 else valid_asks[0][0]
                            p["slippage"] = round(max(0.0, p["vwap"] - (p.get("bestAsk") or p["vwap"])), 4)
                            p["vwap_edge"] = opt_edge
                        else:
                            p["opt_buy_usd"] = None
                            if valid_asks:
                                vwap_top = valid_asks[0][0]
                                p["vwap"] = vwap_top
                                p["slippage"] = 0.0
                                vwap_fee = vwap_top + (fee_r * vwap_top * (1.0 - vwap_top))
                                p["vwap_edge"] = round(p_norm - vwap_fee, 4)
                        changed = True
                    else:
                        eval_target = (lm_tmp["eff_kelly"] * live_bankroll) if lm_tmp["eff_kelly"] > 0 else (k_val * live_bankroll)
                        if valid_asks and eval_target > 0:
                            opt_usd, opt_vwap, opt_edge = calculate_optimal_market_buy(
                                valid_asks, eval_target, p_norm=p_norm, fee_rate=fee_r, min_edge=min_edge_th
                            )
                            p["opt_buy_usd"] = opt_usd
                            p["opt_buy_vwap"] = opt_vwap
                            p["opt_buy_edge"] = opt_edge
                            p["vwap"] = opt_vwap if opt_usd > 0 else (valid_asks[0][0] if valid_asks else 0.0)
                            p["slippage"] = round(max(0.0, p["vwap"] - (p.get("bestAsk") or p["vwap"])), 4)
                            p["vwap_edge"] = opt_edge
                            changed = True
                except Exception:
                    pass

    if changed:
        try:
            save_history(history)
        except Exception:
            pass

def format_compact_question(q, width=34):
    """Clean, simple question title formatting for table display with standard truncation."""
    if not q:
        return "unknown"
    s = str(q).strip()
    return (s[:width - 3].rstrip() + "...") if len(s) > width else s

def _print_single_market_details(r):
    """Prints a clean, concise market card with resolution rules and execution status."""
    mode_val = r.get("Mode", "active")
    print("\n" + "=" * 72)
    print(f" [MARKET DETAILS: #{r.get('Market ID', 'unknown')}]".center(72))
    print("-" * 72)
    print(f"Question     : {r.get('raw_question') or r.get('Question')}")
    print(f"Event URL    : {r.get('Web Link')}")
    print(f"Market ID    : {r.get('Market ID')}")
    print(f"Side / Mode  : {r.get('Side')} | Mode: {mode_val.upper()} | Source: {r.get('Source', '-')}")
    if mode_val != "skipped":
        print(f"Action/Signal: {r.get('Regime')}")
        if r.get("HeldVal", 0) > 0:
            print(f"Held Equity  : ${r.get('HeldVal', 0):,.2f} ({r.get('HeldFrac', 0)*100:.1f}% MTM)")
        if r.get("PnL") is not None and mode_val in ("resolved", "exited"):
            out_s = r.get("Outcome")
            res_tag = " | Result: WON" if out_s == "1.0" else (" | Result: LOST" if out_s == "0.0" else (" | Result: HALF" if out_s == "0.5" else ""))
            print(f"Realized PnL : ${float(r['PnL']):+,.2f}{res_tag}")
        print(f"Kelly Fraction: {r.get('Kelly', 0)*100:.2f}% | Eff-Kelly: {r.get('EffKelly', 0)*100:.2f}%")
        print(f"Model Edge   : {r.get('Edge', 0)*100:+.2f}% | p_norm: {r.get('p_norm', 0)*100:.1f}%")
        if r.get("bs_u") is not None and r.get("bs_m") is not None:
            print(f"Brier Calib. : BS(User) {r['bs_u']:.4f} | BS(Mkt) {r['bs_m']:.4f} | Net Adv: {r.get('adv', 0.0):+.4f}")
        fee_r = float(r.get("fee_rate", 0.0) or 0.0)
        if fee_r > 0:
            print(f"Fee Rate     : {fee_r*100:.2f}% (CLOB Fee Schedule)")
        entry_p = r.get("entry_price") or r.get("pm", 0)
        ref_pm = r.get("pm", 0)
        px_lbl = "Exit Price" if mode_val == "exited" else "Current Price"
        if entry_p and ref_pm and abs(float(entry_p) - float(ref_pm)) > 0.001:
            print(f"{px_lbl:<13}: {r.get('cur_pm', 0)*100:.1f}% | Entry Price: {float(entry_p)*100:.1f}% | Market Ref (pm): {float(ref_pm)*100:.1f}% | Belief (pu): {r.get('pu', 0)*100:.1f}%")
        else:
            print(f"{px_lbl:<13}: {r.get('cur_pm', 0)*100:.1f}% | Entry Price: {float(entry_p)*100:.1f}% | Belief (pu): {r.get('pu', 0)*100:.1f}%")
        if r.get("DaysUntil") is not None:
            print(f"Time Left    : {r.get('DaysUntil', 0):.1f} days remaining")
        if r.get("CEY") is not None or r.get("Volume") is not None:
            print(f"Scale Metrics: CEY: {r.get('CEY', 0.0):,.0f} | EII: ${r.get('EII', 0.0):,.0f} | Vol: ${r.get('Volume', 0.0):,.0f}")

    vwap_v = r.get("vwap")
    best_ask_v = r.get("bestAsk")
    best_bid_v = r.get("bestBid")
    if best_ask_v is not None or best_bid_v is not None or vwap_v is not None:
        parts = []
        if best_bid_v is not None:
            parts.append(f"Bid: {best_bid_v*100:.1f}%")
        if best_ask_v is not None:
            parts.append(f"Ask: {best_ask_v*100:.1f}%")
        if vwap_v is not None:
            slip_v = r.get("slippage", 0.0)
            parts.append(f"VWAP: {vwap_v*100:.2f}% ({slip_v*100:+.2f}% slip)")
            if r.get("vwap_edge") is not None:
                parts.append(f"VWAP Edge: {r['vwap_edge']*100:+.2f}%")
        print(f"Live Execution: {' | '.join(parts)}")

    opt_usd = r.get("opt_buy_usd")
    opt_vwap = r.get("opt_buy_vwap")
    opt_edge = r.get("opt_buy_edge")
    min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))
    reg_str = str(r.get("Regime", ""))
    if opt_usd is not None and opt_usd > 0 and opt_vwap is not None and any(k in reg_str for k in ("BUY", "ADD")):
        print(f"Optimal Order: Market-buy ${opt_usd:,.2f} @ {opt_vwap*100:.2f}% mean VWAP (Edge: {opt_edge*100:+.2f}%)")
    elif opt_usd == 0.0 and r.get("HeldVal", 0) == 0:
        print(f"Optimal Order: $0 market-buy (Orderbook asks exceed hurdle; no edge >= {min_edge_th*100:.1f}%)")


    snaps = r.get("snapshots")
    if isinstance(snaps, list) and len(snaps) > 0:
        print(f"\nPrediction Trajectory ({len(snaps)} snapshot{'s' if len(snaps) != 1 else ''}):")
        for s_i, s in enumerate(snaps, 1):
            s_tag = "Entry" if s_i == 1 else f"Rev #{s_i - 1}"
            s_pu = float(s.get("pu", 0.0)) * 100.0
            s_pm = float(s.get("pm", 0.0)) * 100.0
            s_k = float(s.get("kelly", 0.0) or 0.0) * 100.0
            s_ego = float(s.get("dynamic_ego", 0.5) or 0.5)
            s_date = str(s.get("date", ""))[:16]
            print(f"  [{s_i}] {s_tag:<6} ({s_date}): pu={s_pu:.1f}% | pm={s_pm:.1f}% | Kelly={s_k:.2f}% | Ego@Time={s_ego:.3f}")

    desc = r.get("description")
    res_src = r.get("resolutionSource")
    m_id = str(r.get("Market ID", ""))
    if not desc and m_id and not m_id.startswith("custom_"):
        try:
            m_resp = cfg.api.get(f"{cfg.GAMMA_API}/markets/{m_id}", timeout=4).json()
            if isinstance(m_resp, dict):
                desc = m_resp.get("description")
                if not res_src:
                    res_src = m_resp.get("resolutionSource")
        except Exception:
            pass

    if res_src:
        print(f"Resolution Src: {res_src}")
    if desc:
        print("\nRules & Resolution Condition:")
        print(desc.strip())

    print("=" * 72)


def _format_source_tag(pred, history_alias=None):
    """Formats 6-8 char source indicator for tables (e.g. 'Main', '0x..9fa7', 'LOCAL')."""
    if pred.get("exclude_from_ego"):
        return "[EXCL]"
    w_tag = pred.get("wallet")
    src = pred.get("source", "local")
    if src == "wallet" and history_alias:
        return history_alias[:8]
    elif w_tag and len(w_tag) >= 6:
        return f"0x..{w_tag[-4:]}"
    return src.upper()


def merge_market_tranches(preds):
    """Merges multiple tranches for a single market into a non-mutating display record."""
    preds_list = ensure_list(preds)
    if not preds_list:
        return None
    if len(preds_list) == 1:
        return preds_list[0]

    wallet_items = [p for p in preds_list if p.get("source") == "wallet"]
    local_items = [p for p in preds_list if p.get("source") != "wallet"]
    if wallet_items:
        primary = dict(wallet_items[0])
        for other_w in wallet_items[1:]:
            primary["shares"] = round(float(primary.get("shares", 0.0) or 0.0) + float(other_w.get("shares", 0.0) or 0.0), 4)
            primary["cost_basis"] = round(float(primary.get("cost_basis", 0.0) or 0.0) + float(other_w.get("cost_basis", 0.0) or 0.0), 4)
        pool = wallet_items + local_items
        latest_p = sorted(pool, key=lambda x: str(x.get("last_pu_date") or x.get("date", "")), reverse=True)[0]
        for k in ("pu", "dynamic_ego", "date", "edge", "kelly", "opt_buy_usd", "opt_buy_vwap", "opt_buy_edge", "snapshots", "last_pu_ts", "last_pu_date"):
            if k in latest_p:
                primary[k] = latest_p[k]
        return primary
    return sorted(local_items, key=lambda x: str(x.get("last_pu_date") or x.get("date", "")), reverse=True)[0]


def iter_unified_active_entries(history, include_exited=True):
    """Yields deduplicated (market_id, merged_primary_entry) across predicted and optionally exited."""
    seen_mids = set()
    for m_id, val in history.get("predicted", {}).items():
        merged = merge_market_tranches(val)
        if merged is not None:
            seen_mids.add(m_id)
            yield m_id, merged
    if include_exited:
        for m_id, val in history.get("exited", {}).items():
            if m_id in seen_mids:
                continue
            merged = merge_market_tranches(val)
            if merged is not None:
                seen_mids.add(m_id)
                yield m_id, merged


def classify_active_signal(is_held, dyn_action, eff_kelly, live_edge, min_edge_th=None, opt_buy_usd=None, vwap_edge=None):
    """Canonical signal classification shared by dashboard, stats, and review table."""
    if min_edge_th is None:
        min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))
    effective_min_edge = max(float(min_edge_th), 0.01)

    if is_held:
        sig = dyn_action.get("signal_code", "HOLD")
        if sig == "ADD" and opt_buy_usd == 0.0:
            return "HOLD"
        return sig

    if float(eff_kelly) > 0.005 and float(live_edge) >= effective_min_edge:
        if opt_buy_usd == 0.0 or (vwap_edge is not None and (vwap_edge <= 0 or vwap_edge < effective_min_edge)):
            return "WATCH"
        return "BUY"
    return "WATCH"

def classify_luck_z(z):
    """Classifies a binomial Luck Z-score into a concise variance regime label."""
    if z is None:
        return "N/A"
    if z >= 2.0:
        return "Extremely Lucky"
    if z >= 1.25:
        return "Running Hot"
    if z <= -2.0:
        return "Extremely Unlucky"
    if z <= -1.25:
        return "Running Cold"
    return "Normal Variance"


def _compute_resolved_reflection(res_items):
    """Computes resolved calibration bias (actual - expected) and unweighted/Kelly-weighted Luck Z-scores."""
    if not res_items:
        return {
            "res_count": 0,
            "res_actual_win_rate": None,
            "res_exp_win_bal": None,
            "res_exp_win_u": None,
            "res_calib_bias_bal": None,
            "res_calib_bias_u": None,
            "res_luck_z": None,
            "res_luck_z_weighted": None,
        }
    n = len(res_items)
    actual_wins = sum(float(x["outcome"]) for x in res_items)
    exp_wins_bal = sum(x["win_prob_bal"] for x in res_items)
    exp_wins_u = sum(x["win_prob_u"] for x in res_items)
    actual_rate = actual_wins / n
    exp_rate_bal = exp_wins_bal / n
    exp_rate_u = exp_wins_u / n

    var_sum = sum(x["var_i"] for x in res_items)
    luck_z = ((actual_wins - exp_wins_bal) / math.sqrt(var_sum)) if var_sum > 1e-9 else 0.0

    k_sum = sum(x["kelly"] for x in res_items)
    w_list = [x["kelly"] / k_sum for x in res_items] if k_sum > 0 else [1.0 / n] * n
    w_actual = sum(w * float(x["outcome"]) for w, x in zip(w_list, res_items))
    w_exp_bal = sum(w * x["win_prob_bal"] for w, x in zip(w_list, res_items))
    w_var = sum((w ** 2) * x["var_i"] for w, x in zip(w_list, res_items))
    luck_z_w = ((w_actual - w_exp_bal) / math.sqrt(w_var)) if w_var > 1e-9 else 0.0

    return {
        "res_count": n,
        "res_actual_win_rate": actual_rate,
        "res_exp_win_bal": exp_rate_bal,
        "res_exp_win_u": exp_rate_u,
        "res_calib_bias_bal": actual_rate - exp_rate_bal,
        "res_calib_bias_u": actual_rate - exp_rate_u,
        "res_luck_z": luck_z,
        "res_luck_z_weighted": luck_z_w,
    }


def calculate_prediction_stats(history, sub_mode="predicted", base_ego=None, refresh_meta=True):
    """Calculates weighted portfolio stats: balance percentage, effective N, expected win rate, variance."""
    # Auto-prune closed markets from skipped history in background
    auto_prune_closed_markets(history, force=(sub_mode == "skipped"))

    if refresh_meta and sub_mode in ["predicted", "all"]:
        ensure_predicted_market_metadata(history)
    if base_ego is None:
        base_ego = calculate_base_ego(history, verbose=False)

    seen_mids = set()
    preds_list = []
    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        for mid, merged_p in iter_unified_active_entries(
            history, include_exited=(sub_mode in ["predicted", "predicted_skipped"])
        ):
            seen_mids.add(mid)
            preds_list.append((merged_p, "active"))
    if sub_mode in ["exited", "all"]:
        for mid, m_ex in history.get("exited", {}).items():
            if sub_mode == "all" and mid in seen_mids:
                continue
            for p in ensure_list(m_ex):
                preds_list.append((p, "exited"))
    if sub_mode in ["resolved", "all"]:
        for m_res in history.get("resolved", {}).values():
            for p in ensure_list(m_res):
                if p.get("outcome") is not None and p.get("outcome") in (0.0, 1.0, 0.5):
                    preds_list.append((p, "resolved"))

    if not preds_list:
        return None

    ws_b = history.get("wallet_summary", {})
    active_bankroll = cfg.BANKROLL
    if isinstance(ws_b, dict) and ws_b.get("bankroll") and float(ws_b.get("bankroll")) > 0:
        active_bankroll = float(ws_b.get("bankroll"))

    active_pid_stats = history.get("profile_id") or get_active_profile()
    is_live_wallet_stats = bool(active_pid_stats and str(active_pid_stats).startswith("0x"))

    items = []
    for p, state in preds_list:
        pos_side = p.get("position", "Yes")
        pu = float(p.get("pu", 0.5))
        pm = float(p.get("pm", 0.5))
        is_exited = (state == "exited") or (bool(p.get("exited")) and state != "active")
        is_res = (state == "resolved")
        cur_pm = p.get("exit_price") if (is_exited and p.get("exit_price") is not None) else p.get("cur_pm")
        kelly = max(0.0, float(p.get("entry_kelly") or p.get("effective_kelly", 0.0) or p.get("kelly", 0.0) or 0.0))
        ego = float(p.get("dynamic_ego", base_ego))
        outcome = p.get("outcome")
        f_rate = float(p.get("fee_rate", 0.0) or 0.0)

        if state == "active":
            ego = compute_dynamic_ego(base_ego, float(p.get("u_spread", 0.0) or 0.0))
            is_w_p = (p.get("source", "local") == "wallet")
            cur_p_v = float(cur_pm if cur_pm is not None else pm)
            if is_live_wallet_stats and not is_w_p:
                lv_i, h_k_i, _ = 0.0, 0.0, 0.0
            else:
                lv_i, h_k_i, _ = calculate_position_holding(p, cur_p_v, active_bankroll)
            is_held_i = (h_k_i > 0 or lv_i > 0)
            eval_px_i = cur_pm if is_held_i else (float(p.get("bestAsk") or cur_pm or pm))
            lm = calculate_logistic_metrics(pu, pm, ego, held_kelly=None, cur_pm=eval_px_i, fee_rate=(0.0 if is_held_i else f_rate))
            p_edge = float(lm["live_active_edge"])
            if is_held_i:
                sp_i = float(p.get("spread", 0.01) or 0.01)
                dyn_i = calculate_dynamic_kelly_action(
                    win_prob_bal=lm["win_prob_bal"],
                    cur_p=cur_p_v,
                    held_frac=h_k_i,
                    bankroll=active_bankroll,
                    fee_rate=f_rate,
                    half_spread=max(0.0005, sp_i / 2.0),
                    entry_kelly=max(lm["eff_kelly"], kelly),
                    best_bid=p.get("bestBid"),
                    best_ask=p.get("bestAsk")
                )
                eff_k_stat = dyn_i["f_star_eff"]
                kelly = eff_k_stat
            else:
                dyn_i = {}
                eff_k_stat = lm["eff_kelly"]
            active_sig_i = classify_active_signal(
                is_held_i,
                dyn_i,
                eff_k_stat,
                p_edge,
                opt_buy_usd=p.get("opt_buy_usd"),
                vwap_edge=p.get("vwap_edge"),
            )
        else:
            kelly = calculate_trade_weight({**p, "exclude_from_ego": False}, bankroll=active_bankroll)
            lm = calculate_logistic_metrics(pu, pm, ego, held_kelly=kelly, cur_pm=cur_pm, fee_rate=f_rate)
            p_edge = float(lm["active_edge"])
            eff_k_stat = lm["eff_kelly"]
            active_sig_i = None

        end_d = p.get("endDate") or p.get("end_date")
        _, sec_left, d_until = parse_market_date(end_d)
        if d_until is not None:
            d_until = max(0.05, d_until) if sec_left and sec_left > 0 else 0.05

        d_edge = (p_edge / d_until) if (d_until and d_until > 0) else None
        win_prob_bal = lm["win_prob_bal"]
        var_i = win_prob_bal * (1.0 - win_prob_bal)
        alloc_i = (kelly * active_bankroll) if state == "active" else 0.0
        eff_alloc_i = (eff_k_stat * active_bankroll) if state == "active" else 0.0

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
            "eff_kelly": eff_k_stat,
            "active_edge": lm["active_edge"],
            "live_active_edge": lm["live_active_edge"],
            "eff_m": lm["eff_m"],
            "eff_live": lm["eff_live"],
            "eff_norm": lm["eff_norm"],
            "regime": "EXITED" if is_exited else lm["regime"],
            "is_exited": is_exited,
            "realized_pnl": float(p.get("realized_pnl", 0.0) or 0.0),
            "realized_pnl_raw": p.get("realized_pnl"),
            "alloc": alloc_i,
            "eff_alloc": eff_alloc_i,
            "win_prob_bal": win_prob_bal,
            "win_prob_u": lm["win_prob_u"],
            "win_prob_m": lm["win_prob_m"],
            "var_i": var_i,
            "source": p.get("source", "local"),
            "is_resolved": is_res,
            "outcome": float(outcome) if (is_res and outcome is not None) else None,
            "is_held": is_held_i if state == "active" else False,
            "held_val": lv_i if (state == "active" and is_held_i) else 0.0,
            "held_frac": h_k_i if (state == "active" and is_held_i) else 0.0,
            "active_sig": active_sig_i if state == "active" else None,
            "daily_edge": d_edge,
            "days_until": d_until,
            "apy": p.get("apy")
        })

    w_key = "eff_kelly" if sub_mode == "predicted" else "kelly"
    sized_items = [x for x in items if x[w_key] > 0]
    total_kelly = sum(x["kelly"] for x in items)
    total_eff_kelly = sum(x["eff_kelly"] for x in items)
    total_alloc = sum(x["alloc"] for x in items)
    total_eff_alloc = sum(x["eff_alloc"] for x in items)
    total_pnl = sum(x["realized_pnl"] for x in items if x["state"] in ("resolved", "exited"))
    local_count = sum(1 for x in items if x["source"] != "wallet")
    wallet_count = sum(1 for x in items if x["source"] == "wallet")

    total_w_val = sum(x[w_key] for x in sized_items)
    if total_w_val > 0 and sized_items:
        weights = [x[w_key] / total_w_val for x in sized_items]
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
    avg_active_edge = sum(w * (x["live_active_edge"] if x["state"] == "active" else x["active_edge"]) for w, x in zip(weights, used_items))

    norm_variance = sum((w ** 2) * x["var_i"] for w, x in zip(weights, used_items))
    norm_std_dev = math.sqrt(norm_variance)
    sum_w_sq = sum(w ** 2 for w in weights)
    effective_n = (1.0 / sum_w_sq) if sum_w_sq > 0 else 0.0

    held_items = [x for x in items if x.get("is_held") and x.get("held_val", 0) > 0]
    total_held_val = sum(x["held_val"] for x in held_items)
    if total_held_val > 0 and held_items:
        weights_held = [x["held_val"] / total_held_val for x in held_items]
        sum_w_held_sq = sum(w ** 2 for w in weights_held)
        effective_n_held = (1.0 / sum_w_held_sq) if sum_w_held_sq > 0 else 0.0
        held_bal_pct = (effective_n_held / len(held_items) * 100.0) if len(held_items) > 0 else 0.0
    else:
        effective_n_held = 0.0
        held_bal_pct = 0.0

    held_count = len(held_items)
    buy_count = sum(1 for x in items if (not x.get("is_held")) and x.get("active_sig") == "BUY")
    watch_count = len(items) - held_count - buy_count

    valid_days = [x["days_until"] for x in items if x.get("days_until") is not None]
    valid_apys = []
    for x in items:
        try:
            if x.get("apy") is not None and -1.0 <= float(x["apy"]) <= 100.0:
                valid_apys.append(float(x["apy"]))
        except Exception:
            pass
    avg_daily_edge = sum(w * x["daily_edge"] for w, x in zip(weights, used_items) if x.get("daily_edge") is not None) if used_items else None
    avg_days = sum(valid_days) / len(valid_days) if valid_days else None
    avg_apy = sum(valid_apys) / len(valid_apys) if valid_apys else None

    res_items = [
        x for x in items
        if x.get("is_resolved") and x.get("outcome") is not None
        and not (is_live_wallet_stats and x.get("source") != "wallet" and x.get("realized_pnl_raw") is None)
    ]
    reflection = _compute_resolved_reflection(res_items)

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
        "bankroll": active_bankroll,
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
        "held_count": held_count,
        "total_held_val": total_held_val,
        "effective_n_held": effective_n_held,
        "held_bal_pct": held_bal_pct,
        "buy_count": buy_count,
        "watch_count": watch_count,
        "avg_daily_edge": avg_daily_edge,
        "avg_days": avg_days,
        "avg_apy": avg_apy,
        **reflection,
        "items": items
    }

def print_prediction_stats(history, sub_mode="predicted", base_ego=None, refresh_meta=True):
    """Prints balanced portfolio summary block."""
    stats = calculate_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego, refresh_meta=refresh_meta)
    if not stats:
        return
    n_denom = stats['active_count'] if stats['active_count'] > 0 else stats['total_count']
    bal_pct = (stats['effective_n'] / n_denom * 100.0) if n_denom > 0 else 0.0
    status_lbl = {"predicted": "active", "exited": "exited", "resolved": "resolved", "all": "total"}.get(sub_mode, "active")
    sized_note = f" ({stats['active_count']} sized)" if stats['active_count'] != stats['total_count'] else ""

    print(f"--- PORTFOLIO & PREDICTION STATS [{status_lbl.upper()}] ---")
    if sub_mode == "predicted":
        src_note = f" ({stats['local_count']} LOCAL, {stats['wallet_count']} WALLET)"
        print(f"Predictions      : {stats['total_count']} active ({stats['held_count']} held, {stats['buy_count']} buy, {stats['watch_count']} watch){src_note}")
        if stats['held_count'] > 0:
            print(f"Effective N      : Held: {stats['effective_n_held']:.2f} / {stats['held_count']} equal-sized bets ({stats['held_bal_pct']:.1f}% size-balanced)")
        else:
            print(f"Effective N      : -")
    else:
        print(f"Predictions      : {stats['total_count']} {status_lbl}{sized_note} ({stats['local_count']} LOCAL, {stats['wallet_count']} WALLET)")
        print(f"Effective N      : {stats['effective_n']:.2f} / {n_denom} equal-sized bets ({bal_pct:.1f}% size-balanced)")
    active_pid = history.get("profile_id") or get_active_profile()
    is_wallet = bool(active_pid and active_pid.startswith("0x"))
    w_sum = history.get("wallet_summary")
    if is_wallet:
        w_addr = history.get("wallet_address") or active_pid
        w_label = ansi_link(f"https://polymarket.com/profile/{w_addr}", "Wallet Bankroll") if w_addr else "Wallet Bankroll"
        if isinstance(w_sum, dict) and "cash" in w_sum and "portfolio" in w_sum:
            print(f"{w_label}  : ${stats['bankroll']:,.2f} (Cash: ${w_sum['cash']:,.2f} | Open Positions: ${w_sum['portfolio']:,.2f})")
    else:
        print(f"Paper Bankroll   : ${stats['bankroll']:,.2f} (Sandbox Capital)")

    if sub_mode == "predicted":
        print(f"Model Exposure   : ${stats['total_eff_alloc']:,.2f} / ${stats['bankroll']:,.2f} ({stats['total_eff_kelly']*100:.1f}% Eff-Kelly)")
        if stats.get("avg_daily_edge") is not None:
            vel_s = f"{stats['avg_daily_edge']*100:+.2f}%/d"
            h_s = f" | Avg Horizon: {stats['avg_days']:.1f}d" if stats.get("avg_days") else ""
            print(f"Portfolio Velocity: {vel_s}{h_s}")
    else:
        avg_k = stats['total_kelly'] / max(1, stats['active_count'])
        wins, losses = 0, 0
        for x in stats["items"]:
            if x.get("state") not in ("resolved", "exited"):
                continue
            if is_wallet and x.get("source") != "wallet" and x.get("realized_pnl_raw") is None:
                continue
            out = x.get("outcome")
            pnl = float(x.get("realized_pnl", 0.0) or 0.0)
            if out == 1.0:
                wins += 1
            elif out == 0.0:
                losses += 1
            elif out is None:
                if pnl > 0.0001:
                    wins += 1
                elif pnl < -0.0001:
                    losses += 1
        wl_str = f" | Record: {wins}W - {losses}L" if sub_mode in ("resolved", "exited", "all") else ""
        print(f"Realized PnL     : ${stats['total_pnl']:+,.2f}{wl_str} | Avg Sizing: {avg_k*100:.2f}% Kelly")

    print(f"Expected Win Rate: {stats['exp_win_bal']*100:.2f}% Balanced (User: {stats['exp_win_u']*100:.2f}% | Market: {stats['exp_win_m']*100:.2f}%)")
    if stats.get("res_count", 0) > 0:
        cb_u = stats["res_calib_bias_u"] * 100.0
        cb_b = stats["res_calib_bias_bal"] * 100.0
        lz = stats["res_luck_z"]
        lzw = stats["res_luck_z_weighted"]
        print(
            f"Calibration Bias : Actual {stats['res_actual_win_rate']*100:.1f}% "
            f"| vs User ({stats['res_exp_win_u']*100:.1f}%): {cb_u:+.1f}% "
            f"| vs Pooled ({stats['res_exp_win_bal']*100:.1f}%): {cb_b:+.1f}%"
        )
        print(
            f"Luck Z-Score     : {lz:+.2f} sigma [{classify_luck_z(lz)}] "
            f"| Kelly-Wt: {lzw:+.2f} sigma [{classify_luck_z(lzw)}]"
        )
    if sub_mode == "resolved":
        print(f"Logistic Eff.    : Entry {stats['avg_eff_m']*100:.1f}% | Pooled (p_norm) {stats['avg_eff_norm']*100:.1f}% | Avg Entry Edge: {stats['avg_active_edge']*100:+.2f}%\n")
    else:
        print(f"Logistic Eff.    : Entry {stats['avg_eff_m']*100:.1f}% | Live {stats['avg_eff_live']*100:.1f}% | Pooled (p_norm) {stats['avg_eff_norm']*100:.1f}% | Avg Edge: {stats['avg_active_edge']*100:+.2f}%\n")

TABLE_CONFIGS = {
    "ego_impact": {
        "headers": ["#", "Question", "Side", "Outcome", "Kelly", "BS (User)", "BS (Mkt)", "Net Adv", "PnL", "Date", "Src"],
        "widths": [3, 34, 4, 7, 7, 9, 9, 9, 10, 10, 8],
        "aligns": ["right", "left", "left", "left", "right", "right", "right", "right", "right", "left", "left"],
        "q_idx": 1
    },
    "predicted": {
        "headers": ["#", "Question", "Action / Signal", "Held Pos", "Kelly", "Edge", "Vel/d", "Side", "Live", "p_norm", "Date", "Src"],
        "widths": [3, 34, 26, 12, 7, 7, 9, 4, 6, 6, 10, 6],
        "aligns": ["right", "left", "left", "right", "right", "right", "right", "left", "right", "right", "left", "left"],
        "q_idx": 1
    },
    "exited": {
        "headers": ["#", "Question", "Realized PnL", "Kelly", "Edge", "Side", "Entry", "Exit Px", "pu", "p_norm", "Date", "Src"],
        "widths": [3, 34, 13, 7, 7, 4, 6, 7, 6, 6, 10, 8],
        "aligns": ["right", "left", "right", "right", "right", "left", "right", "right", "right", "right", "left", "left"],
        "q_idx": 1
    },
    "resolved": {
        "headers": ["#", "Question", "Result", "Realized PnL", "Kelly", "Edge", "Side", "Entry", "pu", "p_norm", "Date", "Src"],
        "widths": [3, 34, 6, 13, 7, 7, 4, 6, 6, 6, 10, 8],
        "aligns": ["right", "left", "left", "right", "right", "right", "left", "right", "right", "right", "left", "left"],
        "q_idx": 1
    },
    "skipped": {
        "headers": ["#", "Question", "Date"],
        "widths": [3, 78, 16],
        "aligns": ["right", "left", "left"],
        "q_idx": 1
    },
    "all": {
        "headers": ["#", "Question", "State", "Status / Signal", "PnL / Pos", "Kelly", "Edge", "Side", "Live/Exit", "p_norm", "Date"],
        "widths": [3, 34, 8, 24, 13, 7, 7, 4, 9, 6, 10],
        "aligns": ["right", "left", "left", "left", "right", "right", "right", "left", "right", "right", "left"],
        "q_idx": 1
    }
}

def print_review_table(history, sub_mode, expanded=False, sort_by=None, refresh_meta=True):
    """Renders formatted history table with ANSI hyperlinks, sorting, and action bar."""
    # Auto-prune closed markets from skipped history in background
    auto_prune_closed_markets(history, force=(sub_mode == "skipped"))

    if refresh_meta and sub_mode in ["predicted", "all"]:
        ensure_predicted_market_metadata(history)
    base_ego = calculate_base_ego(history, verbose=False)

    ws_b = history.get("wallet_summary", {})
    live_bankroll = cfg.BANKROLL
    if isinstance(ws_b, dict) and ws_b.get("bankroll") and float(ws_b.get("bankroll")) > 0:
        live_bankroll = float(ws_b.get("bankroll"))

    rows = []
    alias_tag = history.get("alias")

    # 1. Active / Predicted Rows (Unifying Open Positions, Watched Predictions & Exited Unresolved Markets)
    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        profile_id = history.get("profile_id") or get_active_profile()
        is_live_wallet_profile = bool(profile_id and str(profile_id).startswith("0x"))

        for market_id, primary in iter_unified_active_entries(
            history, include_exited=(sub_mode in ["predicted", "predicted_skipped"])
        ):

            p_side = primary.get("position", "Yes")
            p_pu = float(primary.get("pu", 0.5))
            p_pm = float(primary.get("pm", 0.5))
            p_cur_pm = primary.get("cur_pm")
            p_cur_p = float(p_cur_pm if p_cur_pm is not None else p_pm)
            p_k = float(primary.get("entry_kelly") or primary.get("effective_kelly", 0.0) or primary.get("kelly", 0.0) or 0.0)
            p_ego = compute_dynamic_ego(base_ego, float(primary.get("u_spread", 0.0) or 0.0))

            is_wallet = (primary.get("source", "local") == "wallet")
            if is_live_wallet_profile and not is_wallet:
                live_val, held_mtm_k, sh = 0.0, 0.0, 0.0
            else:
                live_val, held_mtm_k, sh = calculate_position_holding(primary, p_cur_p, live_bankroll)

            is_held_pos = (held_mtm_k > 0 or live_val > 0)
            eval_cur_px = p_cur_pm if is_held_pos else (float(primary.get("bestAsk") or p_cur_pm or p_pm))
            p_fee_rate = float(primary.get("fee_rate", 0.0) or 0.0)
            p_lm = calculate_logistic_metrics(p_pu, p_pm, p_ego, held_kelly=None, cur_pm=eval_cur_px, fee_rate=(0.0 if is_held_pos else p_fee_rate))

            spread_val = float(primary.get("spread", 0.01) or 0.01)
            half_sp = max(0.0005, spread_val / 2.0)
            dyn_action = calculate_dynamic_kelly_action(
                win_prob_bal=p_lm["win_prob_bal"],
                cur_p=p_cur_p,
                held_frac=held_mtm_k,
                bankroll=live_bankroll,
                fee_rate=p_fee_rate,
                half_spread=half_sp,
                entry_kelly=max(p_lm["eff_kelly"], p_k),
                best_bid=primary.get("bestBid"),
                best_ask=primary.get("bestAsk")
            )

            end_date_str = primary.get("endDate") or primary.get("end_date")
            _, seconds_left, days_until = parse_market_date(end_date_str)
            active_edge = float(p_lm["live_active_edge"])
            price_ref = p_lm["win_prob_m"] if p_lm["win_prob_m"] > 0 else (p_cur_p if p_cur_p > 0 else 0.5)

            if days_until is not None and days_until > 0 and price_ref > 0:
                daily_edge = active_edge / days_until
                ann_edge = daily_edge * 365.0 * 100.0
                _, apy_val = calculate_annualized_yield(active_edge, price_ref, days_until)
            else:
                daily_edge = None
                ann_edge = None
                apy_val = primary.get("apy")

            if seconds_left is not None:
                if seconds_left <= 0:
                    ends_in_disp = "Ended"
                elif days_until >= 1.0:
                    ends_in_disp = f"{days_until:.1f}d"
                else:
                    ends_in_disp = f"{max(0.1, seconds_left / 3600.0):.1f}h"
            else:
                ends_in_disp = "-"

            vel_d_disp = f"{daily_edge*100:+.2f}%/d" if daily_edge is not None else "-"
            apy_disp = f"{float(apy_val)*100:+.1f}%" if (apy_val is not None and -1.0 <= float(apy_val) <= 99.99) else ("-" if apy_val is None else ">9999%")

            src_val = _format_source_tag(primary, alias_tag)

            if is_held_pos:
                if dyn_action["signal_code"] == "ADD":
                    add_usd = dyn_action["delta_usd"]
                    opt_usd = primary.get("opt_buy_usd")
                    opt_vwap = primary.get("opt_buy_vwap")
                    if opt_usd is not None and opt_usd > 0 and opt_usd < add_usd * 0.95:
                        regime_str = f"ADD +${opt_usd:,.0f} (Mean {opt_vwap*100:.1f}%)"
                    elif opt_usd == 0.0:
                        regime_str = "HOLD (Ask exceeds hurdle)"
                    else:
                        regime_str = dyn_action["action_label"]
                else:
                    regime_str = dyn_action["action_label"]
            else:
                min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))
                effective_min_edge = max(min_edge_th, 0.01)
                vwap_edge = primary.get("vwap_edge")
                opt_usd = primary.get("opt_buy_usd")
                opt_vwap = primary.get("opt_buy_vwap")
                sig_unheld = classify_active_signal(
                    False, dyn_action, p_lm["eff_kelly"], active_edge,
                    min_edge_th=min_edge_th, opt_buy_usd=opt_usd, vwap_edge=vwap_edge,
                )
                if sig_unheld == "BUY":
                    buy_usd = p_lm["eff_kelly"] * live_bankroll if live_bankroll > 0 else 0.0
                    if opt_usd is not None and opt_usd > 0 and opt_usd < buy_usd * 0.95:
                        regime_str = f"EXEC BUY ${opt_usd:,.0f} (Mean {opt_vwap*100:.1f}%)"
                    else:
                        regime_str = f"EXEC BUY ${buy_usd:,.0f} ({p_lm['eff_kelly']*100:.1f}%)"
                elif active_edge <= 0:
                    regime_str = f"WATCH (No Edge: {active_edge*100:+.1f}%)"
                elif opt_usd == 0.0 or (vwap_edge is not None and vwap_edge <= 0):
                    regime_str = "WATCH (Ask exceeds hurdle)"
                elif vwap_edge is not None and vwap_edge < effective_min_edge:
                    regime_str = f"WATCH (Depth Edge < {effective_min_edge*100:.1f}%)"
                else:
                    regime_str = f"WATCH (Thin Edge: {active_edge*100:+.1f}%)"

            q_title = primary.get("question", "unknown")
            compact_q = format_compact_question(q_title, 34)

            eff_target_k = dyn_action["f_star_eff"] if is_held_pos else p_lm["eff_kelly"]
            vol_m = float(primary.get("volume", 0.0) or primary.get("volumeNum", 0.0) or 0.0)
            eff_m_val = 4.0 * max(0.0, min(1.0, p_cur_p)) * (1.0 - max(0.0, min(1.0, p_cur_p)))
            eii_m = eff_m_val * vol_m
            days_c = max(0.1, days_until) if days_until is not None else 30.0
            cey_m = eii_m / math.sqrt(days_c + 7.0)

            snaps = primary.get("snapshots") or []
            latest_snap_date = snaps[-1].get("date") if snaps and isinstance(snaps[-1], dict) else None
            last_pu_date = primary.get("last_pu_date") or latest_snap_date or primary.get("date") or ""
            last_pu_ts = primary.get("last_pu_ts")
            if last_pu_date:
                try:
                    dt = datetime.fromisoformat(str(last_pu_date).strip().replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    last_pu_ts = dt.timestamp()
                except Exception:
                    last_pu_ts = 0.0
            else:
                last_pu_ts = 0.0

            rows.append({
                "Mode": "active",
                "Source": src_val,
                "Side": p_side,
                "Market ID": market_id,
                "Date": str(last_pu_date)[:10] if last_pu_date else "unknown",
                "LastPuDate": str(last_pu_date),
                "LastPuTs": last_pu_ts or 0.0,
                "Question": compact_q,
                "raw_question": q_title,
                "Web Link": f"https://polymarket.com/event/{primary.get('slug', 'unknown')}",
                "Kelly": eff_target_k,
                "EffKelly": eff_target_k,
                "HeldVal": live_val,
                "HeldFrac": held_mtm_k,
                "Shares": sh,
                "Edge": active_edge,
                "DailyEdge": daily_edge,
                "APY": apy_val,
                "DaysUntil": days_until,
                "CEY": cey_m,
                "EII": eii_m,
                "Volume": vol_m,
                "VelDStr": vel_d_disp,
                "APYStr": apy_disp,
                "EndsInStr": ends_in_disp,
                "DistTarget": p_lm["dist_to_target"],
                "pu": p_pu,
                "p_norm": p_lm["win_prob_bal"],
                "pm": p_pm,
                "cur_pm": float(p_cur_pm) if p_cur_pm is not None else p_pm,
                "Target": p_lm["target_win_m"],
                "PnL": primary.get("realized_pnl"),
                "Regime": regime_str,
                "Outcome": "",
                "vwap": primary.get("vwap"),
                "slippage": primary.get("slippage"),
                "bestAsk": primary.get("bestAsk"),
                "bestBid": primary.get("bestBid"),
                "vwap_edge": primary.get("vwap_edge"),
                "opt_buy_usd": primary.get("opt_buy_usd"),
                "opt_buy_vwap": primary.get("opt_buy_vwap"),
                "opt_buy_edge": primary.get("opt_buy_edge"),
                "entry_price": primary.get("entry_price") or primary.get("pm"),
                "fee_rate": p_fee_rate,
                "clobTokenIds": primary.get("clobTokenIds"),
                "snapshots": snaps
            })

    # 2. Exited Rows
    if sub_mode in ["exited", "all"]:
        for market_id, value in history.get("exited", {}).items():
            for pred in ensure_list(value):
                pu_v = float(pred.get("pu", 0.5))
                pm_v = float(pred.get("pm", 0.5))
                cur_pm_v = pred.get("cur_pm")
                exit_price_v = pred.get("exit_price")
                k_v = calculate_trade_weight({**pred, "exclude_from_ego": False}, bankroll=live_bankroll)
                ego_v = float(pred.get("dynamic_ego", base_ego))
                f_rate_v = float(pred.get("fee_rate", 0.0) or 0.0)
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v, cur_pm=cur_pm_v, fee_rate=f_rate_v)

                side_val = pred.get("position", "Yes" if lm["is_yes"] else "No")
                pu_disp = pu_v
                pm_disp = pm_v
                p_norm_disp = lm["win_prob_bal"]

                if exit_price_v is not None:
                    exit_disp = float(exit_price_v)
                else:
                    exit_disp = float(cur_pm_v) if cur_pm_v is not None else pm_v

                src_val = _format_source_tag(pred, alias_tag)

                out_val = exit_disp
                trade_ego = calculate_trade_ego_metrics({**pred, "exclude_from_ego": False}, cat="exited", bankroll=live_bankroll)
                w = 0.0 if pred.get("exclude_from_ego") else trade_ego["weight"]
                bs_u = trade_ego["bs_u"]
                bs_m = trade_ego["bs_m"]
                ego_drag = trade_ego["ego_drag"]

                q_ex = pred.get("question", "unknown")
                ts = pred.get("last_synced_ts") or pred.get("timestamp")
                if ts and int(ts) > 0:
                    d_val = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
                else:
                    d_val = str(pred.get("date", "unknown"))[:10]

                vol_ex = float(pred.get("volume", 0.0) or pred.get("volumeNum", 0.0) or 0.0)
                eff_ex = 4.0 * max(0.0, min(1.0, exit_disp)) * (1.0 - max(0.0, min(1.0, exit_disp)))
                eii_ex = eff_ex * vol_ex
                _, _, days_ex = parse_market_date(pred.get("endDate") or pred.get("end_date"))
                cey_ex = eii_ex / math.sqrt(max(0.1, days_ex if days_ex is not None else 30.0) + 7.0)

                rows.append({
                    "Mode": "exited",
                    "Source": src_val,
                    "Side": side_val,
                    "Market ID": market_id,
                    "Date": d_val,
                    "Question": format_compact_question(q_ex, 34),
                    "raw_question": q_ex,
                    "Web Link": f"https://polymarket.com/event/{pred.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "Edge": lm["active_edge"],
                    "DistTarget": lm["dist_to_target"],
                    "CEY": cey_ex,
                    "EII": eii_ex,
                    "Volume": vol_ex,
                    "pu": pu_disp,
                    "p_norm": p_norm_disp,
                    "pm": pm_disp,
                    "cur_pm": exit_disp,
                    "Target": lm["target_win_m"],
                    "PnL": float(pred.get("realized_pnl", 0.0) or 0.0),
                    "Regime": "EXITED (Await Res)",
                    "Outcome": "",
                    "EgoDrag": ego_drag,
                    "bs_u": bs_u,
                    "bs_m": bs_m,
                    "adv": -ego_drag,
                    "weight": w,
                    "fee_rate": f_rate_v,
                    "entry_price": pred.get("entry_price") or pm_disp,
                    "snapshots": pred.get("snapshots", [])
                })

    # 3. Skipped Rows
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

    # 4. Resolved Rows
    if sub_mode in ["resolved", "all"]:
        for market_id, value in history.get("resolved", {}).items():
            for res in ensure_list(value):
                # Ensure market is completely resolved with a definitive settled outcome (0.0, 1.0, 0.5)
                if res.get("outcome") is None or res.get("outcome") not in (0.0, 1.0, 0.5):
                    continue
                pu_v = float(res.get("pu", 0.5))
                pm_v = float(res.get("pm", 0.5))
                k_v = calculate_trade_weight({**res, "exclude_from_ego": False}, bankroll=live_bankroll)
                ego_v = float(res.get("dynamic_ego", base_ego))
                f_rate_v = float(res.get("fee_rate", 0.0) or 0.0)
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v, fee_rate=f_rate_v)

                side_val = res.get("position", "Yes" if lm["is_yes"] else "No")
                pu_disp = pu_v
                pm_disp = pm_v
                p_norm_disp = lm["win_prob_bal"]

                out_val = float(res.get("outcome", 0.5) or 0.5)
                trade_ego = calculate_trade_ego_metrics({**res, "exclude_from_ego": False}, cat="resolved", bankroll=live_bankroll)
                w = 0.0 if res.get("exclude_from_ego") else trade_ego["weight"]
                bs_u = trade_ego["bs_u"]
                bs_m = trade_ego["bs_m"]
                ego_drag = trade_ego["ego_drag"]

                src_val = _format_source_tag(res, alias_tag)

                q_res = res.get("question", "unknown")
                ts = res.get("last_synced_ts") or res.get("timestamp")
                if ts and int(ts) > 0:
                    d_val = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
                else:
                    d_val = str(res.get("date", "unknown"))[:10]

                vol_res = float(res.get("volume", 0.0) or res.get("volumeNum", 0.0) or 0.0)
                eff_res = 4.0 * max(0.0, min(1.0, pm_disp)) * (1.0 - max(0.0, min(1.0, pm_disp)))
                eii_res = eff_res * vol_res
                _, _, days_res = parse_market_date(res.get("endDate") or res.get("end_date"))
                cey_res = eii_res / math.sqrt(max(0.1, days_res if days_res is not None else 30.0) + 7.0)

                rows.append({
                    "Mode": "resolved",
                    "Source": src_val,
                    "Side": side_val,
                    "Market ID": market_id,
                    "Date": d_val,
                    "Question": format_compact_question(q_res, 34),
                    "raw_question": q_res,
                    "Web Link": f"https://polymarket.com/event/{res.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "Edge": lm["active_edge"],
                    "DistTarget": lm["dist_to_target"],
                    "CEY": cey_res,
                    "EII": eii_res,
                    "Volume": vol_res,
                    "pu": pu_disp,
                    "p_norm": p_norm_disp,
                    "pm": pm_disp,
                    "cur_pm": pm_disp,
                    "Target": lm["target_win_m"],
                    "PnL": res.get("realized_pnl"),
                    "Regime": lm["regime"],
                    "Outcome": str(res.get("outcome", 0.5)),
                    "EgoDrag": ego_drag,
                    "bs_u": bs_u,
                    "bs_m": bs_m,
                    "adv": -ego_drag,
                    "weight": w,
                    "fee_rate": f_rate_v,
                    "entry_price": res.get("entry_price") or pm_disp,
                    "snapshots": res.get("snapshots", [])
                })

    if not rows:
        print(f"\n[!] No markets found in '{sub_mode}'.")
        return

    # Default Sort per View (Least Mean Action: instant render without pre-prompt; re-sort anytime via 's')
    default_sorts = {
        "predicted": "velocity",
        "exited": "date",
        "resolved": "ego_impact",
        "skipped": "date",
        "all": "kelly",
    }
    sort_map = {
        "1": "date", "date": "date", "pu": "date", "last_pu": "date", "updated": "date",
        "2": "kelly", "kelly": "kelly", "k": "kelly",
        "3": "edge", "edge": "edge", "e": "edge",
        "4": "exit_prox", "exit": "exit_prox", "prox": "exit_prox",
        "5": "velocity", "vel": "velocity", "velocity": "velocity", "v": "velocity", "daily": "velocity",
        "6": "ego_impact", "ego": "ego_impact", "brier": "ego_impact", "b": "ego_impact",
        "7": "cey", "cey": "cey", "c": "cey",
    }
    if sort_by is None:
        sort_by = default_sorts.get(sub_mode, "date")

    if sort_by == "date":
        rows.sort(key=lambda x: (x.get("LastPuTs") or 0.0, x.get("LastPuDate", ""), x.get("Date", "")), reverse=True)
    elif sort_by == "kelly":
        rows.sort(key=lambda x: (0 if x.get("Kelly") is None else 1, x.get("Kelly") or 0.0, x.get("Edge") or 0.0), reverse=True)
    elif sort_by == "edge":
        rows.sort(key=lambda x: (0 if x.get("Edge") is None else 1, x.get("Edge") or 0.0, x.get("Kelly") or 0.0), reverse=True)
    elif sort_by == "exit_prox":
        rows.sort(key=lambda x: (1 if x.get("DistTarget") is None else 0, x.get("DistTarget") if x.get("DistTarget") is not None else float("inf")))
    elif sort_by == "velocity":
        rows.sort(key=lambda x: (0 if x.get("DailyEdge") is None else 1, x.get("DailyEdge") or 0.0, x.get("Edge") or 0.0, x.get("Kelly") or 0.0), reverse=True)
    elif sort_by == "ego_impact":
        rows.sort(key=lambda x: (0 if x.get("EgoDrag") is None else 1, 0 if x.get("Source") == "[EXCL]" else 1, x.get("EgoDrag") or 0.0, x.get("Kelly") or 0.0), reverse=True)
    elif sort_by == "cey":
        rows.sort(key=lambda x: (0 if x.get("CEY") is None else 1, x.get("CEY") or 0.0, x.get("Edge") or 0.0, x.get("Kelly") or 0.0), reverse=True)

    if sort_by == "ego_impact":
        config = TABLE_CONFIGS["ego_impact"]
    else:
        config = dict(TABLE_CONFIGS.get(sub_mode, TABLE_CONFIGS["all"]))
        if sort_by == "cey" and sub_mode == "predicted":
            config["headers"] = ["#" if h == "#" else ("CEY" if h == "Vel/d" else h) for h in config["headers"]]
    headers = config["headers"]
    widths = config["widths"]
    aligns = config["aligns"]
    q_idx = config["q_idx"]

    def format_cell(text, width, align):
        return str(text).rjust(width)[:width] if align == "right" else str(text).ljust(width)[:width]

    header_str = " | ".join(format_cell(h, w, a) for h, w, a in zip(headers, widths, aligns))
    separator_str = "-+-".join("-" * w for w in widths)

    print("\n" + "=" * len(header_str))
    sort_disp = "LAST PU DATE" if sort_by == "date" else sort_by.upper()
    print(f" [{sub_mode.upper()} MARKETS - sorted by {sort_disp}] ".center(len(header_str), "="))
    print("=" * len(header_str))
    print(header_str)
    print(separator_str)

    for row_i, r in enumerate(rows, 1):
        kelly_val = r.get("Kelly")
        pu_val = r.get("pu")
        p_norm_val = r.get("p_norm")
        pm_val = r.get("pm")
        cur_pm_val = r.get("cur_pm")
        pnl_val = r.get("PnL")
        regime_val = r.get("Regime", "-")
        side_str = r.get("Side", "-")
        src_str = r.get("Source", "-")
        edge_v = r.get("Edge")

        kelly_str = f"{kelly_val*100:.2f}%" if kelly_val is not None else "-"
        edge_str = f"{edge_v*100:+.2f}%" if (edge_v is not None and abs(edge_v) < 0.0095 and abs(edge_v) > 1e-6) else (f"{edge_v*100:+.1f}%" if edge_v is not None else "-")
        pu_str = f"{pu_val*100:.1f}%" if pu_val is not None else "-"
        p_norm_str = f"{p_norm_val*100:.1f}%" if p_norm_val is not None else "-"
        pm_str = f"{pm_val*100:.1f}%" if pm_val is not None else "-"
        live_str = f"{cur_pm_val*100:.1f}%" if cur_pm_val is not None else "-"
        pnl_str = f"${pnl_val:+,.2f}" if pnl_val is not None else "-"

        outcome_val = r.get("Outcome")
        if outcome_val == "1.0": outcome_str = "WON"
        elif outcome_val == "0.0": outcome_str = "LOST"
        elif outcome_val == "0.5": outcome_str = "HALF"
        else: outcome_str = "-"

        row_num = str(row_i)
        if sort_by == "ego_impact":
            bs_u_s = f"{r['bs_u']:.4f}" if r.get("bs_u") is not None else "-"
            bs_m_s = f"{r['bs_m']:.4f}" if r.get("bs_m") is not None else "-"
            adv_s = f"{r['adv']:+.4f}" if r.get("adv") is not None else "-"
            st_out = outcome_str if r["Mode"] == "resolved" else ("SOLD" if r["Mode"] == "exited" else "-")
            vals = [row_num, r["Question"], side_str, st_out, kelly_str, bs_u_s, bs_m_s, adv_s, pnl_str, r["Date"], src_str]
        elif sub_mode == "predicted":
            held_v = r.get("HeldVal", 0.0)
            held_f = r.get("HeldFrac", 0.0)
            held_str = f"${held_v:,.0f} ({held_f*100:.1f}%)" if held_v > 0 else "-"
            vel_or_cey = f"{r.get('CEY', 0.0):,.0f}" if sort_by == "cey" else r.get("VelDStr", "-")
            vals = [row_num, r["Question"], regime_val, held_str, kelly_str, edge_str, vel_or_cey, side_str, live_str, p_norm_str, r["Date"], src_str]
        elif sub_mode == "exited":
            vals = [row_num, r["Question"], pnl_str, kelly_str, edge_str, side_str, pm_str, live_str, pu_str, p_norm_str, r["Date"], src_str]
        elif sub_mode == "resolved":
            vals = [row_num, r["Question"], outcome_str, pnl_str, kelly_str, edge_str, side_str, pm_str, pu_str, p_norm_str, r["Date"], src_str]
        elif sub_mode == "skipped":
            vals = [row_num, r["Question"], r["Date"]]
        else:
            st_res = f"{outcome_str} ({pnl_str})" if r["Mode"] == "resolved" else (f"EXITED ({pnl_str})" if r["Mode"] == "exited" else regime_val)
            pos_or_pnl = pnl_str if r["Mode"] in ("resolved", "exited") else (f"${r.get('HeldVal', 0):,.0f}" if r.get('HeldVal', 0) > 0 else "-")
            vals = [row_num, r["Question"], r["Mode"].upper(), st_res, pos_or_pnl, kelly_str, edge_str, side_str, live_str, p_norm_str, r["Date"]]

        row_parts = []
        for i, (v, w, a) in enumerate(zip(vals, widths, aligns)):
            if i == q_idx:
                q_visible = (v[:w-3] + "...") if len(v) > w else v
                q_padded = q_visible.ljust(w)
                row_parts.append(ansi_link(r['Web Link'], q_padded))
            else:
                row_parts.append(format_cell(v, w, a))
        print(" | ".join(row_parts))

        if expanded:
            full_q = r.get("raw_question") or r.get("Question")
            print(f"    | Full: {full_q}")
            if r.get("Web Link"):
                print(f"    | URL : {r.get('Web Link')}")

    print("=" * len(header_str) + "\n")
    if sub_mode in ["predicted", "exited", "resolved", "all"]:
        print_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego, refresh_meta=False)

    # Action Bar: Launch Prediction
    if sub_mode == "predicted":
        exit_signals = [r for r in rows if any(k in r.get("Regime", "") for k in ("SELL", "EXIT", "TAKE PROFIT"))]
        if exit_signals:
            print(f"  [*] Take-Profit / Exit Alert: {len(exit_signals)} position{'s' if len(exit_signals) > 1 else ''} near or reached target (Limit Sell or Market Sell on Polymarket recommended)")

    prompt_action = "predict / update" if sub_mode == "predicted" else "view market"
    sort_hint = ", 's' (sort)" if sub_mode in ("predicted", "exited", "resolved", "all") else ""
    cal_hint = ", 'cal' (calibrate ego)" if sub_mode in ("resolved", "exited", "all") else ""
    exp_hint = ", 'x' (collapse)" if expanded else ", 'x' (expand questions)"
    live_hint = ", 'l' (live refresh)" if sub_mode in ("predicted", "exited", "resolved", "all") else ""
    u_hint = ", 'u <#>' (undo update)" if sub_mode == "predicted" else ""
    active_pid_bar = history.get("profile_id") or get_active_profile()
    is_wallet_bar = bool(active_pid_bar and str(active_pid_bar).startswith("0x"))
    if is_wallet_bar and sub_mode in ("predicted", "all"):
        from trade import is_live_trading_ready
        ord_hint = " / 'orders'" if is_live_trading_ready(profile_id=active_pid_bar) else ""
        trade_hint = f", 'buy <#>' / 'sell <#>'{ord_hint}"
    else:
        trade_hint = ""
    range_str = f"({len(rows)})" if len(rows) == 1 else f"(1-{len(rows)})"
    act = input(f"Select row # to {prompt_action} {range_str}{sort_hint}{exp_hint}{live_hint}, 'd <#>' (details){u_hint}{trade_hint}{cal_hint} [Enter for Main Menu]: ").strip()
    act_low = act.lower()
    if act_low in ("x", "expand", "exp"):
        return print_review_table(history, sub_mode, expanded=not expanded, sort_by=sort_by, refresh_meta=False)
    elif (
        act_low in ("s", "sort", "date", "pu", "last_pu", "updated", "kelly", "edge", "exit", "prox", "vel", "velocity", "daily", "brier", "cey", "c")
        or act_low.startswith("s ")
        or act_low.startswith("sort ")
    ) and sub_mode in ("predicted", "exited", "resolved", "all"):
        if act_low in sort_map and act_low not in ("s", "sort"):
            new_sort = sort_map[act_low]
        elif " " in act_low:
            sub_tok = act_low.split(None, 1)[1].strip()
            new_sort = sort_map.get(sub_tok, sort_by)
        else:
            cycle_map = {
                "predicted": ["velocity", "cey", "kelly", "edge", "exit_prox", "date"],
                "exited": ["date", "ego_impact", "cey", "kelly", "edge"],
                "resolved": ["ego_impact", "date", "cey", "kelly", "edge"],
                "all": ["kelly", "velocity", "cey", "edge", "ego_impact", "date"],
            }
            seq = cycle_map.get(sub_mode, ["date", "kelly", "edge", "cey"])
            next_s = seq[(seq.index(sort_by) + 1) % len(seq)] if sort_by in seq else seq[0]
            vel_h = "  [5] Velocity (Edge/d)" if sub_mode in ("predicted", "all") else ""
            ego_h = "  [6] Ego Impact (Brier)" if sub_mode in ("resolved", "exited", "all") else ""
            cey_h = "  [7] CEY"
            d_lbl = "Last PU Date" if sub_mode == "predicted" else "Date"
            s_in = input(f"Sort '{sub_mode}' by: [1] {d_lbl}  [2] Kelly  [3] Edge  [4] Exit Proximity{vel_h}{ego_h}{cey_h} [Enter -> {next_s.upper()}]: ").strip().lower()
            new_sort = sort_map.get(s_in, next_s) if s_in else next_s
        return print_review_table(history, sub_mode, expanded=expanded, sort_by=new_sort, refresh_meta=False)
    elif act.lower() in ("l", "live", "refresh") and sub_mode in ("predicted", "exited", "resolved", "all"):
        active_pid = history.get("profile_id") or get_active_profile()
        if active_pid and str(active_pid).startswith("0x"):
            from wallet import import_polymarket_positions, update_resolutions
            print(f"[*] Syncing live wallet & resolutions for '{active_pid[:6]}...{active_pid[-4:]}'...")
            history = import_polymarket_positions(active_pid, history, auto_sync=True)
            history = update_resolutions(history)
            save_history(history, profile_id=active_pid)
        else:
            from wallet import update_resolutions
            history = update_resolutions(history)
            save_history(history)
        if sub_mode in ("predicted", "all"):
            print("[*] Refreshing live CLOB orderbooks & VWAP for active markets...")
            ensure_predicted_market_metadata(history, fetch_clob=True)
        return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
    elif act.lower().startswith("d ") or act.lower().startswith("details "):
        parts = act.split()
        if len(parts) > 1 and parts[1].isdigit():
            d_idx = int(parts[1]) - 1
            if 0 <= d_idx < len(rows):
                _print_single_market_details(rows[d_idx])
                input("\nPress Enter to return to table: ")
                return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
            else:
                print(f"[!] Row number out of range {range_str}.")
                return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
        else:
            print("[!] Usage: 'd <row#>' (e.g. 'd 1')")
            return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
    elif (act.lower().startswith("u ") or act.lower().startswith("undo ")) and sub_mode == "predicted":
        parts = act.split()
        if len(parts) > 1 and parts[1].isdigit():
            u_idx = int(parts[1]) - 1
            if 0 <= u_idx < len(rows):
                r_target = rows[u_idx]
                m_id = str(r_target.get("Market ID"))
                preds = ensure_list(history.get("predicted", {}).get(m_id, [])) or ensure_list(history.get("exited", {}).get(m_id, []))
                if preds:
                    p_target = preds[0]
                    snaps = p_target.get("snapshots")
                    if isinstance(snaps, list) and len(snaps) >= 2:
                        last_s = snaps[-1]
                        prev_s = snaps[-2]
                        conf = input(f"Undo latest update on #{u_idx + 1} (revert pu: {float(last_s.get('pu', 0))*100:.1f}% -> {float(prev_s.get('pu', 0))*100:.1f}%)? (y/n): ").strip().lower()
                        if conf == "y":
                            snaps.pop()
                            restore_snapshot_to_prediction(p_target, prev_s, base_ego=base_ego)
                            save_history(history)
                            print(f"[+] Successfully undid latest update on #{u_idx + 1}.")
                            return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
                    else:
                        print(f"[!] No previous updates to undo for row #{u_idx + 1} (only initial prediction exists).")
                else:
                    print(f"[!] Market not found in active history.")
            else:
                print(f"[!] Row number out of range {range_str}.")
        else:
            print("[!] Usage: 'u <row#>' (e.g. 'u 2')")
        return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
    elif act.lower() in ("orders", "open orders") and sub_mode in ("predicted", "all"):
        _handle_open_orders_view(history)
        return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
    elif (act.lower().startswith("buy ") or act.lower().startswith("sell ")) and sub_mode in ("predicted", "all"):
        parts = act.split()
        cmd = parts[0].lower()
        if len(parts) > 1 and parts[1].isdigit():
            t_idx = int(parts[1]) - 1
            if 0 <= t_idx < len(rows):
                traded = _handle_live_trade_row(history, rows[t_idx], side=cmd.upper())
                if traded:
                    history = load_history()
                return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=bool(traded))
            else:
                print(f"[!] Row number out of range {range_str}.")
                return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
        else:
            print(f"[!] Usage: '{cmd} <row#>' (e.g. '{cmd} 1')")
            return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
    elif act.lower() in ("cal", "ego", "e"):
        from manage import toggle_ego_market_inclusion
        toggle_ego_market_inclusion(history)
        return
    elif act.isdigit():
        row_num = int(act)
        if 1 <= row_num <= len(rows):
            sel_r = rows[row_num - 1]
            if sel_r.get("Mode") == "resolved":
                _print_single_market_details(sel_r)
                input("\nPress Enter to return to table: ")
                return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)
            _handle_targeted_predict(rows, preselected_idx=row_num - 1)
            history = load_history()
            return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=(sub_mode in ("predicted", "all")))
        else:
            print(f"[!] Row number out of range {range_str}.")
            return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)

def _resolve_row_clob_token(history, row):
    """Resolves the CLOB token ID for a review table row (fetching from Gamma if not cached)."""
    from market import get_market_clob_tokens
    m_id = str(row.get("Market ID") or "")
    pos_side = row.get("Side", "Yes")
    clob_ids = row.get("clobTokenIds")
    if not clob_ids and m_id:
        for bucket in ("predicted", "exited"):
            preds = ensure_list(history.get(bucket, {}).get(m_id, []))
            if preds and preds[0].get("clobTokenIds"):
                clob_ids = preds[0].get("clobTokenIds")
                break
    if not clob_ids and m_id and not m_id.startswith("custom_"):
        try:
            r = cfg.api.get(f"{cfg.GAMMA_API}/markets?id={m_id}", timeout=5).json()
            if isinstance(r, list) and r:
                clob_ids = r[0].get("clobTokenIds")
        except Exception:
            pass
    yes_tok, no_tok = get_market_clob_tokens({"clobTokenIds": clob_ids})
    return (no_tok if pos_side == "No" else yes_tok), pos_side


def _handle_open_orders_view(history):
    """Displays resting CLOB limit orders and allows cancelling one or all."""
    from trade import get_clob_config, fetch_open_orders, cancel_open_order, get_clob_client
    active_pid = history.get("profile_id") or get_active_profile()
    conf = get_clob_config(profile_id=active_pid, detect_remote=False)
    if not conf["is_live_wallet"] or not conf["has_private_key"]:
        print("[*] Zero-Key mode active (no CLOB_PRIVATE_KEY on disk). Manage open orders directly at https://polymarket.com/portfolio.")
        input("Press Enter to return: ")
        return
    try:
        client = get_clob_client(profile_id=active_pid)
        orders = fetch_open_orders(profile_id=active_pid, client=client)
    except Exception as e:
        print(f"[!] Could not fetch open orders: {e}")
        input("Press Enter to return: ")
        return

    if not orders:
        print("\n[*] No open CLOB orders found.")
        input("Press Enter to return: ")
        return

    print("\n--- OPEN CLOB ORDERS ---")
    for i, o in enumerate(orders, 1):
        oid = o.get("id") or o.get("orderID") or "?"
        side = o.get("side", "?")
        px = float(o.get("price", 0) or 0)
        sz = float(o.get("original_size") or o.get("size") or 0)
        matched = float(o.get("size_matched") or 0)
        print(f"  [{i}] {side:<4} {sz:g} sh @ {px*100:.1f}% (filled: {matched:g}) | ID: {oid}")

    c_in = input("\nCancel order # (e.g. '1'), 'all' to cancel all, or Enter to go back: ").strip().lower()
    if c_in == "all":
        try:
            client.cancel_all()
            print("[+] Cancelled all open orders.")
        except Exception as e:
            print(f"[!] Cancel all failed: {e}")
    elif c_in.isdigit():
        idx = int(c_in) - 1
        if 0 <= idx < len(orders):
            oid = orders[idx].get("id") or orders[idx].get("orderID")
            try:
                cancel_open_order(oid, profile_id=active_pid, client=client)
                print(f"[+] Cancelled order {oid}.")
            except Exception as e:
                print(f"[!] Cancel failed: {e}")


def _handle_live_trade_row(history, row, side="SELL"):
    """Interactive BUY/SELL pre-flight (zero-key browser ticket or direct CLOB execution)."""
    from trade import (
        get_clob_config,
        fetch_orderbook,
        preview_buy_order,
        preview_sell_order,
        execute_order,
        format_browser_ticket,
        prompt_browser_confirm_and_sync,
        sync_wallet_after_trade,
    )

    active_pid = history.get("profile_id") or get_active_profile()
    conf = get_clob_config(profile_id=active_pid, detect_remote=False)
    if not conf["is_live_wallet"]:
        print("[!] Pre-flight trading requires a 0x wallet profile (switch via Option 4).")
        input("Press Enter to continue: ")
        return False

    if row.get("Mode") != "active":
        print("[!] Only active markets can be traded.")
        input("Press Enter to continue: ")
        return False

    token_id, pos_side = _resolve_row_clob_token(history, row)
    if not token_id:
        print("[!] Could not resolve CLOB token ID for this market.")
        input("Press Enter to continue: ")
        return False

    try:
        book = fetch_orderbook(token_id)
    except Exception as e:
        print(f"[!] Failed to fetch live CLOB book: {e}")
        input("Press Enter to continue: ")
        return False

    q_disp = row.get("raw_question") or row.get("Question")
    bid_s = f"{book['best_bid']*100:.1f}%" if book["best_bid"] else "N/A"
    ask_s = f"{book['best_ask']*100:.1f}%" if book["best_ask"] else "N/A"
    mode_lbl = "LIVE" if conf["has_private_key"] else "PRE-FLIGHT"
    print(f"\n--- {mode_lbl} {side} [{pos_side}] ---")
    print(f"Market    : {q_disp}")
    print(f"Live Book : Bid {bid_s} | Ask {ask_s} | Tick {book['tick_size']} | Min Size {book['min_order_size']:g} sh")
    if conf["has_private_key"]:
        conf_full = get_clob_config(profile_id=active_pid, detect_remote=True)
        print(f"Auth      : {conf_full['signature_label']} ({conf_full['signature_note']})")
    else:
        print("Mode      : Zero-Key (Pre-Flight & Browser Confirm)")

    if side == "SELL":
        held_sh = float(row.get("Shares", 0.0) or 0.0)
        if held_sh <= 0:
            print("[!] You have 0 tracked shares to sell in this row.")
            input("Press Enter to continue: ")
            return False
        sh_raw = input(f"Shares to sell [Enter for ALL {held_sh:,.2f} sh, or % like '50%']: ").strip()
        if not sh_raw:
            trade_amt = held_sh
        elif sh_raw.endswith("%"):
            try:
                pct = max(0.0, min(100.0, float(sh_raw[:-1].strip())))
                trade_amt = round(held_sh * (pct / 100.0), 2)
            except ValueError:
                print("[!] Invalid percentage.")
                return False
        else:
            try:
                trade_amt = max(0.0, min(held_sh, float(sh_raw)))
            except ValueError:
                print("[!] Invalid share amount.")
                return False
    else:
        ws = history.get("wallet_summary", {})
        cash = float(ws.get("cash", cfg.BANKROLL) or 0.0) if isinstance(ws, dict) else float(cfg.BANKROLL)
        bankroll = float(ws.get("bankroll", cfg.BANKROLL) or cfg.BANKROLL) if isinstance(ws, dict) else float(cfg.BANKROLL)
        held_f = float(row.get("HeldFrac", 0.0) or 0.0)
        tgt_k = float(row.get("Kelly", 0.0) or 0.0)
        opt_usd = row.get("opt_buy_usd")
        rec_usd = float(opt_usd) if opt_usd is not None else max(0.0, (tgt_k - held_f) * bankroll)
        rec_usd = min(rec_usd, cash) if cash > 0 else rec_usd
        usd_raw = input(f"USDC to spend (Cash: ${cash:,.2f}) [Enter for ${rec_usd:,.2f}]: ").strip().replace("$", "")
        try:
            trade_amt = float(usd_raw) if usd_raw else rec_usd
        except ValueError:
            print("[!] Invalid dollar amount.")
            return False

    ot_in = input(f"Order Type: [1] Market {side.title()} (FAK immediate) [Enter]  [2] Limit {side.title()} (GTC resting): ").strip()
    if ot_in == "2":
        def_px = (row.get("Target") or book["best_ask"] or book["best_bid"] or 0.5) if side == "SELL" else (book["best_bid"] or book["best_ask"] or 0.5)
        px_in = input(f"Limit price % (e.g. '{def_px*100:.0f}' for {def_px*100:.0f}%) [Enter for {def_px*100:.1f}%]: ").strip().replace("%", "")
        try:
            lim_px = (float(px_in) / 100.0) if px_in else float(def_px)
        except ValueError:
            print("[!] Invalid price.")
            return False
        if side == "SELL":
            preview = preview_sell_order(token_id, trade_amt, order_type="GTC", limit_price=lim_px, profile_id=active_pid)
        else:
            preview = preview_buy_order(token_id, trade_amt, order_type="GTC", limit_price=lim_px, profile_id=active_pid)
    elif side == "SELL":
        preview = preview_sell_order(token_id, trade_amt, order_type="FAK", profile_id=active_pid)
    else:
        preview = preview_buy_order(
            token_id,
            trade_amt,
            p_norm=row.get("Target"),
            fee_rate=float(row.get("fee_rate", 0.0) or 0.0),
            min_edge=float(getattr(cfg, "MIN_EDGE", 0.0) or 0.0),
            order_type="FAK",
            profile_id=active_pid,
        )

    if not preview.get("ok"):
        print(f"[!] Order rejected in preflight: {preview.get('reason')}")
        input("Press Enter to continue: ")
        return False

    if not conf["has_private_key"]:
        ev_url = row.get("Web Link", "")
        print("\n" + format_browser_ticket(preview, question=q_disp, outcome=pos_side, event_url=ev_url))
        return prompt_browser_confirm_and_sync([ev_url], active_pid, history)

    print("\n--- ORDER SUMMARY ---")
    if side == "SELL":
        print(f"Action    : SELL {preview['shares']:,.2f} {pos_side} shares ({preview['order_type']})")
        print(f"Price     : Est VWAP {preview['est_vwap']*100:.1f}% | Floor/Limit {preview['price_cap']*100:.1f}%")
        print(f"Proceeds  : ~${preview['est_proceeds']:,.2f} USDC")
    else:
        print(f"Action    : BUY ${preview['amount_usdc']:,.2f} of {pos_side} (~{preview['est_shares']:,.2f} sh, {preview['order_type']})")
        print(f"Price     : Est VWAP {preview['est_vwap']*100:.1f}% | Cap/Limit {preview['price_cap']*100:.1f}%")

    conf_str = input("Type 'CONFIRM' to submit live order: ").strip()
    if conf_str != "CONFIRM":
        print("[*] Live order cancelled.")
        input("Press Enter to continue: ")
        return False

    res = execute_order(preview, profile_id=active_pid)
    if res.get("ok"):
        print(f"[+] Order submitted: {res.get('status')} (ID: {res.get('order_id') or 'N/A'})")
        sync_wallet_after_trade(active_pid, history)
        input("Press Enter to continue: ")
        return True
    else:
        print(f"[!] Order failed: {res.get('error')}")
        input("Press Enter to continue: ")
        return False


def _handle_targeted_predict(rows, preselected_idx=None):
    """Launches targeted prediction on a specific table row."""
    if preselected_idx is not None:
        sel_idx = preselected_idx
    else:
        sel_str = input(f"Select row # to predict / update (1-{len(rows)}) [Enter to cancel]: ").strip()
        if not sel_str.isdigit():
            return
        sel_idx = int(sel_str) - 1

    if not (0 <= sel_idx < len(rows)):
        print(f"[!] Row number out of range (1-{len(rows)}).")
        return
    sel_row = rows[sel_idx]
    link = sel_row.get("Web Link", "")
    target_slug = link.split("/event/")[-1].split("?")[0] if "/event/" in link else ""
    target_mid = sel_row.get("Market ID")
    if not target_slug or target_slug == "unknown":
        print(f"[!] Could not extract market slug for: {sel_row.get('Question')}")
        return
    print(f"\n[+] Selected #{sel_idx + 1}: {sel_row['Question'][:70]}")
    from session import run_prediction_session
    run_prediction_session(
        mode="target",
        target_slugs=[target_slug],
        target_market_id=target_mid,
        exclude_mode="none"
    )

def print_main_menu_dashboard(history):
    """Renders the top-level status bar and summary metrics for the main menu without redundancies."""
    base_ego = calculate_base_ego(history, verbose=False)
    counts = get_history_counts(history)
    n_pred = counts["active"]
    n_exit = counts["exited"]
    n_res = counts["resolved"]
    n_skip = counts["skipped"]

    active_pid = history.get("profile_id") or get_active_profile()
    is_wallet = active_pid.startswith("0x")
    ws = history.get("wallet_summary", {})
    cash_val = float(ws.get("cash", 0.0) or 0.0)
    pos_val = float(ws.get("portfolio", 0.0) or 0.0)
    live_bankroll = float(ws.get("bankroll") or cfg.BANKROLL) if isinstance(ws, dict) else cfg.BANKROLL

    alias = history.get("alias", "")
    if not alias:
        if active_pid.startswith("0xbC73"):
            alias = "Main"
        elif active_pid == "local":
            alias = "Paper"

    if is_wallet:
        short_addr = f"{active_pid[:6]}...{active_pid[-4:]}"
        prof_link = ansi_link(f"https://polymarket.com/profile/{active_pid}", short_addr)
        alias_str = f"{alias} [{prof_link}]" if alias else prof_link
        bankroll_add = float(history.get("bankroll_add", 0.0) or history.get("wallet_summary", {}).get("bankroll_add", 0.0) or 0.0)
        add_str = f" | Off-Platform +${bankroll_add:,.2f}" if bankroll_add > 0 else ""
        cap_line = f"  Capital State  : ${live_bankroll:,.2f} Total (Cash ${cash_val:,.2f} | Open Positions ${pos_val:,.2f}{add_str})"
        hdr_tag = f"PROFILE: {alias_str} | Polymarket Live Portfolio"
    else:
        alias_str = f"{alias} ({active_pid})" if alias else active_pid
        cap_line = f"  Capital State  : ${live_bankroll:,.2f} Paper Bankroll (Sandbox)"
        hdr_tag = f"PROFILE: {alias_str} | Paper Trading Sandbox"

    active_eff_kelly = 0.0
    active_edge_w = 0.0
    active_signals = {}
    for _, p in iter_unified_active_entries(history, include_exited=True):
        k = max(0.0, float(p.get("entry_kelly") or p.get("effective_kelly", 0.0) or p.get("kelly", 0.0) or 0.0))
        is_w = (p.get("source") == "wallet")
        cur_p = float(p.get("cur_pm") if p.get("cur_pm") is not None else p.get("pm", 0.5))
        if is_wallet and not is_w:
            live_val, held_mtm_k = 0.0, 0.0
        else:
            live_val, held_mtm_k, _ = calculate_position_holding(p, cur_p, live_bankroll)

        pu_dash = float(p.get("pu", 0.5))
        ego_dash = compute_dynamic_ego(base_ego, float(p.get("u_spread", 0.0) or 0.0))
        spread_val = float(p.get("spread", 0.01) or 0.01)
        half_sp = max(0.0005, spread_val / 2.0)
        f_rate = float(p.get("fee_rate", 0.0) or 0.0)
        is_held_dash = (held_mtm_k > 0 or live_val > 0)
        eval_px_dash = p.get("cur_pm") if is_held_dash else (float(p.get("bestAsk") or p.get("cur_pm") or p.get("pm", 0.5)))
        lm = calculate_logistic_metrics(
            pu_dash, p.get("pm"), ego_dash,
            held_kelly=None, cur_pm=eval_px_dash,
            fee_rate=(0.0 if is_held_dash else f_rate)
        )

        dyn_action = calculate_dynamic_kelly_action(
            win_prob_bal=lm["win_prob_bal"],
            cur_p=cur_p,
            held_frac=held_mtm_k,
            bankroll=live_bankroll,
            fee_rate=f_rate,
            half_spread=half_sp,
            entry_kelly=max(lm["eff_kelly"], k),
            best_bid=p.get("bestBid"),
            best_ask=p.get("bestAsk")
        )

        eff_k = dyn_action["f_star_eff"] if is_held_dash else lm["eff_kelly"]
        act_edge = float(lm["live_active_edge"])

        active_eff_kelly += eff_k
        active_edge_w += eff_k * act_edge

        sig = classify_active_signal(
            is_held_dash,
            dyn_action,
            eff_k,
            act_edge,
            opt_buy_usd=p.get("opt_buy_usd"),
            vwap_edge=p.get("vwap_edge"),
        )
        active_signals[sig] = active_signals.get(sig, 0) + 1

    avg_edge = (active_edge_w / active_eff_kelly) if active_eff_kelly > 0 else 0.0
    sig_summary = " ".join(f"{s}:{cnt}" for s, cnt in sorted(active_signals.items())) if active_signals else "-"

    wins, losses = 0, 0
    pnl_total = 0.0
    for preds in history.get("resolved", {}).values():
        for p in ensure_list(preds):
            if is_wallet and p.get("source") != "wallet" and p.get("realized_pnl") is None:
                continue
            out = p.get("outcome")
            if out == 1.0:
                wins += 1
            elif out == 0.0:
                losses += 1
            pnl_total += float(p.get("realized_pnl", 0.0) or 0.0)
    for preds in history.get("exited", {}).values():
        for p in ensure_list(preds):
            pnl = float(p.get("realized_pnl", 0.0) or 0.0)
            if pnl > 0.0001:
                wins += 1
            elif pnl < -0.0001:
                losses += 1
            pnl_total += pnl

    total_games = wins + losses
    win_pct_str = f" ({wins/total_games*100:.1f}%)" if total_games > 0 else ""
    pnl_str = f"+${pnl_total:,.2f}" if pnl_total >= 0 else f"-${abs(pnl_total):,.2f}"

    box_w = 96
    hdr_visible_len = len(strip_ansi(hdr_tag))
    hdr_pad = max(0, (box_w - hdr_visible_len - 4) // 2)
    hdr_line = " " * hdr_pad + f"[ {hdr_tag} ]"

    res_stats = calculate_prediction_stats(history, sub_mode="resolved", base_ego=base_ego, refresh_meta=False)

    print("\n" + "=" * box_w)
    print(hdr_line)
    print("-" * box_w)
    print(cap_line)
    print(f"  Live Exposure  : {active_eff_kelly*100:.1f}% Eff-Kelly (${active_eff_kelly*live_bankroll:,.2f}) | Avg Edge: {avg_edge*100:+.2f}% | Signals: {sig_summary}")
    print(f"  Track Record   : {wins}W - {losses}L{win_pct_str} | Realized PnL: {pnl_str} | Base Ego: {base_ego:.3f}")
    if res_stats and res_stats.get("res_count", 0) > 0:
        lz = res_stats["res_luck_z"]
        lzw = res_stats["res_luck_z_weighted"]
        cb_u = res_stats["res_calib_bias_u"] * 100.0
        print(f"  Reflection     : Bias vs User: {cb_u:+.1f}% | Luck: {lz:+.2f}s [{classify_luck_z(lz)}] (Wt: {lzw:+.2f}s)")
    print("=" * box_w)
