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
    """Enriches predicted markets with live prices, spread, endDate, and volume from Gamma API in bulk."""
    predicted_dict = history.get("predicted", {})
    if not predicted_dict:
        return
    if base_ego is None:
        base_ego = calculate_base_ego(history, verbose=False)

    m_ids = [str(mid) for mid in predicted_dict.keys() if not str(mid).startswith("custom_")]
    if not m_ids:
        return

    changed = False

    # Auto-consolidate any fragmented/duplicate predictions for the same market while preserving staged tranches
    for mid, p_val in list(predicted_dict.items()):
        preds = ensure_list(p_val)
        if len(preds) > 1:
            wallet_items = [p for p in preds if p.get("source") == "wallet"]
            local_tranches = [p for p in preds if p.get("source") != "wallet" and p.get("is_tranche")]
            local_non_tranches = [p for p in preds if p.get("source") != "wallet" and not p.get("is_tranche")]
            if wallet_items:
                canonical = wallet_items[0]
                for other_w in wallet_items[1:]:
                    canonical["shares"] = round(float(canonical.get("shares", 0.0) or 0.0) + float(other_w.get("shares", 0.0) or 0.0), 4)
                    canonical["cost_basis"] = round(float(canonical.get("cost_basis", 0.0) or 0.0) + float(other_w.get("cost_basis", 0.0) or 0.0), 4)
                non_tranche_pool = wallet_items + local_non_tranches
                latest_p = sorted(non_tranche_pool, key=lambda x: str(x.get("date", "")), reverse=True)[0]
                canonical["pu"] = latest_p.get("pu", canonical.get("pu"))
                canonical["dynamic_ego"] = latest_p.get("dynamic_ego", canonical.get("dynamic_ego"))
                canonical["date"] = latest_p.get("date", canonical.get("date"))
                canonical["edge"] = latest_p.get("edge", canonical.get("edge"))
                canonical["kelly"] = latest_p.get("kelly", canonical.get("kelly"))
                canonical["opt_buy_usd"] = latest_p.get("opt_buy_usd")
                canonical["opt_buy_vwap"] = latest_p.get("opt_buy_vwap")
                canonical["opt_buy_edge"] = latest_p.get("opt_buy_edge")
                new_preds = [canonical] + local_tranches
                if len(new_preds) != len(preds) or len(wallet_items) > 1:
                    predicted_dict[mid] = new_preds
                    changed = True
            elif len(local_non_tranches) > 1:
                latest = sorted(local_non_tranches, key=lambda x: str(x.get("date", "")), reverse=True)[0]
                predicted_dict[mid] = [latest] + local_tranches
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

                    preds = ensure_list(predicted_dict[mid])
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
    if fetch_clob and predicted_dict:
        import concurrent.futures
        from model import calculate_vwap, calculate_optimal_market_buy, calculate_logistic_metrics, calculate_dynamic_kelly_action
        from market import get_market_clob_tokens

        ws_b = history.get("wallet_summary", {})
        live_bankroll = float(ws_b.get("bankroll") or cfg.BANKROLL) if isinstance(ws_b, dict) else cfg.BANKROLL
        min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))

        clob_tasks = []
        for mid, preds_list in predicted_dict.items():
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
                    is_held = (cost_basis > 0 or shares_val > 0 or p.get("source") == "wallet")

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
                                p["vwap"] = valid_asks[0][0]
                                p["slippage"] = 0.0
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

    if r.get("tranche_rows"):
        print("\nTranche Breakdown:")
        for tr in r["tranche_rows"]:
            print(f"  {tr['label']}: {tr['regime']} | Held: {tr['held_str']} | Kelly: {tr['kelly_str']} | Px: {tr['live_str']}")

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

def calculate_prediction_stats(history, sub_mode="predicted", base_ego=None, refresh_meta=True):
    """Calculates weighted portfolio stats: balance percentage, effective N, expected win rate, variance."""
    # Auto-prune closed markets from skipped history in background
    auto_prune_closed_markets(history, force=(sub_mode == "skipped"))

    if refresh_meta and sub_mode in ["predicted", "all"]:
        ensure_predicted_market_metadata(history)
    if base_ego is None:
        base_ego = calculate_base_ego(history, verbose=False)

    preds_list = []
    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        for m_preds in history.get("predicted", {}).values():
            for p in ensure_list(m_preds):
                preds_list.append((p, "active"))
    if sub_mode in ["exited", "all"]:
        for m_ex in history.get("exited", {}).values():
            for p in ensure_list(m_ex):
                preds_list.append((p, "exited"))
    if sub_mode in ["resolved", "all"]:
        for m_res in history.get("resolved", {}).values():
            for p in ensure_list(m_res):
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
        is_exited = (state == "exited") or bool(p.get("exited"))
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
                eff_k_stat = lm["eff_kelly"]
        else:
            kelly = calculate_trade_weight({**p, "exclude_from_ego": False}, bankroll=active_bankroll)
            lm = calculate_logistic_metrics(pu, pm, ego, held_kelly=kelly, cur_pm=cur_pm, fee_rate=f_rate)
            p_edge = float(lm["active_edge"])
            eff_k_stat = lm["eff_kelly"]

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
    total_pnl = sum(x["realized_pnl"] for x in items)
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
        "avg_daily_edge": avg_daily_edge,
        "avg_days": avg_days,
        "avg_apy": avg_apy,
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
        "widths": [3, 34, 26, 12, 7, 7, 8, 4, 6, 6, 10, 6],
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

    # 1. Active / Predicted Rows
    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        profile_id = get_active_profile()
        is_live_wallet_profile = bool(profile_id and str(profile_id).startswith("0x"))

        for market_id, value in history.get("predicted", {}).items():
            preds = ensure_list(value)
            if not preds:
                continue

            tot_held_val = 0.0
            tot_held_k = 0.0
            tot_shares = 0.0
            tot_target_k = 0.0
            has_wallet = any(p.get("source") == "wallet" for p in preds)
            has_local = any(p.get("source") != "wallet" for p in preds)
            tranche_items = []

            for t_idx, pred in enumerate(preds, 1):
                side_val = pred.get("position", "Yes")
                pu_v = float(pred.get("pu", 0.5))
                pm_v = float(pred.get("pm", 0.5))
                cur_pm_v = pred.get("cur_pm")
                k_v = float(pred.get("entry_kelly") or pred.get("effective_kelly", 0.0) or pred.get("kelly", 0.0) or 0.0)
                ego_v = compute_dynamic_ego(base_ego, float(pred.get("u_spread", 0.0) or 0.0))

                cur_p = float(cur_pm_v if cur_pm_v is not None else pm_v)
                is_wallet = (pred.get("source", "local") == "wallet")

                if is_live_wallet_profile and not is_wallet:
                    live_val, held_mtm_k, sh = 0.0, 0.0, 0.0
                else:
                    live_val, held_mtm_k, sh = calculate_position_holding(pred, cur_p, live_bankroll)

                tot_held_val += live_val
                tot_held_k += held_mtm_k
                tot_shares += sh
                tot_target_k += k_v

                spread_val = float(pred.get("spread", 0.01) or 0.01)
                half_sp = max(0.0005, spread_val / 2.0)
                f_rate = float(pred.get("fee_rate", 0.0) or 0.0)
                is_t_held = (held_mtm_k > 0 or live_val > 0)
                eval_t_px = cur_pm_v if is_t_held else (float(pred.get("bestAsk") or cur_pm_v or pm_v))
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=None, cur_pm=eval_t_px, fee_rate=(0.0 if is_t_held else f_rate))
                dyn_action = calculate_dynamic_kelly_action(
                    win_prob_bal=lm["win_prob_bal"],
                    cur_p=cur_p,
                    held_frac=held_mtm_k,
                    bankroll=live_bankroll,
                    fee_rate=f_rate,
                    half_spread=half_sp,
                    entry_kelly=max(lm["eff_kelly"], k_v),
                    best_bid=pred.get("bestBid"),
                    best_ask=pred.get("bestAsk")
                )

                min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))
                end_date_str = pred.get("endDate") or pred.get("end_date")
                _, seconds_left, days_until = parse_market_date(end_date_str)
                active_edge = float(lm["live_active_edge"])

                if held_mtm_k > 0 or live_val > 0:
                    if dyn_action["signal_code"] == "ADD":
                        add_usd = dyn_action["delta_usd"]
                        opt_usd = pred.get("opt_buy_usd")
                        opt_vwap = pred.get("opt_buy_vwap")
                        if opt_usd is not None and opt_usd > 0 and opt_usd < add_usd * 0.95:
                            regime_str = f"ADD +${opt_usd:,.0f} (Mean {opt_vwap*100:.1f}%)"
                        elif opt_usd == 0.0:
                            regime_str = "HOLD (Ask exceeds hurdle)"
                        else:
                            regime_str = dyn_action["action_label"]
                    else:
                        regime_str = dyn_action["action_label"]
                elif active_edge >= min_edge_th:
                    buy_usd = lm["eff_kelly"] * live_bankroll if live_bankroll > 0 else 0.0
                    opt_usd = pred.get("opt_buy_usd")
                    opt_vwap = pred.get("opt_buy_vwap")
                    if opt_usd is not None and opt_usd > 0:
                        if opt_usd < buy_usd * 0.95:
                            regime_str = f"EXEC BUY ${opt_usd:,.0f} (Mean {opt_vwap*100:.1f}%)"
                        else:
                            regime_str = f"EXEC BUY ${buy_usd:,.0f} ({lm['eff_kelly']*100:.1f}%)"
                    elif opt_usd == 0.0:
                        regime_str = "WATCH (Ask exceeds hurdle)"
                    else:
                        regime_str = f"EXEC BUY ${buy_usd:,.0f} ({lm['eff_kelly']*100:.1f}%)"
                elif active_edge > 0:
                    regime_str = f"WATCH (Thin Edge: {active_edge*100:+.1f}%)"
                else:
                    regime_str = f"WATCH (No Edge: {active_edge*100:+.1f}%)"

                src_val = _format_source_tag(pred, alias_tag)
                live_disp = float(cur_pm_v) if cur_pm_v is not None else pm_v
                price_ref = lm["win_prob_m"] if lm["win_prob_m"] > 0 else (cur_p if cur_p > 0 else 0.5)

                if days_until is not None and days_until > 0 and price_ref > 0:
                    daily_edge = active_edge / days_until
                    _, apy_val = calculate_annualized_yield(active_edge, price_ref, days_until)
                else:
                    daily_edge = None
                    apy_val = pred.get("apy")

                vel_d_disp = f"{daily_edge*100:+.2f}%/d" if daily_edge is not None else "-"
                held_t_str = f"${live_val:,.0f} ({held_mtm_k*100:.1f}%)" if live_val > 0 else "-"
                t_label = f" |-- T{t_idx}: {'Wallet Pos' if is_wallet else 'Staged Tranche'}"

                t_eff_k = dyn_action["f_star_eff"] if is_t_held else lm["eff_kelly"]
                tranche_items.append({
                    "label": t_label,
                    "regime": regime_str,
                    "held_str": held_t_str,
                    "kelly_str": f"{t_eff_k*100:.2f}%",
                    "edge_str": f"{active_edge*100:+.1f}%",
                    "vel_str": vel_d_disp,
                    "side": side_val,
                    "live_str": f"{live_disp*100:.1f}%",
                    "p_norm_str": f"{lm['win_prob_bal']*100:.1f}%",
                    "date": str(pred.get("date", "unknown"))[:10],
                    "src": src_val
                })

            primary = preds[0]
            p_side = primary.get("position", "Yes")
            p_pu = float(primary.get("pu", 0.5))
            p_pm = float(primary.get("pm", 0.5))
            p_cur_pm = primary.get("cur_pm")
            p_cur_p = float(p_cur_pm if p_cur_pm is not None else p_pm)
            p_ego = compute_dynamic_ego(base_ego, float(primary.get("u_spread", 0.0) or 0.0))
            is_held_pos = (tot_held_k > 0 or tot_held_val > 0)
            eval_cur_px = p_cur_pm if is_held_pos else (float(primary.get("bestAsk") or p_cur_pm or p_pm))
            p_fee_rate = float(primary.get("fee_rate", 0.0) or 0.0)
            p_lm = calculate_logistic_metrics(p_pu, p_pm, p_ego, held_kelly=None, cur_pm=eval_cur_px, fee_rate=(0.0 if is_held_pos else p_fee_rate))

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

            if has_wallet and has_local:
                src_val = "Main+L" if alias_tag == "Main" else "WAL+LOC"
            elif has_wallet:
                src_val = _format_source_tag(primary, alias_tag)
            else:
                src_val = "LOCAL"

            if tot_held_k > 0 or tot_held_val > 0:
                spread_val = float(primary.get("spread", 0.01) or 0.01)
                half_sp = max(0.0005, spread_val / 2.0)
                dyn_action = calculate_dynamic_kelly_action(
                    win_prob_bal=p_lm["win_prob_bal"],
                    cur_p=p_cur_p,
                    held_frac=tot_held_k,
                    bankroll=live_bankroll,
                    fee_rate=float(primary.get("fee_rate", 0.0) or 0.0),
                    half_spread=half_sp,
                    entry_kelly=max(p_lm["eff_kelly"], tot_target_k),
                    best_bid=primary.get("bestBid"),
                    best_ask=primary.get("bestAsk")
                )
                if dyn_action["signal_code"] == "ADD":
                    add_usd = dyn_action["delta_usd"]
                    opt_usd = primary.get("opt_buy_usd")
                    opt_vwap = primary.get("opt_buy_vwap")
                    if opt_usd is not None and opt_usd > 0 and opt_usd < add_usd * 0.95:
                        regime_str = f"ADD +${opt_usd:,.0f} (Mean {opt_vwap*100:.1f}%)"
                    elif opt_usd == 0.0:
                        regime_str = f"HOLD (Ask exceeds hurdle)"
                    else:
                        regime_str = dyn_action["action_label"]
                else:
                    regime_str = dyn_action["action_label"]
            else:
                min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))
                vwap_edge = primary.get("vwap_edge")
                opt_usd = primary.get("opt_buy_usd")
                opt_vwap = primary.get("opt_buy_vwap")
                if active_edge >= min_edge_th:
                    buy_usd = p_lm["eff_kelly"] * live_bankroll if live_bankroll > 0 else 0.0
                    if opt_usd is not None and opt_usd > 0:
                        if opt_usd < buy_usd * 0.95:
                            regime_str = f"EXEC BUY ${opt_usd:,.0f} (Mean {opt_vwap*100:.1f}%)"
                        else:
                            regime_str = f"EXEC BUY ${buy_usd:,.0f} ({p_lm['eff_kelly']*100:.1f}%)"
                    elif opt_usd == 0.0 or (vwap_edge is not None and vwap_edge <= 0):
                        regime_str = "WATCH (Ask exceeds hurdle)"
                    elif vwap_edge is not None and vwap_edge < min_edge_th:
                        regime_str = f"WATCH (Depth Edge < {min_edge_th*100:.1f}%)"
                    else:
                        regime_str = f"EXEC BUY ${buy_usd:,.0f} ({p_lm['eff_kelly']*100:.1f}%)"
                elif active_edge > 0:
                    regime_str = f"WATCH (Thin Edge: {active_edge*100:+.1f}%)"
                else:
                    regime_str = f"WATCH (No Edge: {active_edge*100:+.1f}%)"

            q_title = primary.get("question", "unknown")
            q_suf = f" ({len(preds)} tranches)" if len(preds) > 1 else ""
            compact_q = format_compact_question(q_title, 34 - len(q_suf)) + q_suf

            eff_target_k = dyn_action["f_star_eff"] if (tot_held_k > 0 or tot_held_val > 0) else p_lm["eff_kelly"]

            rows.append({
                "Mode": "active",
                "Source": src_val,
                "Side": p_side,
                "Market ID": market_id,
                "Date": str(primary.get("date", "unknown"))[:10],
                "Question": compact_q,
                "raw_question": q_title,
                "Web Link": f"https://polymarket.com/event/{primary.get('slug', 'unknown')}",
                "Kelly": eff_target_k,
                "EffKelly": eff_target_k,
                "HeldVal": tot_held_val,
                "HeldFrac": tot_held_k,
                "Shares": tot_shares,
                "Edge": active_edge,
                "DailyEdge": daily_edge,
                "APY": apy_val,
                "DaysUntil": days_until,
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
                "snapshots": primary.get("snapshots", []),
                "tranche_rows": tranche_items if len(preds) > 1 else []
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
        "1": "date", "date": "date",
        "2": "kelly", "kelly": "kelly", "k": "kelly",
        "3": "edge", "edge": "edge", "e": "edge",
        "4": "exit_prox", "exit": "exit_prox", "prox": "exit_prox",
        "5": "velocity", "vel": "velocity", "velocity": "velocity", "v": "velocity", "daily": "velocity",
        "6": "ego_impact", "ego": "ego_impact", "brier": "ego_impact", "b": "ego_impact",
    }
    if sort_by is None:
        sort_by = default_sorts.get(sub_mode, "date")

    if sort_by == "tranches" and sub_mode == "predicted":
        unbundled_rows = []
        for r in rows:
            if r.get("tranche_rows"):
                for t_item in r["tranche_rows"]:
                    unbundled_rows.append({
                        "Mode": "active",
                        "Source": t_item["src"],
                        "Side": t_item["side"],
                        "Market ID": r["Market ID"],
                        "Date": t_item["date"],
                        "Question": f"{r['Question'].split(' (')[0]} ({t_item['label'].split(': ')[-1]})",
                        "Web Link": r["Web Link"],
                        "Kelly": float(t_item["kelly_str"].replace("%", "")) / 100.0,
                        "EffKelly": float(t_item["kelly_str"].replace("%", "")) / 100.0,
                        "HeldVal": float(t_item["held_str"].split(" ")[0].replace("$", "").replace(",", "")) if "$" in t_item["held_str"] else 0.0,
                        "HeldFrac": float(t_item["held_str"].split("(")[1].replace("%)", "")) / 100.0 if "(" in t_item["held_str"] else 0.0,
                        "Edge": float(t_item["edge_str"].replace("%", "")) / 100.0 if "%" in t_item["edge_str"] else 0.0,
                        "VelDStr": t_item["vel_str"],
                        "pu": r.get("pu"),
                        "p_norm": float(t_item["p_norm_str"].replace("%", "")) / 100.0 if "%" in t_item["p_norm_str"] else r.get("p_norm"),
                        "pm": r.get("pm"),
                        "cur_pm": float(t_item["live_str"].replace("%", "")) / 100.0 if "%" in t_item["live_str"] else r.get("cur_pm"),
                        "PnL": None,
                        "Regime": t_item["regime"],
                        "Outcome": ""
                    })
            else:
                unbundled_rows.append(r)
        rows = unbundled_rows

    if sort_by == "date":
        rows.sort(key=lambda x: (x.get("Date", "") if x.get("Date", "") != "unknown" else "", x.get("Kelly") or 0.0), reverse=True)
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

    if sort_by == "ego_impact":
        config = TABLE_CONFIGS["ego_impact"]
    else:
        config = TABLE_CONFIGS.get(sub_mode, TABLE_CONFIGS["all"])
    headers = config["headers"]
    widths = config["widths"]
    aligns = config["aligns"]
    q_idx = config["q_idx"]

    def format_cell(text, width, align):
        return str(text).rjust(width)[:width] if align == "right" else str(text).ljust(width)[:width]

    header_str = " | ".join(format_cell(h, w, a) for h, w, a in zip(headers, widths, aligns))
    separator_str = "-+-".join("-" * w for w in widths)

    print("\n" + "=" * len(header_str))
    print(f" [{sub_mode.upper()} MARKETS - sorted by {sort_by.upper()}] ".center(len(header_str), "="))
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
            vals = [row_num, r["Question"], regime_val, held_str, kelly_str, edge_str, r.get("VelDStr", "-"), side_str, live_str, p_norm_str, r["Date"], src_str]
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

        if sub_mode == "predicted" and r.get("tranche_rows"):
            for tr in r["tranche_rows"]:
                t_vals = [
                    "", tr["label"], tr["regime"], tr["held_str"], tr["kelly_str"],
                    tr["edge_str"], tr["vel_str"], tr["side"], tr["live_str"],
                    tr["p_norm_str"], tr["date"], tr["src"]
                ]
                t_parts = [format_cell(v, w, a) for v, w, a in zip(t_vals, widths, aligns)]
                print(" | ".join(t_parts))

    print("=" * len(header_str) + "\n")
    if sub_mode in ["predicted", "exited", "resolved", "all"]:
        print_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego, refresh_meta=False)

    # Action Bar: Launch Tranche Prediction
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
    range_str = f"({len(rows)})" if len(rows) == 1 else f"(1-{len(rows)})"
    act = input(f"Select row # to {prompt_action} {range_str}{sort_hint}{exp_hint}{live_hint}, 'd <#>' (details){u_hint}{cal_hint} [Enter for Main Menu]: ").strip()
    act_low = act.lower()
    if act_low in ("x", "expand", "exp"):
        return print_review_table(history, sub_mode, expanded=not expanded, sort_by=sort_by, refresh_meta=False)
    elif (
        act_low in ("s", "sort", "date", "kelly", "edge", "exit", "prox", "vel", "velocity", "daily", "brier")
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
                "predicted": ["velocity", "kelly", "edge", "exit_prox", "date"],
                "exited": ["date", "ego_impact", "kelly", "edge"],
                "resolved": ["ego_impact", "date", "kelly", "edge"],
                "all": ["kelly", "velocity", "edge", "ego_impact", "date"],
            }
            seq = cycle_map.get(sub_mode, ["date", "kelly", "edge"])
            next_s = seq[(seq.index(sort_by) + 1) % len(seq)] if sort_by in seq else seq[0]
            vel_h = "  [5] Velocity (Edge/d)" if sub_mode in ("predicted", "all") else ""
            ego_h = "  [6] Ego Impact (Brier)" if sub_mode in ("resolved", "exited", "all") else ""
            s_in = input(f"Sort '{sub_mode}' by: [1] Date  [2] Kelly  [3] Edge  [4] Exit Proximity{vel_h}{ego_h} [Enter -> {next_s.upper()}]: ").strip().lower()
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
                preds = ensure_list(history.get("predicted", {}).get(m_id, []))
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
                    print(f"[!] Market not found in predicted history.")
            else:
                print(f"[!] Row number out of range {range_str}.")
        else:
            print("[!] Usage: 'u <row#>' (e.g. 'u 2')")
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
            _handle_targeted_tranche_predict(rows, preselected_idx=row_num - 1)
            history = load_history()
            return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=(sub_mode in ("predicted", "all")))
        else:
            print(f"[!] Row number out of range {range_str}.")
            return print_review_table(history, sub_mode, expanded=expanded, sort_by=sort_by, refresh_meta=False)

def _handle_targeted_tranche_predict(rows, preselected_idx=None):
    """Launches targeted prediction / tranche addition on a specific table row."""
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
    for preds in history.get("predicted", {}).values():
        for p in ensure_list(preds):
            k = max(0.0, float(p.get("entry_kelly") or p.get("effective_kelly", 0.0) or p.get("kelly", 0.0) or 0.0))
            is_w = (p.get("source") == "wallet")
            cur_p = float(p.get("cur_pm") if p.get("cur_pm") is not None else p.get("pm", 0.5))
            if is_wallet and not is_w:
                live_val, held_mtm_k = 0.0, 0.0
            else:
                live_val, held_mtm_k, _ = calculate_position_holding(p, cur_p, live_bankroll)

            pos_side = p.get("position", "Yes")
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
            delta_k = max(0.0, eff_k - held_mtm_k) if held_mtm_k > 0 else eff_k
            act_edge = float(lm["live_active_edge"])

            active_eff_kelly += eff_k
            active_edge_w += eff_k * act_edge

            min_edge_th = float(getattr(cfg, "MIN_EDGE", 0.01))
            opt_buy_usd = p.get("opt_buy_usd")
            if held_mtm_k > 0 or live_val > 0:
                if dyn_action["signal_code"] == "ADD" and opt_buy_usd == 0.0:
                    sig = "HOLD"
                else:
                    sig = dyn_action["signal_code"]
            elif act_edge >= min_edge_th and (opt_buy_usd is None or opt_buy_usd > 0):
                sig = "BUY"
            else:
                sig = "WATCH"
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

    print("\n" + "=" * box_w)
    print(hdr_line)
    print("-" * box_w)
    print(cap_line)
    print(f"  Live Exposure  : {active_eff_kelly*100:.1f}% Eff-Kelly (${active_eff_kelly*live_bankroll:,.2f}) | Avg Edge: {avg_edge*100:+.2f}% | Signals: {sig_summary}")
    print(f"  Track Record   : {wins}W - {losses}L{win_pct_str} | Realized PnL: {pnl_str} | Base Ego: {base_ego:.3f}")
    print("=" * box_w)
