"""Synchronization of Polymarket wallet positions and automatic market resolution auditing."""
from datetime import datetime, timezone
import time
import config as cfg
from config import api
from utils import ensure_list, parse_outcome_prices, parse_market_date, ansi_link
from model import (
    compute_dynamic_ego, calculate_logistic_metrics, calculate_base_ego,
    calculate_annualized_yield, parse_user_input
)
from market import (
    get_wallet_usdc_balance, get_wallet_portfolio_value,
    resolve_position_market_metadata, determine_redeemable_outcome
)


def import_polymarket_positions(wallet_address, history, auto_sync=False):
    """Synchronizes open, closed, and exited positions from Polymarket data endpoints."""
    saved_wallet = history.get("wallet_address") or cfg.POLYGON_WALLET
    if not wallet_address:
        if auto_sync and saved_wallet:
            wallet_address = saved_wallet
        else:
            prompt_str = f"Enter Polymarket wallet address (0x...) [Press Enter for {saved_wallet}]: " if saved_wallet else "Enter Polymarket wallet address (0x...): "
            user_w = input(prompt_str).strip()
            wallet_address = user_w if user_w else saved_wallet

    if not wallet_address or not wallet_address.startswith("0x"):
        print("[!] Invalid address format.")
        return history

    history["wallet_address"] = wallet_address
    history.setdefault("predicted", {})
    history.setdefault("exited", {})
    history.setdefault("resolved", {})
    history.setdefault("skipped", {})
    profile_url = f"https://polymarket.com/profile/{wallet_address}"
    linked_wallet = ansi_link(profile_url, wallet_address)
    base_ego = calculate_base_ego(history)

    print(f"[*] Syncing Polymarket wallet {linked_wallet}...")

    def _fetch_v2_positions(status_filter):
        pos_list = []
        cursor = None
        max_retries = 3
        while True:
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    c_param = f"&cursor={cursor}" if cursor else ""
                    url = f"https://data-api.polymarket.com/v2/positions?user={wallet_address}&status={status_filter}&limit=100{c_param}"
                    resp = api.get(url, timeout=10).json()
                    raw_list = resp.get("data", []) if isinstance(resp, dict) else (resp if isinstance(resp, list) else [])
                    for p in raw_list:
                        if not isinstance(p, dict):
                            continue
                        c_id = p.get("condition_id") or p.get("conditionId")
                        if not c_id:
                            continue
                        tb = float(p.get("total_size") if p.get("total_size") is not None else p.get("totalBought", 0.0) or 0.0)
                        cur_sz = float(p.get("current_size") if p.get("current_size") is not None else p.get("size", 0.0) or 0.0)
                        sz = cur_sz if status_filter == "OPEN" else (tb if tb > 0 else cur_sz)
                        avg_p = float(p.get("avg_price") if p.get("avg_price") is not None else p.get("avgPrice", 0.5) or 0.5)
                        cur_p = float(p.get("current_price") if p.get("current_price") is not None else p.get("curPrice", avg_p) or avg_p)
                        init_val = float(p.get("total_cost_usdc") if p.get("total_cost_usdc") is not None else p.get("initialValue", sz * avg_p) or (sz * avg_p))
                        cur_val = float(p.get("current_value") if p.get("current_value") is not None else p.get("currentValue", sz * cur_p) or (sz * cur_p))
                        r_pnl = float(p.get("realized_pnl") if p.get("realized_pnl") is not None else p.get("realizedPnl", 0.0) or 0.0)
                        pos_list.append({
                            "conditionId": c_id,
                            "outcome": p.get("outcome", "Yes"),
                            "outcomeIndex": int(p.get("outcome_index") if p.get("outcome_index") is not None else p.get("outcomeIndex", 0) or 0),
                            "avgPrice": avg_p,
                            "curPrice": cur_p,
                            "size": sz,
                            "totalBought": tb,
                            "initialValue": init_val,
                            "currentValue": cur_val,
                            "realizedPnl": r_pnl,
                            "redeemable": bool(p.get("redeemable")),
                            "title": p.get("title") or p.get("question"),
                            "slug": p.get("slug"),
                            "endDate": p.get("end_date") or p.get("endDate"),
                            "status": p.get("status", status_filter),
                            "eventId": p.get("event_id"),
                            "eventSlug": p.get("event_slug"),
                            "timestamp": p.get("timestamp") or 0,
                            "last_event_at": p.get("last_event_at") or p.get("timestamp") or 0,
                            "first_entry_at": p.get("first_entry_at") or 0,
                            "_from_closed": (status_filter == "CLOSED")
                        })
                    pagination = resp.get("pagination", {}) if isinstance(resp, dict) else {}
                    if pagination.get("has_more") and pagination.get("next_cursor"):
                        cursor = pagination["next_cursor"]
                    else:
                        cursor = None
                    success = True
                    break
                except Exception as e:
                    if attempt < max_retries:
                        time.sleep(1.0 * attempt)
                    else:
                        print(f"[!] Error fetching {status_filter} positions after {max_retries} attempts: {e}")
                        return None
            if not success or not cursor:
                break
        return pos_list

    open_positions = _fetch_v2_positions("OPEN")
    if open_positions is None:
        print("[!] Network error fetching open positions from Polymarket. Preserving existing active portfolio state.")
        return history

    closed_positions = _fetch_v2_positions("CLOSED")
    if closed_positions is None:
        closed_positions = []

    open_keys = {((p.get("conditionId") or "").lower(), p.get("outcome")) for p in open_positions if isinstance(p, dict)}
    for cp in closed_positions:
        if not isinstance(cp, dict):
            continue
        tb = float(cp.get("totalBought") or 0.0)
        avg_p = float(cp.get("avgPrice") or 0.5)
        cur_p = float(cp.get("curPrice") if cp.get("curPrice") is not None else avg_p)
        r_pnl = round(float(cp.get("realizedPnl") or 0.0), 2)
        gross_val = round(tb * avg_p, 2)
        if cur_p <= 0.01 and r_pnl < 0:
            cp["initialValue"] = round(abs(r_pnl), 2)
        else:
            cp["initialValue"] = gross_val
        cp["size"] = tb

    positions = list(open_positions) + [
        cp for cp in closed_positions
        if isinstance(cp, dict) and ((cp.get("conditionId") or "").lower(), cp.get("outcome")) not in open_keys
    ]

    trades_by_cond = {}
    try:
        t_cursor = None
        while True:
            c_param = f"&cursor={t_cursor}" if t_cursor else ""
            act_url = f"https://data-api.polymarket.com/v2/trades?user={wallet_address}&limit=500{c_param}"
            act_resp = api.get(act_url, timeout=10).json()
            raw_trades = act_resp.get("data", []) if isinstance(act_resp, dict) else (act_resp if isinstance(act_resp, list) else [])
            for ev in raw_trades:
                if not isinstance(ev, dict):
                    continue
                c_id = ev.get("condition_id") or ev.get("conditionId")
                if not c_id:
                    continue
                c_id_lower = c_id.lower()
                sz = float(ev.get("size") or 0.0)
                px = float(ev.get("price") or 0.5)
                trades_by_cond.setdefault(c_id_lower, []).append({
                    "ts": int(ev.get("timestamp") or 0),
                    "side": str(ev.get("side") or "BUY").upper(),
                    "usdc": round(sz * px, 4),
                    "shares": sz,
                    "price": px
                })
            pagination = act_resp.get("pagination", {}) if isinstance(act_resp, dict) else {}
            if pagination.get("has_more") and pagination.get("next_cursor"):
                t_cursor = pagination["next_cursor"]
            else:
                break
        for c_id in trades_by_cond:
            trades_by_cond[c_id].sort(key=lambda x: x["ts"])
    except Exception:
        trades_by_cond = {}

    print("[*] Calculating Live Cash & Portfolio Values...")
    prev_bankroll = float(history.get("wallet_summary", {}).get("bankroll", cfg.BANKROLL) or cfg.BANKROLL)
    cash_balance = get_wallet_usdc_balance(wallet_address)
    if cash_balance is None:
        cash_balance = history.get("wallet_summary", {}).get("cash", 0.0)
    portfolio_value = get_wallet_portfolio_value(wallet_address, open_positions)

    bankroll_add = float(history.get("bankroll_add", 0.0) or history.get("wallet_summary", {}).get("bankroll_add", 0.0) or 0.0)
    total_bankroll = round(cash_balance + portfolio_value + bankroll_add, 4)
    cfg.BANKROLL = total_bankroll
    history["bankroll_add"] = bankroll_add
    history["wallet_summary"] = {
        "cash": round(cash_balance, 4),
        "portfolio": round(portfolio_value, 4),
        "bankroll_add": bankroll_add,
        "bankroll": total_bankroll,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    }
    add_tag = f" + ${bankroll_add:,.2f} off-platform" if bankroll_add > 0 else ""
    print(f"[*] Wallet Profile Updated: ${cfg.BANKROLL:,.2f} (Cash: ${cash_balance:,.2f} | Positions: ${portfolio_value:,.2f}{add_tag})")

    if not positions:
        print("[*] No positions returned from Polymarket endpoints.")
    else:
        print(f"Found {len(open_positions)} active/unclosed + {len(closed_positions)} closed/redeemed positions. Syncing...")

    imported_predicted = 0
    imported_exited = 0
    imported_resolved = 0
    updated_existing = 0
    event_cache = {}
    wallet_cond_ids = {(p.get("conditionId") or "").lower() for p in open_positions if isinstance(p, dict) and p.get("conditionId")}
    to_import = []

    for pos in positions:
        cond_id = pos.get("conditionId")
        if not cond_id:
            continue
        pos_outcome_str = pos.get("outcome", "Yes")
        try:
            pos_outcome_idx = int(pos.get("outcomeIndex", 0))
        except Exception:
            pos_outcome_idx = 0

        avg_price = round(float(pos.get("avgPrice", 0.5)), 6)
        cur_price = round(float(pos.get("curPrice", avg_price)), 6)
        size = float(pos.get("size", 0.0))
        initial_val = round(float(pos.get("initialValue", size * avg_price)), 4)
        realized_pnl = round(float(pos.get("realizedPnl", pos.get("cashPnl", 0.0)) or 0.0), 4)
        is_from_closed = bool(pos.get("_from_closed"))

        already_in_resolved = False
        for m_id, res_val in list(history.get("resolved", {}).items()):
            res_list = ensure_list(res_val)
            matching_res = [
                r for r in res_list
                if r.get("conditionId") == cond_id
                and (r.get("source") != "wallet" or r.get("position", pos_outcome_str) == pos_outcome_str)
            ]
            if matching_res:
                already_in_resolved = True
                for r in matching_res:
                    if r.get("realized_pnl") is None and realized_pnl != 0.0:
                        r["realized_pnl"] = realized_pnl
                    updated_existing += 1
                break
        if already_in_resolved:
            continue

        if is_from_closed:
            tot_bought = float(pos.get("totalBought") or size or 0.0)
            calc_exit_price = max(0.0001, min(0.9999, round(avg_price + (realized_pnl / tot_bought), 4))) if tot_bought > 0 else cur_price
            last_evt_ts = int(pos.get("last_event_at") or pos.get("timestamp") or 0)
            exit_date_str = datetime.fromtimestamp(last_evt_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if last_evt_ts > 0 else None

            already_in_exited = False
            for m_id, ex_val in list(history.get("exited", {}).items()):
                ex_list = ensure_list(ex_val)
                matching_ex = [
                    e for e in ex_list
                    if e.get("conditionId") == cond_id
                    and (e.get("source") != "wallet" or e.get("position", pos_outcome_str) == pos_outcome_str)
                ]
                if not matching_ex:
                    continue
                already_in_exited = True
                for e in matching_ex:
                    e["cur_pm"] = cur_price
                    e["realized_pnl"] = realized_pnl
                    if tot_bought > 0 and avg_price > 0:
                        e["entry_cost"] = round(tot_bought * avg_price, 4)
                        e["total_size"] = tot_bought
                        e["avg_price"] = avg_price
                        e["entry_price"] = avg_price
                        if len(e.get("snapshots", [])) <= 1:
                            e["pm"] = avg_price
                            if e.get("snapshots"):
                                e["snapshots"][0]["pm"] = avg_price
                                e["snapshots"][0]["entry_price"] = avg_price
                    e["entry_bankroll"] = float(e.get("entry_bankroll") or total_bankroll)
                    if calc_exit_price is not None:
                        e["exit_price"] = calc_exit_price
                    if exit_date_str:
                        e["date"] = exit_date_str
                    updated_existing += 1
                break
            if already_in_exited:
                continue

        m, market_id, question, slug, end_date = resolve_position_market_metadata(pos, event_cache)
        cond_trades = trades_by_cond.get((cond_id or "").lower(), [])
        latest_trade_ts = max((t["ts"] for t in cond_trades), default=int(pos.get("timestamp") or 0))

        is_market_resolved = bool(m and (m.get("closed") or m.get("umaResolutionStatus") == "resolved"))
        is_resolved_pos = bool(pos.get("redeemable")) or is_market_resolved or (is_from_closed and (cur_price >= 0.99 or cur_price <= 0.01))
        is_exited_open_pos = is_from_closed and not is_resolved_pos
        pos["redeemable"] = is_resolved_pos
        pos["_exited"] = is_exited_open_pos

        _, _, days_until = parse_market_date(end_date)
        days_until = days_until or 0.0

        already_stored = False
        c_id_lower = (cond_id or "").lower()
        for m_id in list(history.get("predicted", {}).keys()):
            preds_list = ensure_list(history["predicted"][m_id])
            matching_preds = [
                p for p in preds_list
                if (p.get("conditionId") or "").lower() == c_id_lower and (
                    p.get("source") != "wallet" or p.get("position", pos_outcome_str) == pos_outcome_str
                )
            ]
            if not matching_preds:
                continue

            already_stored = True
            wallet_preds = [p for p in matching_preds if p.get("source") == "wallet"]
            local_preds = [p for p in matching_preds if p.get("source") != "wallet"]

            prev_wallet_usdc = sum(
                float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                for wp in wallet_preds
            )
            max_synced_ts = max((int(wp.get("last_synced_ts", 0)) for wp in wallet_preds), default=0)
            if wallet_preds and max_synced_ts == 0 and abs(initial_val - prev_wallet_usdc) <= 0.05:
                max_synced_ts = latest_trade_ts

            unseen_trades = [t for t in cond_trades if t["ts"] > max_synced_ts] if wallet_preds else cond_trades
            if unseen_trades:
                existing_rem_usdc = prev_wallet_usdc
                prev_sh_est = sum(
                    float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                    / max(0.001, float(wp.get("tranche_pm", wp.get("pm", avg_price))))
                    for wp in wallet_preds
                )
                existing_rem_shares = prev_sh_est
                new_buys = []
                for tr in unseen_trades:
                    if tr["side"] == "BUY":
                        new_buys.append({
                            "usdc": tr["usdc"],
                            "shares": tr["shares"],
                            "pm": tr["price"],
                            "ts": tr["ts"]
                        })
                    elif tr["side"] == "SELL":
                        sell_sh = tr["shares"]
                        tot_sh = existing_rem_shares + sum(b["shares"] for b in new_buys)
                        if tot_sh > 0:
                            keep_ratio = max(0.0, (tot_sh - sell_sh) / tot_sh)
                            existing_rem_shares *= keep_ratio
                            existing_rem_usdc *= keep_ratio
                            for b in new_buys:
                                b["shares"] *= keep_ratio
                                b["usdc"] *= keep_ratio
                delta_usdc = round(sum(b["usdc"] for b in new_buys), 4) if wallet_preds else initial_val
                new_buy_sh = sum(b["shares"] for b in new_buys)
                if new_buy_sh > 0 and sum(b["usdc"] for b in new_buys) > 0:
                    implied_pm = round(sum(b["usdc"] for b in new_buys) / new_buy_sh, 6)
                    delta_pm = implied_pm if 0.001 <= implied_pm <= 0.999 else avg_price
                else:
                    delta_pm = avg_price
            else:
                delta_usdc = round(max(0.0, initial_val - prev_wallet_usdc), 4)
                if not wallet_preds:
                    delta_pm = avg_price
                else:
                    prev_shares = sum(
                        float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                        / max(0.001, float(wp.get("tranche_pm", wp.get("pm", avg_price))))
                        for wp in wallet_preds
                    )
                    delta_shares = size - prev_shares
                    if delta_shares > 0 and delta_usdc > 0:
                        implied_pm = round(delta_usdc / delta_shares, 6)
                        delta_pm = implied_pm if 0.001 <= implied_pm <= 0.999 else avg_price
                    else:
                        delta_pm = avg_price

            unmatched_delta_for_import = False
            if local_preds and delta_usdc > 0.05:
                default_idx = min(
                    range(1, len(local_preds) + 1),
                    key=lambda idx_i: abs(float(local_preds[idx_i - 1].get("kelly", 0.0)) * cfg.BANKROLL - delta_usdc)
                )
                web_link = f"https://polymarket.com/event/{slug}"
                linked_q = ansi_link(web_link, question)
                if auto_sync or len(local_preds) == 1:
                    sel_idx = default_idx
                    print(f"\n[SYNC TRANCHE] Automatically matched on-chain execution (+${delta_usdc:,.2f}) with local prediction: {linked_q}")
                else:
                    print(f"\n[SYNC TRANCHE] On-chain execution detected (+${delta_usdc:,.2f} @ {delta_pm*100:.2f}%) for: {linked_q}")
                    for i_lp, lp_item in enumerate(local_preds, 1):
                        rec_tag = " (Recommended match)" if i_lp == default_idx else ""
                        t_usd = float(lp_item.get("kelly", 0.0)) * cfg.BANKROLL
                        print(f"  [{i_lp}] Date: {lp_item.get('date')} | pu: {float(lp_item.get('pu', 0.5))*100:.1f}% | pm: {float(lp_item.get('pm', 0.5))*100:.1f}% | Target: ${t_usd:,.2f} ({float(lp_item.get('kelly', 0))*100:.2f}%){rec_tag}")
                    print("  [0] None (keep local prediction(s) unexecuted & import on-chain trade separately)")
                    sel_raw = input(f"  Select which local prediction was executed [Enter for {default_idx}]: ").strip()
                    try:
                        sel_idx = int(sel_raw) if sel_raw else default_idx
                    except ValueError:
                        sel_idx = default_idx

                if 1 <= sel_idx <= len(local_preds):
                    chosen_lp = local_preds.pop(sel_idx - 1)
                    if wallet_preds:
                        prev_tot = sum(float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) for wp in wallet_preds)
                        for wp in wallet_preds:
                            w_sh = (float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) / prev_tot) if prev_tot > 0 else (1.0 / len(wallet_preds))
                            wp["cost_basis"] = round(max(0.0, initial_val - delta_usdc) * w_sh, 4)
                    ego_lp = float(chosen_lp.get("dynamic_ego", base_ego))
                    lp_fee = float(chosen_lp.get("fee_rate", 0.0) or (m.get("fee_rate", 0.0) if m else 0.0) or 0.0)
                    orig_lp_pm = float(chosen_lp.get("pm", delta_pm))
                    lp_sp = float(chosen_lp.get("spread", 0.01) or (m.get("spread", 0.01) if m else 0.01) or 0.01)
                    ref_lp_pm = orig_lp_pm if abs(orig_lp_pm - delta_pm) <= max(0.02, lp_sp + 0.005) else delta_pm
                    lm_lp = calculate_logistic_metrics(chosen_lp.get("pu", ref_lp_pm), ref_lp_pm, ego_lp, fee_rate=lp_fee)
                    tot_b_lp = float(pos.get("totalBought") or size or 0.0)
                    chosen_lp["question"] = question
                    chosen_lp["slug"] = slug
                    chosen_lp["tranche_pm"] = delta_pm
                    chosen_lp["entry_price"] = delta_pm
                    chosen_lp["avg_price"] = avg_price
                    chosen_lp["pm"] = ref_lp_pm
                    chosen_lp["cur_pm"] = cur_price
                    chosen_lp["fee_rate"] = lp_fee
                    chosen_lp["cost_basis"] = round(0.0 if is_exited_open_pos else delta_usdc, 4)
                    chosen_lp["entry_cost"] = round(tot_b_lp * avg_price, 4) if (tot_b_lp > 0 and avg_price > 0) else round(delta_usdc, 4)
                    chosen_lp["total_size"] = round(tot_b_lp if tot_b_lp > 0 else size, 4)
                    chosen_lp["entry_bankroll"] = float(chosen_lp.get("entry_bankroll") or total_bankroll)
                    chosen_lp["shares"] = round(size, 4) if not is_exited_open_pos else 0.0
                    chosen_lp["current_value"] = round(float(pos.get("currentValue", size * cur_price)), 4) if not is_exited_open_pos else 0.0
                    chosen_lp["realized_pnl"] = realized_pnl
                    chosen_lp["kelly"] = lm_lp["eff_model_kelly"]
                    chosen_lp["raw_kelly"] = lm_lp["model_kelly"]
                    chosen_lp["entry_kelly"] = max(float(chosen_lp.get("entry_kelly", 0.0) or 0.0), lm_lp["eff_model_kelly"])
                    chosen_lp["edge"] = lm_lp["active_edge"]
                    _, chosen_lp["apy"] = calculate_annualized_yield(chosen_lp["edge"], lm_lp["win_prob_m"], days_until)
                    chosen_lp["position"] = pos.get("outcome", "Yes")
                    chosen_lp["outcomeIndex"] = pos_outcome_idx
                    chosen_lp["source"] = "wallet"
                    chosen_lp["wallet"] = wallet_address
                    chosen_lp["exited"] = is_exited_open_pos
                    chosen_lp["is_watch"] = False
                    if chosen_lp.get("action") in ("WATCH", "PASS", "NONE", "THIN_EDGE", None):
                        chosen_lp["action"] = "HOLD"
                    if latest_trade_ts > 0:
                        chosen_lp["last_synced_ts"] = latest_trade_ts
                    chosen_lp["snapshots"] = [{
                        "ts": int(latest_trade_ts if latest_trade_ts > 0 else time.time()),
                        "date": chosen_lp.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                        "pu": float(chosen_lp.get("pu", ref_lp_pm)),
                        "pm": ref_lp_pm,
                        "entry_price": delta_pm,
                        "kelly": chosen_lp["kelly"],
                        "entry_kelly": chosen_lp["entry_kelly"],
                        "raw_kelly": chosen_lp["raw_kelly"],
                        "edge": chosen_lp["edge"],
                        "apy": chosen_lp["apy"],
                        "dynamic_ego": float(chosen_lp.get("dynamic_ego", base_ego)),
                        "action": chosen_lp.get("action", "HOLD"),
                        "source": "wallet",
                    }]
                    wallet_preds.append(chosen_lp)
                    if is_exited_open_pos:
                        imported_exited += 1
                    else:
                        imported_predicted += 1
                else:
                    unmatched_delta_for_import = True
                    if wallet_preds:
                        prev_tot = sum(float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) for wp in wallet_preds)
                        for wp in wallet_preds:
                            w_sh = (float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) / prev_tot) if prev_tot > 0 else (1.0 / len(wallet_preds))
                            wp["cost_basis"] = round(max(0.0, initial_val - delta_usdc) * w_sh, 4)

            wallet_target_usdc = max(0.0, initial_val - delta_usdc) if unmatched_delta_for_import else initial_val

            if wallet_preds:
                if not local_preds and delta_usdc > 0.05:
                    q_disp = (question[:42] + '...') if len(question) > 45 else question
                    print(f"[*] Absorbed tranche +${delta_usdc:,.2f} into existing position: {q_disp}")
                raw_weights = [
                    float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                    for wp in wallet_preds
                ]
                total_w = sum(raw_weights)
                tot_b_wp = float(pos.get("totalBought") or size or 0.0)
                for wp, w_i in zip(wallet_preds, raw_weights):
                    share = (w_i / total_w) if total_w > 0 else (1.0 / len(wallet_preds))
                    tranche_pm = float(wp.get("tranche_pm", wp.get("pm", avg_price)))
                    ego_wp = float(wp.get("dynamic_ego", base_ego))
                    f_rate_wp = float(wp.get("fee_rate", 0.0) or (m.get("fee_rate", 0.0) if m else 0.0) or 0.0)
                    absorbed_add = (not local_preds and delta_usdc > 0.05 and avg_price > 0)
                    if absorbed_add:
                        wp["entry_price"] = avg_price
                        prev_ec = float(wp.get("entry_cost") or max(0.0, (wallet_target_usdc - delta_usdc) * share))
                        prev_eb = float(wp.get("entry_bankroll") or prev_bankroll or total_bankroll)
                        add_sh_usdc = delta_usdc * share
                        wp["entry_bankroll"] = round((prev_ec * prev_eb + add_sh_usdc * total_bankroll) / max(0.01, prev_ec + add_sh_usdc), 4)
                    else:
                        wp["entry_price"] = wp.get("entry_price") or tranche_pm
                        wp["entry_bankroll"] = float(wp.get("entry_bankroll") or total_bankroll)
                    decision_pm = float(wp.get("pm", tranche_pm))
                    lm_wp = calculate_logistic_metrics(wp.get("pu", decision_pm), decision_pm, ego_wp, fee_rate=f_rate_wp)
                    base_ek = float(wp.get("entry_kelly", 0.0) or wp.get("kelly", 0.0) or lm_wp["eff_model_kelly"])
                    if absorbed_add:
                        lm_avg = calculate_logistic_metrics(wp.get("pu", avg_price), avg_price, ego_wp, fee_rate=f_rate_wp)
                        lm_del = calculate_logistic_metrics(wp.get("pu", delta_pm), delta_pm, ego_wp, fee_rate=f_rate_wp)
                        wp["entry_kelly"] = max(base_ek, lm_avg["eff_model_kelly"], lm_del["eff_model_kelly"])
                    else:
                        wp["entry_kelly"] = base_ek
                    wp["question"] = question
                    wp["slug"] = slug
                    wp["pm"] = decision_pm
                    wp["cur_pm"] = cur_price
                    wp["fee_rate"] = f_rate_wp
                    wp["cost_basis"] = round(0.0 if is_exited_open_pos else (wallet_target_usdc * share), 4)
                    wp["entry_cost"] = round((tot_b_wp * avg_price if (tot_b_wp > 0 and avg_price > 0) else wallet_target_usdc) * share, 4)
                    wp["total_size"] = round((tot_b_wp if tot_b_wp > 0 else size) * share, 4)
                    wp["avg_price"] = avg_price
                    wp["shares"] = round(size * share, 4) if not is_exited_open_pos else 0.0
                    wp["current_value"] = round(float(pos.get("currentValue", size * cur_price)) * share, 4) if not is_exited_open_pos else 0.0
                    wp["realized_pnl"] = round(realized_pnl * share, 4)
                    wp["kelly"] = lm_wp["eff_model_kelly"]
                    wp["raw_kelly"] = lm_wp["model_kelly"]
                    wp["edge"] = lm_wp["active_edge"]
                    _, wp["apy"] = calculate_annualized_yield(wp["edge"], lm_wp["win_prob_m"], days_until)
                    wp["position"] = pos.get("outcome", "Yes")
                    wp["outcomeIndex"] = pos_outcome_idx
                    wp["source"] = "wallet"
                    wp["wallet"] = wallet_address
                    wp["exited"] = is_exited_open_pos
                    wp["is_watch"] = False
                    if wp.get("action") in ("WATCH", "PASS", "NONE", "THIN_EDGE", None):
                        wp["action"] = "HOLD"
                    if end_date or (m and m.get("endDate")):
                        wp["endDate"] = end_date or m.get("endDate")
                    if m and m.get("volume") is not None:
                        wp["volume"] = float(m.get("volume") or 0.0)
                    if m and m.get("volume24hr") is not None:
                        wp["volume24hr"] = float(m.get("volume24hr") or 0.0)
                    if latest_trade_ts > 0:
                        wp["last_synced_ts"] = latest_trade_ts
                    updated_existing += 1

            for lp in local_preds:
                lp["source"] = "local"

            if unmatched_delta_for_import:
                to_import.append({
                    "pos": pos,
                    "m": m,
                    "market_id": market_id,
                    "question": question,
                    "slug": slug,
                    "end_date": end_date,
                    "avg_price": delta_pm,
                    "cur_price": cur_price,
                    "size": size,
                    "initial_val": delta_usdc,
                    "realized_pnl": realized_pnl,
                    "days_until": days_until,
                    "cond_id": cond_id,
                    "latest_trade_ts": latest_trade_ts
                })

            target_key = market_id
            if m_id != target_key:
                history["predicted"].pop(m_id, None)
                if preds_list:
                    history["predicted"].setdefault(target_key, []).extend(preds_list)
            else:
                if preds_list:
                    history["predicted"][target_key] = preds_list
                else:
                    history["predicted"].pop(target_key, None)

            if pos.get("redeemable"):
                outcome_val = determine_redeemable_outcome(pos, m)
                cur_preds = ensure_list(history["predicted"].get(target_key, []))
                to_resolve = [
                    p for p in cur_preds
                    if p.get("conditionId") == cond_id and p.get("source") == "wallet"
                    and p.get("position", pos_outcome_str) == pos_outcome_str
                ]
                remaining_pred = [p for p in cur_preds if p not in to_resolve]
                if remaining_pred:
                    history["predicted"][target_key] = remaining_pred
                else:
                    history["predicted"].pop(target_key, None)
                for p in to_resolve:
                    tot_b_res = float(pos.get("totalBought") or size or 0.0)
                    if tot_b_res > 0 and avg_price > 0:
                        p["entry_cost"] = round(tot_b_res * avg_price, 4)
                        p["total_size"] = tot_b_res
                        p["avg_price"] = avg_price
                    elif not p.get("entry_cost") and float(p.get("cost_basis", 0.0) or 0.0) > 0:
                        p["entry_cost"] = float(p.get("cost_basis", 0.0))
                    p["entry_bankroll"] = float(p.get("entry_bankroll") or total_bankroll)
                    p["outcome"] = outcome_val
                    p["position"] = pos_outcome_str
                    p["outcomeIndex"] = pos_outcome_idx
                    p["realized_pnl"] = realized_pnl
                    p["source"] = "wallet"
                if to_resolve:
                    history["resolved"].setdefault(target_key, []).extend(to_resolve)
            elif is_exited_open_pos:
                cur_preds = ensure_list(history["predicted"].get(target_key, []))
                to_exit = [
                    p for p in cur_preds
                    if p.get("conditionId") == cond_id and p.get("source") == "wallet"
                    and p.get("position", pos_outcome_str) == pos_outcome_str
                ]
                remaining_pred = [p for p in cur_preds if p not in to_exit]
                if remaining_pred:
                    history["predicted"][target_key] = remaining_pred
                else:
                    history["predicted"].pop(target_key, None)
                tot_bought = float(pos.get("totalBought") or size or 0.0)
                calc_exit_price = max(0.0001, min(0.9999, round(avg_price + (realized_pnl / tot_bought), 4))) if tot_bought > 0 else cur_price
                last_evt_ts = int(pos.get("last_event_at") or pos.get("timestamp") or latest_trade_ts or 0)
                exit_date_str = datetime.fromtimestamp(last_evt_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if last_evt_ts > 0 else None
                for p in to_exit:
                    if tot_bought > 0 and avg_price > 0:
                        p["entry_cost"] = round(tot_bought * avg_price, 4)
                        p["total_size"] = tot_bought
                        p["avg_price"] = avg_price
                        p["entry_price"] = avg_price
                        if len(p.get("snapshots", [])) <= 1:
                            p["pm"] = avg_price
                            if p.get("snapshots"):
                                p["snapshots"][0]["pm"] = avg_price
                                p["snapshots"][0]["entry_price"] = avg_price
                    elif not p.get("entry_cost") and float(p.get("cost_basis", 0.0) or 0.0) > 0:
                        p["entry_cost"] = float(p.get("cost_basis", 0.0))
                    p["entry_bankroll"] = float(p.get("entry_bankroll") or total_bankroll)
                    p["exited"] = True
                    p["cost_basis"] = 0.0
                    p["realized_pnl"] = realized_pnl
                    if calc_exit_price is not None:
                        p["exit_price"] = calc_exit_price
                    if exit_date_str:
                        p["date"] = exit_date_str
                    c_id_l = (p.get("conditionId") or cond_id or "").lower()
                    sells = [t for t in trades_by_cond.get(c_id_l, []) if t["side"] == "SELL"]
                    if sells:
                        p["cur_pm"] = round(sells[-1]["price"], 6)
                        p["last_synced_ts"] = max(int(p.get("last_synced_ts", 0)), sells[-1]["ts"])
                if to_exit:
                    history["exited"].setdefault(target_key, []).extend(to_exit)
            break

        # If open wallet position was previously archived in exited, revive it automatically
        if not already_stored and not is_from_closed:
            for m_id in list(history.get("exited", {}).keys()):
                ex_list = ensure_list(history["exited"][m_id])
                revive_matches = [
                    e for e in ex_list
                    if (e.get("conditionId") or "").lower() == c_id_lower
                    and (e.get("source") != "wallet" or e.get("position", pos_outcome_str) == pos_outcome_str)
                ]
                if revive_matches:
                    rem_ex = [e for e in ex_list if e not in revive_matches]
                    if rem_ex:
                        history["exited"][m_id] = rem_ex
                    else:
                        history["exited"].pop(m_id, None)
                    for e in revive_matches:
                        e["exited"] = False
                        e["source"] = "wallet"
                        e["cost_basis"] = initial_val
                        e["shares"] = size
                        e["current_value"] = float(pos.get("currentValue", size * cur_price))
                    history.setdefault("predicted", {}).setdefault(m_id, []).extend(revive_matches)
                    already_stored = True
                    break

        if already_stored:
            continue

        to_import.append({
            "pos": pos,
            "m": m,
            "market_id": market_id,
            "question": question,
            "slug": slug,
            "end_date": end_date,
            "avg_price": avg_price,
            "cur_price": cur_price,
            "size": size,
            "initial_val": initial_val,
            "realized_pnl": realized_pnl,
            "days_until": days_until,
            "cond_id": cond_id,
            "latest_trade_ts": latest_trade_ts
        })

    closed_lookup = {
        ((cp.get("conditionId") or "").lower(), cp.get("outcome", "Yes")): cp
        for cp in closed_positions if isinstance(cp, dict)
    }
    for m_id in list(history.get("predicted", {}).keys()):
        preds_list = ensure_list(history["predicted"][m_id])
        kept_preds = []
        moved_exits = []
        for p in preds_list:
            c_id = p.get("conditionId")
            if p.get("source") == "wallet" and c_id and (c_id or "").lower() not in wallet_cond_ids:
                if not p.get("entry_cost") and float(p.get("cost_basis", 0.0) or 0.0) > 0:
                    p["entry_cost"] = float(p.get("cost_basis", 0.0))
                p["entry_bankroll"] = float(p.get("entry_bankroll") or total_bankroll)
                p["cost_basis"] = 0.0
                p["exited"] = True
                cp_match = closed_lookup.get(((c_id or "").lower(), p.get("position", "Yes")))
                if cp_match:
                    p["realized_pnl"] = round(float(cp_match.get("realizedPnl", 0.0) or 0.0), 4)
                    cp_tb = float(cp_match.get("totalBought") or cp_match.get("size") or 0.0)
                    cp_avg = float(cp_match.get("avgPrice") or p.get("pm", 0.5))
                    if cp_tb > 0 and cp_avg > 0:
                        p["entry_cost"] = round(cp_tb * cp_avg, 4)
                        p["total_size"] = cp_tb
                        p["avg_price"] = cp_avg
                        p["entry_price"] = cp_avg
                        if len(p.get("snapshots", [])) <= 1:
                            p["pm"] = cp_avg
                            if p.get("snapshots"):
                                p["snapshots"][0]["pm"] = cp_avg
                                p["snapshots"][0]["entry_price"] = cp_avg
                        p["exit_price"] = max(0.0001, min(0.9999, round(cp_avg + (p["realized_pnl"] / cp_tb), 4)))
                    cp_ts = int(cp_match.get("last_event_at") or cp_match.get("timestamp") or 0)
                    if cp_ts > 0:
                        p["date"] = datetime.fromtimestamp(cp_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                sells = [t for t in trades_by_cond.get((c_id or "").lower(), []) if t["side"] == "SELL"]
                if sells:
                    p["cur_pm"] = round(sells[-1]["price"], 6)
                    p["last_synced_ts"] = max(int(p.get("last_synced_ts", 0)), sells[-1]["ts"])
                moved_exits.append(p)
            else:
                kept_preds.append(p)
        if moved_exits:
            history["exited"].setdefault(m_id, []).extend(moved_exits)
            if kept_preds:
                history["predicted"][m_id] = kept_preds
            else:
                history["predicted"].pop(m_id, None)

    if auto_sync and to_import:
        n_res_imp = sum(1 for x in to_import if x["pos"].get("redeemable"))
        n_ex_imp = sum(1 for x in to_import if x["pos"].get("_exited"))
        n_open_imp = len(to_import) - n_res_imp - n_ex_imp
        imp_ans = input(f"\n[?] Found {len(to_import)} unimported wallet position(s) ({n_open_imp} active, {n_ex_imp} exited, {n_res_imp} resolved). Import now? (y/n) [Enter=n]: ").strip().lower()
        if imp_ans != 'y':
            to_import = []

    imported_records = {}
    idx = 0
    skip_all_remaining = False
    while idx < len(to_import):
        item = to_import[idx]
        pos = item["pos"]
        m = item["m"]
        market_id = item["market_id"]
        question = item["question"]
        slug = item["slug"]
        end_date = item["end_date"]
        avg_price = item["avg_price"]
        cur_price = item.get("cur_price", avg_price)
        size = item["size"]
        initial_val = item["initial_val"]
        realized_pnl = item.get("realized_pnl", 0.0)
        days_until = item["days_until"]
        cond_id = item["cond_id"]
        latest_trade_ts = item.get("latest_trade_ts", 0)

        outcome_str = pos.get("outcome", "Yes")
        try:
            outcome_idx = int(pos.get("outcomeIndex", 0))
        except Exception:
            outcome_idx = 0
        web_link = f"https://polymarket.com/event/{slug}"
        linked_question = ansi_link(web_link, question)

        if not skip_all_remaining:
            progress_str = f" {idx + 1}/{len(to_import)}" if len(to_import) > 1 else ""
            print(f"\n[IMPORT{progress_str}] Question: {linked_question}")
            if pos.get("redeemable"):
                res_out = determine_redeemable_outcome(pos, m)
                pos_res = "WON" if res_out == 1.0 else ("LOST" if res_out == 0.0 else "50/50")
                status_str = f"RESOLVED ({pos_res} | PnL: ${realized_pnl:+,.2f})"
            elif pos.get("_exited"):
                status_str = f"EXITED EARLY (Sold | PnL: ${realized_pnl:+,.2f} | Live: {cur_price*100:.1f}%)"
            else:
                status_str = f"OPEN (Live: {cur_price*100:.1f}% | PnL: ${realized_pnl:+,.2f})"

            print(f"         Position: {outcome_str}  |  Entry Price (pm): {avg_price*100:.1f}%  |  Cost Basis: ${initial_val:,.2f}  |  Status: {status_str}")

        go_back = False
        while True:
            if skip_all_remaining:
                lower, upper = avg_price, avg_price
                break
            pu_input = input("         Enter your subjective probability bounds % (e.g., 60-70 or 65), 'b' (back), 'a' (skip all), 'q' (quit), or enter to skip: ").strip()
            if pu_input.lower() in ('b', 'back', 'r', 'redo'):
                if idx > 0:
                    idx -= 1
                    if idx in imported_records:
                        prev_cat, prev_mid, prev_entry = imported_records.pop(idx)
                        if prev_mid in history.get(prev_cat, {}) and prev_entry in history[prev_cat][prev_mid]:
                            history[prev_cat][prev_mid].remove(prev_entry)
                            if not history[prev_cat][prev_mid]:
                                history[prev_cat].pop(prev_mid, None)
                        if prev_cat == "resolved":
                            imported_resolved = max(0, imported_resolved - 1)
                        elif prev_cat == "exited":
                            imported_exited = max(0, imported_exited - 1)
                        else:
                            imported_predicted = max(0, imported_predicted - 1)
                    print("\n[!] Rolling back to previous question...")
                    go_back = True
                    break
                else:
                    print("         [!] Cannot go back. This is the first question to import.\n")
                    continue
            elif pu_input.lower() in ('q', 'quit'):
                print("\n[!] Quitting position import...")
                idx = len(to_import)
                break
            elif pu_input.lower() == 'a':
                skip_all_remaining = True
                lower, upper = avg_price, avg_price
                break
            elif pu_input.lower() in ('s', 'skip') or not pu_input:
                lower, upper = avg_price, avg_price
                break
            else:
                try:
                    lower, upper = parse_user_input(pu_input)
                    break
                except ValueError as e:
                    print(f"         [!] Error: {e}. Please try again (or enter 'b' to go back, enter to skip).")
                except Exception as e:
                    print(f"         [!] Error: {e}. Please try again.")

        if go_back:
            continue
        if idx >= len(to_import):
            break

        pu_val = round((lower + upper) / 2.0, 6)
        dynamic_ego = compute_dynamic_ego(base_ego, upper - lower)
        m_fee = (m.get("feeSchedule") or {}).get("rate") if m else None
        if m_fee is None and m:
            f = float(m.get("makerBaseFee") or 0.0)
            m_fee = f / 1e18 if f > 1 else f
        m_fee = float(m_fee or 0.0)
        lm = calculate_logistic_metrics(pu_val, avg_price, dynamic_ego, fee_rate=m_fee)
        edge = lm["active_edge"]
        kelly_val = lm["eff_model_kelly"]
        raw_kelly_val = lm["model_kelly"]
        _, apy = calculate_annualized_yield(edge, lm["win_prob_m"], days_until)

        is_exited_flag = bool(pos.get("_exited"))
        exit_ts = pos.get("last_event_at") or latest_trade_ts
        if is_exited_flag and exit_ts and int(exit_ts) > 0:
            rec_date = datetime.fromtimestamp(int(exit_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        elif pos.get("redeemable") and exit_ts and int(exit_ts) > 0:
            rec_date = datetime.fromtimestamp(int(exit_ts), tz=timezone.utc).strftime("%Y-%m-%d")
        elif not is_exited_flag and not pos.get("redeemable"):
            entry_ts = pos.get("first_entry_at") or pos.get("timestamp") or latest_trade_ts
            if entry_ts and int(entry_ts) > 0:
                rec_date = datetime.fromtimestamp(int(entry_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            else:
                rec_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        else:
            rec_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        tot_sz_import = float(pos.get("totalBought") or size or 0.0)
        calc_import_exit_px = None
        if is_exited_flag:
            if tot_sz_import > 0:
                calc_import_exit_px = max(0.0001, min(0.9999, round(avg_price + (realized_pnl / tot_sz_import), 4)))
            else:
                calc_import_exit_px = cur_price

        pred_entry = {
            "question": question,
            "slug": slug,
            "date": rec_date,
            "pu": pu_val,
            "pm": avg_price,
            "cur_pm": cur_price,
            "snapshots": [{
                "ts": int(latest_trade_ts if latest_trade_ts > 0 else time.time()),
                "date": rec_date,
                "pu": pu_val,
                "pm": avg_price,
                "entry_price": avg_price,
                "kelly": kelly_val,
                "entry_kelly": kelly_val,
                "raw_kelly": raw_kelly_val,
                "edge": edge,
                "apy": apy,
                "dynamic_ego": dynamic_ego,
                "action": "HOLD",
                "source": "wallet",
            }],
            "dynamic_ego": dynamic_ego,
            "kelly": kelly_val,
            "entry_kelly": kelly_val,
            "raw_kelly": raw_kelly_val,
            "cost_basis": round(0.0 if is_exited_flag else initial_val, 4),
            "entry_cost": round(tot_sz_import * avg_price, 4) if (tot_sz_import > 0 and avg_price > 0) else round(initial_val, 4),
            "total_size": round(tot_sz_import, 4) if tot_sz_import > 0 else round(size, 4),
            "avg_price": avg_price,
            "entry_bankroll": total_bankroll,
            "shares": round(size, 4) if not is_exited_flag else 0.0,
            "current_value": round(float(pos.get("currentValue", size * cur_price)), 4) if not is_exited_flag else 0.0,
            "realized_pnl": realized_pnl,
            "exit_price": calc_import_exit_px if is_exited_flag else None,
            "edge": edge,
            "apy": apy,
            "conditionId": cond_id,
            "position": outcome_str,
            "outcomeIndex": outcome_idx,
            "source": "wallet",
            "wallet": wallet_address,
            "exited": is_exited_flag,
            "endDate": end_date or (m.get("endDate") if m else None),
            "volume": float(m.get("volume") or 0.0) if m else 0.0,
            "volume24hr": float(m.get("volume24hr") or 0.0) if m else 0.0,
            "fee_rate": m_fee,
            "spread": float(m.get("spread") or 0.01) if m else 0.01,
            "clobTokenIds": m.get("clobTokenIds") if m else None
        }
        if latest_trade_ts > 0:
            pred_entry["last_synced_ts"] = latest_trade_ts

        if pos.get("redeemable"):
            pred_entry["outcome"] = determine_redeemable_outcome(pos, m)
            history["resolved"].setdefault(market_id, []).append(pred_entry)
            imported_resolved += 1
            imported_records[idx] = ("resolved", market_id, pred_entry)
        elif is_exited_flag:
            history["exited"].setdefault(market_id, []).append(pred_entry)
            imported_exited += 1
            imported_records[idx] = ("exited", market_id, pred_entry)
        else:
            history["predicted"].setdefault(market_id, []).append(pred_entry)
            imported_predicted += 1
            imported_records[idx] = ("predicted", market_id, pred_entry)

        idx += 1

    local_unexecuted = []
    for m_id in list(history.get("predicted", {}).keys()):
        preds_list = ensure_list(history["predicted"][m_id])
        for p in preds_list:
            if p.get("source") != "wallet":
                p["source"] = "local"
                local_unexecuted.append(p)

    print(f"\n[+] Sync complete: {imported_predicted} active imported, {imported_exited} exited imported, {imported_resolved} resolved imported, {updated_existing} existing synced.")
    if local_unexecuted:
        print(f"[!] Unexecuted Local Predictions (Not in Polymarket Wallet): {len(local_unexecuted)}")
        for lp in local_unexecuted:
            q_short = lp.get("question", "Unknown")
            if len(q_short) > 65:
                q_short = q_short[:62] + "..."
            lp_link = f"https://polymarket.com/event/{lp.get('slug', 'unknown')}"
            q_linked = ansi_link(lp_link, q_short)
            k_pct = float(lp.get("kelly", 0.0)) * 100.0
            pu_pct = float(lp.get("pu", 0.5)) * 100.0
            pm_pct = float(lp.get("pm", 0.5)) * 100.0
            print(f"    - [LOCAL] ({lp.get('date', 'unknown')}) {q_linked} | pu: {pu_pct:.1f}% | pm: {pm_pct:.1f}% | Target Kelly: {k_pct:.2f}%")
    return history


def update_resolutions(history):
    """Audits unresolved predictions in predicted bucket against Gamma API."""
    resolved_count = 0
    history.setdefault("predicted", {})
    history.setdefault("resolved", {})
    market_ids = list(history.get("predicted", {}).keys())
    if market_ids:
        print(f"[*] Auditing {len(market_ids)} unresolved market(s) for resolution...")

    for market_id in market_ids:
        if str(market_id).startswith("custom_"):
            continue
        try:
            resp = api.get(f"{cfg.GAMMA_API}/markets/{market_id}", timeout=10).json()
            is_closed = resp.get("closed", False)
            uma_resolved = resp.get("umaResolutionStatus") == "resolved"

            if is_closed or uma_resolved:
                prices = parse_outcome_prices(resp)
                if len(prices) < 2:
                    prices = [0.5, 0.5]

                pred_list = ensure_list(history["predicted"][market_id])
                history["resolved"].setdefault(market_id, [])
                is_wallet_prof = bool(str(history.get("profile_id") or history.get("wallet_address") or "").startswith("0x"))

                for pred_data in pred_list:
                    cb = float(pred_data.get("cost_basis", 0.0) or pred_data.get("entry_cost", 0.0) or 0.0)
                    sh = float(pred_data.get("shares", 0.0) or 0.0)
                    is_unexec_local = (is_wallet_prof and pred_data.get("source") != "wallet" and cb <= 0.0 and sh <= 0.0)
                    is_watch_pred = bool(pred_data.get("is_watch") or pred_data.get("action") == "WATCH")
                    pos_pos = str(pred_data.get("position", "Yes")).lower()
                    out_idx = 1 if pos_pos in ("no", "1") else 0
                    if "outcomeIndex" in pred_data:
                        try:
                            out_idx = int(pred_data.get("outcomeIndex", out_idx))
                        except Exception:
                            pass
                    if len(prices) > out_idx:
                        outcome = 1.0 if prices[out_idx] >= 0.99 else (0.0 if prices[out_idx] <= 0.01 else 0.5)
                    else:
                        outcome = 0.5
                    resolved_entry = dict(pred_data)
                    resolved_entry["outcome"] = outcome
                    if is_unexec_local or is_watch_pred:
                        resolved_entry["realized_pnl"] = None
                    elif resolved_entry.get("realized_pnl") is None or float(resolved_entry.get("realized_pnl", 0.0) or 0.0) == 0.0:
                        ws = history.get("wallet_summary", {}) if isinstance(history, dict) else {}
                        active_b = float(ws.get("bankroll") or cfg.BANKROLL) if isinstance(ws, dict) else cfg.BANKROLL
                        k_eval = float(resolved_entry.get("entry_kelly") or resolved_entry.get("kelly", 0.0) or 0.0)
                        if not is_wallet_prof and cb == 0.0 and k_eval > 0:
                            cb = k_eval * active_b
                            if sh == 0.0 and resolved_entry.get("pm", 0.0):
                                sh = cb / float(resolved_entry["pm"])
                        if sh > 0 or cb > 0:
                            if outcome == 1.0:
                                resolved_entry["realized_pnl"] = round(sh * 1.0 - cb, 2)
                            elif outcome == 0.0:
                                resolved_entry["realized_pnl"] = round(-cb, 2)
                            elif outcome == 0.5:
                                resolved_entry["realized_pnl"] = round(sh * 0.5 - cb, 2)
                    history["resolved"][market_id].append(resolved_entry)
                if not history["resolved"][market_id]:
                    history["resolved"].pop(market_id, None)

                del history["predicted"][market_id]
                resolved_count += 1
        except Exception:
            continue

    if resolved_count > 0:
        print(f"[+] Auto-resolved and scored {resolved_count} closed market(s).")
    return history
