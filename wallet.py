"""Synchronization of Polymarket wallet positions and automatic market resolution auditing."""
import concurrent.futures
from datetime import datetime, timezone
import time
import config as cfg
from config import api
from utils import (
    ensure_list,
    parse_outcome_prices,
    parse_market_date,
    ansi_link,
    get_effective_bankroll,
    calculate_implied_exit_price,
    calculate_resolved_pnl,
    apply_closed_exit_fields,
)
from model import (
    compute_dynamic_ego,
    calculate_logistic_metrics,
    calculate_base_ego,
    calculate_annualized_yield,
    parse_user_input,
)
from market import (
    get_wallet_usdc_balance,
    get_wallet_portfolio_value,
    resolve_position_market_metadata,
    determine_redeemable_outcome,
)


def fetch_v2_positions(wallet_address, status_filter):
    """Fetches paginated OPEN or CLOSED positions from Polymarket v2 positions endpoint."""
    pos_list = []
    cursor = None
    max_retries = 3
    while True:
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                c_param = f"&cursor={cursor}" if cursor else ""
                url = (
                    f"https://data-api.polymarket.com/v2/positions"
                    f"?user={wallet_address}&status={status_filter}&limit=100{c_param}"
                )
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
                        "_from_closed": (status_filter == "CLOSED"),
                    })
                pagination = (resp.get("pagination") or {}) if isinstance(resp, dict) else {}
                cursor = pagination["next_cursor"] if (pagination.get("has_more") and pagination.get("next_cursor")) else None
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


def fetch_wallet_trades(wallet_address):
    """Fetches paginated wallet trade events grouped by lowercase conditionId and sorted by timestamp."""
    tb_map = {}
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
            tb_map.setdefault(c_id_lower, []).append({
                "ts": int(ev.get("timestamp") or 0),
                "side": str(ev.get("side") or "BUY").upper(),
                "usdc": round(sz * px, 4),
                "shares": sz,
                "price": px,
            })
        pagination = (act_resp.get("pagination") or {}) if isinstance(act_resp, dict) else {}
        if pagination.get("has_more") and pagination.get("next_cursor"):
            t_cursor = pagination["next_cursor"]
        else:
            break
    for c_id in tb_map:
        tb_map[c_id].sort(key=lambda x: x["ts"])
    return tb_map


def normalize_closed_positions(open_positions, closed_positions):
    """Normalizes closed position cost bases and deduplicates against currently open positions."""
    open_keys = {
        ((p.get("conditionId") or "").lower(), p.get("outcome"))
        for p in open_positions if isinstance(p, dict)
    }
    for cp in closed_positions:
        if not isinstance(cp, dict):
            continue
        tb = float(cp.get("totalBought") or 0.0)
        avg_p = float(cp.get("avgPrice") or 0.5)
        cur_p = float(cp.get("curPrice") if cp.get("curPrice") is not None else avg_p)
        r_pnl = round(float(cp.get("realizedPnl") or 0.0), 2)
        gross_val = round(tb * avg_p, 2)
        cp["initialValue"] = round(abs(r_pnl), 2) if (cur_p <= 0.01 and r_pnl < 0) else gross_val
        cp["size"] = tb

    return list(open_positions) + [
        cp for cp in closed_positions
        if isinstance(cp, dict) and ((cp.get("conditionId") or "").lower(), cp.get("outcome")) not in open_keys
    ]


def index_history_by_condition(history):
    """Builds lowercase conditionId -> [(market_id, entry)] maps for resolved, exited, and predicted."""
    def _index_bucket(bucket_name):
        by_cond = {}
        for m_id, val in history.get(bucket_name, {}).items():
            for entry in ensure_list(val):
                cid = (entry.get("conditionId") or "").lower()
                if cid:
                    by_cond.setdefault(cid, []).append((str(m_id), entry))
        return by_cond

    return _index_bucket("resolved"), _index_bucket("exited"), _index_bucket("predicted")


def reconcile_resolved_match(entries, pos_realized_pnl, bankroll):
    """Backfills missing realized_pnl on already-resolved records matched by conditionId."""
    updated = 0
    for _, r in entries:
        if r.get("realized_pnl") is None and not r.get("is_watch"):
            new_pnl = calculate_resolved_pnl(
                r,
                r.get("outcome"),
                bankroll=bankroll,
                allow_kelly_fallback=False,
                fallback_api_pnl=pos_realized_pnl,
                preserve_exited=False,
            )
            if new_pnl is not None:
                r["realized_pnl"] = new_pnl
        updated += 1
    return updated


def reconcile_exited_match(entries, pos, avg_price, cur_price, size, realized_pnl, total_bankroll):
    """Updates already-exited records matched from closed positions without extra network queries."""
    tot_bought = float(pos.get("totalBought") or size or 0.0)
    calc_exit_price = calculate_implied_exit_price(avg_price, realized_pnl, tot_bought, fallback_price=cur_price)
    last_evt_ts = int(pos.get("last_event_at") or pos.get("timestamp") or 0)
    exit_date_str = datetime.fromtimestamp(last_evt_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if last_evt_ts > 0 else None
    updated = 0
    for _, e in entries:
        apply_closed_exit_fields(
            e,
            avg_price=avg_price,
            total_size=tot_bought,
            realized_pnl=realized_pnl,
            cur_price=cur_price,
            exit_price=calc_exit_price,
            exit_date=exit_date_str,
            bankroll=total_bankroll,
            mark_exited=False,
        )
        updated += 1
    return updated


def compute_tranche_delta(wallet_preds, cond_trades, initial_val, size, avg_price, prev_bankroll, latest_trade_ts):
    """Computes incremental (delta_usdc, delta_pm) since the last synced wallet tranche."""
    prev_wallet_usdc = sum(
        float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
        for wp in wallet_preds
    )
    max_synced_ts = max((int(wp.get("last_synced_ts", 0)) for wp in wallet_preds), default=0)
    if wallet_preds and max_synced_ts == 0 and abs(initial_val - prev_wallet_usdc) <= 0.05:
        max_synced_ts = latest_trade_ts

    unseen_trades = [t for t in cond_trades if t["ts"] > max_synced_ts] if wallet_preds else cond_trades
    if unseen_trades:
        existing_rem_shares = sum(
            float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
            / max(0.001, float(wp.get("tranche_pm", wp.get("pm", avg_price))))
            for wp in wallet_preds
        )
        new_buys = []
        for tr in unseen_trades:
            if tr["side"] == "BUY":
                new_buys.append({
                    "usdc": tr["usdc"],
                    "shares": tr["shares"],
                    "pm": tr["price"],
                    "ts": tr["ts"],
                })
            elif tr["side"] == "SELL":
                sell_sh = tr["shares"]
                tot_sh = existing_rem_shares + sum(b["shares"] for b in new_buys)
                if tot_sh > 0:
                    keep_ratio = max(0.0, (tot_sh - sell_sh) / tot_sh)
                    existing_rem_shares *= keep_ratio
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
        return delta_usdc, delta_pm

    delta_usdc = round(max(0.0, initial_val - prev_wallet_usdc), 4)
    if not wallet_preds:
        return delta_usdc, avg_price

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
    return delta_usdc, delta_pm


def _rebalance_wallet_costs(wallet_preds, initial_val, delta_usdc, prev_bankroll):
    """Re-allocates prior wallet tranche cost_basis after reserving delta_usdc for a new tranche."""
    if not wallet_preds:
        return
    prev_tot = sum(float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) for wp in wallet_preds)
    rem_usdc = max(0.0, initial_val - delta_usdc)
    for wp in wallet_preds:
        w_sh = (float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll)) / prev_tot) if prev_tot > 0 else (1.0 / len(wallet_preds))
        wp["cost_basis"] = round(rem_usdc * w_sh, 4)


def _prompt_local_tranche_match(local_preds, delta_usdc, delta_pm, slug, question, auto_sync):
    """Asks (or auto-selects) which local prediction corresponds to an on-chain tranche."""
    default_idx = min(
        range(1, len(local_preds) + 1),
        key=lambda idx_i: abs(float(local_preds[idx_i - 1].get("kelly", 0.0)) * cfg.BANKROLL - delta_usdc),
    )
    web_link = f"https://polymarket.com/event/{slug}"
    linked_q = ansi_link(web_link, question)
    if auto_sync or len(local_preds) == 1:
        print(f"\n[SYNC TRANCHE] Automatically matched on-chain execution (+${delta_usdc:,.2f}) with local prediction: {linked_q}")
        return default_idx

    print(f"\n[SYNC TRANCHE] On-chain execution detected (+${delta_usdc:,.2f} @ {delta_pm*100:.2f}%) for: {linked_q}")
    for i_lp, lp_item in enumerate(local_preds, 1):
        rec_tag = " (Recommended match)" if i_lp == default_idx else ""
        t_usd = float(lp_item.get("kelly", 0.0)) * cfg.BANKROLL
        print(
            f"  [{i_lp}] Date: {lp_item.get('date')} | pu: {float(lp_item.get('pu', 0.5))*100:.1f}% "
            f"| pm: {float(lp_item.get('pm', 0.5))*100:.1f}% | Target: ${t_usd:,.2f} "
            f"({float(lp_item.get('kelly', 0))*100:.2f}%){rec_tag}"
        )
    print("  [0] None (keep local prediction(s) unexecuted & import on-chain trade separately)")
    sel_raw = input(f"  Select which local prediction was executed [Enter for {default_idx}]: ").strip()
    try:
        return int(sel_raw) if sel_raw else default_idx
    except ValueError:
        return default_idx


def _apply_local_tranche_match(
    chosen_lp, pos, m, delta_usdc, delta_pm, avg_price, cur_price, size,
    realized_pnl, days_until, base_ego, total_bankroll, wallet_address,
    pos_outcome_idx, is_exited_open_pos, latest_trade_ts, question, slug,
):
    """Promotes a matched local prediction into an active/closed wallet tranche record."""
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


def _update_wallet_tranches(
    wallet_preds, pos, m, wallet_target_usdc, delta_usdc, delta_pm, avg_price,
    cur_price, size, realized_pnl, days_until, base_ego, prev_bankroll,
    total_bankroll, wallet_address, pos_outcome_idx, is_exited_open_pos,
    latest_trade_ts, question, slug, end_date, has_local_preds,
):
    """Updates all existing wallet tranches proportionally with live position values and metrics."""
    if not has_local_preds and delta_usdc > 0.05:
        q_disp = (question[:42] + "...") if len(question) > 45 else question
        print(f"[*] Absorbed tranche +${delta_usdc:,.2f} into existing position: {q_disp}")

    raw_weights = [
        float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
        for wp in wallet_preds
    ]
    total_w = sum(raw_weights)
    tot_b_wp = float(pos.get("totalBought") or size or 0.0)
    updated = 0
    for wp, w_i in zip(wallet_preds, raw_weights):
        share = (w_i / total_w) if total_w > 0 else (1.0 / len(wallet_preds))
        tranche_pm = float(wp.get("tranche_pm", wp.get("pm", avg_price)))
        ego_wp = float(wp.get("dynamic_ego", base_ego))
        f_rate_wp = float(wp.get("fee_rate", 0.0) or (m.get("fee_rate", 0.0) if m else 0.0) or 0.0)
        absorbed_add = (not has_local_preds and delta_usdc > 0.05 and avg_price > 0)
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
        updated += 1
    return updated


def _transition_matched_market(
    history, target_key, pos, m, cond_id, pos_outcome_str, pos_outcome_idx,
    avg_price, cur_price, size, realized_pnl, total_bankroll, latest_trade_ts,
    is_exited_open_pos, trades_by_cond, event_cache,
):
    """Moves newly redeemable or exited wallet positions out of predicted into resolved or exited."""
    if pos.get("redeemable"):
        if m is None:
            m, _, _, _, _ = resolve_position_market_metadata(pos, event_cache)
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

        tot_b_res = float(pos.get("totalBought") or size or 0.0)
        for p in to_resolve:
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
            history["resolved"][target_key] = ensure_list(history["resolved"].get(target_key, [])) + to_resolve
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
        calc_exit_price = calculate_implied_exit_price(avg_price, realized_pnl, tot_bought, fallback_price=cur_price)
        last_evt_ts = int(pos.get("last_event_at") or pos.get("timestamp") or latest_trade_ts or 0)
        exit_date_str = datetime.fromtimestamp(last_evt_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if last_evt_ts > 0 else None
        for p in to_exit:
            apply_closed_exit_fields(
                p,
                avg_price=avg_price,
                total_size=tot_bought,
                realized_pnl=realized_pnl,
                exit_price=calc_exit_price,
                exit_date=exit_date_str,
                bankroll=total_bankroll,
                mark_exited=True,
            )
            c_id_l = (p.get("conditionId") or cond_id or "").lower()
            sells = [t for t in trades_by_cond.get(c_id_l, []) if t["side"] == "SELL"]
            if sells:
                p["cur_pm"] = round(sells[-1]["price"], 6)
                p["last_synced_ts"] = max(int(p.get("last_synced_ts", 0)), sells[-1]["ts"])
        if to_exit:
            history["exited"][target_key] = ensure_list(history["exited"].get(target_key, [])) + to_exit


def _revive_exited_position(history, c_id_lower, pos, pos_outcome_str, initial_val, size, cur_price):
    """Revives a previously exited position back into predicted if reopened on-chain."""
    for m_id in list(history.get("exited", {}).keys()):
        ex_list = ensure_list(history["exited"][m_id])
        revive_matches = [
            e for e in ex_list
            if (e.get("conditionId") or "").lower() == c_id_lower
            and (e.get("source") != "wallet" or e.get("position", pos_outcome_str) == pos_outcome_str)
        ]
        if not revive_matches:
            continue
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
        pred_bucket = history.setdefault("predicted", {})
        pred_bucket[m_id] = ensure_list(pred_bucket.get(m_id, [])) + revive_matches
        return True
    return False


def _archive_missing_wallet_positions(history, closed_positions, wallet_cond_ids, trades_by_cond, total_bankroll):
    """Moves active wallet positions no longer present in OPEN endpoints into exited."""
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
                cp_match = closed_lookup.get(((c_id or "").lower(), p.get("position", "Yes")))
                if cp_match:
                    cp_pnl = round(float(cp_match.get("realizedPnl", 0.0) or 0.0), 4)
                    cp_tb = float(cp_match.get("totalBought") or cp_match.get("size") or 0.0)
                    cp_avg = float(cp_match.get("avgPrice") or p.get("pm", 0.5))
                    cp_exit_px = calculate_implied_exit_price(cp_avg, cp_pnl, cp_tb)
                    cp_ts = int(cp_match.get("last_event_at") or cp_match.get("timestamp") or 0)
                    cp_date = datetime.fromtimestamp(cp_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if cp_ts > 0 else None
                    apply_closed_exit_fields(
                        p,
                        avg_price=cp_avg,
                        total_size=cp_tb,
                        realized_pnl=cp_pnl,
                        exit_price=cp_exit_px,
                        exit_date=cp_date,
                        bankroll=total_bankroll,
                        mark_exited=True,
                    )
                else:
                    apply_closed_exit_fields(p, bankroll=total_bankroll, mark_exited=True)

                sells = [t for t in trades_by_cond.get((c_id or "").lower(), []) if t["side"] == "SELL"]
                if sells:
                    p["cur_pm"] = round(sells[-1]["price"], 6)
                    p["last_synced_ts"] = max(int(p.get("last_synced_ts", 0)), sells[-1]["ts"])
                moved_exits.append(p)
            else:
                kept_preds.append(p)

        if moved_exits:
            history["exited"][m_id] = ensure_list(history["exited"].get(m_id, [])) + moved_exits
            if kept_preds:
                history["predicted"][m_id] = kept_preds
            else:
                history["predicted"].pop(m_id, None)


def _format_import_record_date(pos, is_exited_flag, latest_trade_ts):
    """Chooses the canonical UTC timestamp string for an imported position."""
    exit_ts = pos.get("last_event_at") or latest_trade_ts
    if is_exited_flag and exit_ts and int(exit_ts) > 0:
        return datetime.fromtimestamp(int(exit_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    if pos.get("redeemable") and exit_ts and int(exit_ts) > 0:
        return datetime.fromtimestamp(int(exit_ts), tz=timezone.utc).strftime("%Y-%m-%d")
    if not is_exited_flag and not pos.get("redeemable"):
        entry_ts = pos.get("first_entry_at") or pos.get("timestamp") or latest_trade_ts
        if entry_ts and int(entry_ts) > 0:
            return datetime.fromtimestamp(int(entry_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _build_wallet_entry(item, lower, upper, base_ego, total_bankroll, wallet_address, unforecasted=False):
    """Constructs a canonical wallet prediction record for import into history."""
    pos = item["pos"]
    m = item["m"]
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
    rec_date = _format_import_record_date(pos, is_exited_flag, latest_trade_ts)
    tot_sz_import = float(pos.get("totalBought") or size or 0.0)
    calc_import_exit_px = (
        calculate_implied_exit_price(avg_price, realized_pnl, tot_sz_import, fallback_price=cur_price)
        if is_exited_flag else None
    )

    pred_entry = {
        "question": item["question"],
        "slug": item["slug"],
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
        "unforecasted": bool(unforecasted),
        "endDate": item["end_date"] or (m.get("endDate") if m else None),
        "volume": float(m.get("volume") or 0.0) if m else 0.0,
        "volume24hr": float(m.get("volume24hr") or 0.0) if m else 0.0,
        "fee_rate": m_fee,
        "spread": float(m.get("spread") or 0.01) if m else 0.01,
        "clobTokenIds": m.get("clobTokenIds") if m else None,
    }
    if unforecasted:
        pred_entry["exclude_from_ego"] = True
    if latest_trade_ts > 0:
        pred_entry["last_synced_ts"] = latest_trade_ts
    return pred_entry


def _prompt_and_import_new_positions(history, to_import, wallet_address, base_ego, total_bankroll, auto_sync):
    """Interactively imports newly discovered wallet positions with back/skip-all/quit support."""
    if auto_sync and to_import:
        n_res_imp = sum(1 for x in to_import if x["pos"].get("redeemable"))
        n_ex_imp = sum(1 for x in to_import if x["pos"].get("_exited"))
        n_open_imp = len(to_import) - n_res_imp - n_ex_imp
        imp_ans = input(
            f"\n[?] Found {len(to_import)} unimported wallet position(s) "
            f"({n_open_imp} active, {n_ex_imp} exited, {n_res_imp} resolved). Import now? (y/n) [Enter=n]: "
        ).strip().lower()
        if imp_ans != "y":
            return 0, 0, 0

    imported_predicted = 0
    imported_exited = 0
    imported_resolved = 0
    imported_records = {}
    idx = 0
    skip_all_remaining = False

    while idx < len(to_import):
        item = to_import[idx]
        pos = item["pos"]
        m = item["m"]
        market_id = item["market_id"]
        avg_price = item["avg_price"]
        cur_price = item.get("cur_price", avg_price)
        initial_val = item["initial_val"]
        realized_pnl = item.get("realized_pnl", 0.0)
        outcome_str = pos.get("outcome", "Yes")
        linked_question = ansi_link(f"https://polymarket.com/event/{item['slug']}", item["question"])

        if pos.get("redeemable"):
            res_out = determine_redeemable_outcome(pos, m)
            pos_res = "WON" if res_out == 1.0 else ("LOST" if res_out == 0.0 else "50/50")
            status_str = f"RESOLVED ({pos_res} | PnL: ${realized_pnl:+,.2f})"
        elif pos.get("_exited"):
            status_str = f"EXITED EARLY (Sold | PnL: ${realized_pnl:+,.2f} | Live: {cur_price*100:.1f}%)"
        else:
            status_str = f"OPEN (Live: {cur_price*100:.1f}% | PnL: ${realized_pnl:+,.2f})"

        if not skip_all_remaining:
            progress_str = f" {idx + 1}/{len(to_import)}" if len(to_import) > 1 else ""
            print(f"\n[IMPORT{progress_str}] Question: {linked_question}")
            print(f"         Position: {outcome_str}  |  Cost Basis: ${initial_val:,.2f}  (Non-spoiler: pm & outcome hidden until pu)")

        go_back = False
        unforecasted = False
        while True:
            if skip_all_remaining:
                lower, upper = avg_price, avg_price
                unforecasted = True
                break
            pu_input = input(
                "         Enter your subjective probability bounds % (e.g., 60-70 or 65), "
                "'b' (back), 'a' (skip all), 'q' (quit), or enter to skip: "
            ).strip()
            cmd = pu_input.lower()
            if cmd in ("b", "back", "r", "redo"):
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
                print("         [!] Cannot go back. This is the first question to import.\n")
                continue
            if cmd in ("q", "quit"):
                print("\n[!] Quitting position import...")
                idx = len(to_import)
                break
            if cmd == "a":
                skip_all_remaining = True
                lower, upper = avg_price, avg_price
                unforecasted = True
                break
            if cmd in ("s", "skip") or not pu_input:
                lower, upper = avg_price, avg_price
                unforecasted = True
                break
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

        if not skip_all_remaining:
            print(f"         Reveal       : Entry Price (pm): {avg_price*100:.1f}%  |  Status: {status_str}")

        pred_entry = _build_wallet_entry(
            item, lower, upper, base_ego, total_bankroll, wallet_address, unforecasted=unforecasted
        )
        if pos.get("redeemable"):
            pred_entry["outcome"] = determine_redeemable_outcome(pos, m)
            history["resolved"].setdefault(market_id, []).append(pred_entry)
            imported_resolved += 1
            imported_records[idx] = ("resolved", market_id, pred_entry)
        elif pred_entry["exited"]:
            history["exited"].setdefault(market_id, []).append(pred_entry)
            imported_exited += 1
            imported_records[idx] = ("exited", market_id, pred_entry)
        else:
            history["predicted"].setdefault(market_id, []).append(pred_entry)
            imported_predicted += 1
            imported_records[idx] = ("predicted", market_id, pred_entry)
        idx += 1

    return imported_predicted, imported_exited, imported_resolved


def import_polymarket_positions(wallet_address, history, auto_sync=False):
    """Synchronizes open, closed, and exited positions from Polymarket data endpoints."""
    saved_wallet = history.get("wallet_address") or cfg.POLYGON_WALLET
    if not wallet_address:
        if auto_sync and saved_wallet:
            wallet_address = saved_wallet
        else:
            prompt_str = (
                f"Enter Polymarket wallet address (0x...) [Press Enter for {saved_wallet}]: "
                if saved_wallet else "Enter Polymarket wallet address (0x...): "
            )
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
    linked_wallet = ansi_link(f"https://polymarket.com/profile/{wallet_address}", wallet_address)
    base_ego = calculate_base_ego(history)
    print(f"[*] Syncing Polymarket wallet {linked_wallet}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        fut_open = executor.submit(fetch_v2_positions, wallet_address, "OPEN")
        fut_closed = executor.submit(fetch_v2_positions, wallet_address, "CLOSED")
        fut_trades = executor.submit(fetch_wallet_trades, wallet_address)
        fut_cash = executor.submit(get_wallet_usdc_balance, wallet_address)

        open_positions = fut_open.result()
        closed_positions = fut_closed.result()
        try:
            trades_by_cond = fut_trades.result()
        except Exception:
            trades_by_cond = {}
        cash_balance = fut_cash.result()

    if open_positions is None:
        print("[!] Network error fetching open positions from Polymarket. Preserving existing active portfolio state.")
        return history
    if closed_positions is None:
        closed_positions = []

    positions = normalize_closed_positions(open_positions, closed_positions)

    print("[*] Calculating Live Cash & Portfolio Values...")
    prev_bankroll = get_effective_bankroll(history, cfg.BANKROLL)
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
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
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
    wallet_cond_ids = {
        (p.get("conditionId") or "").lower()
        for p in open_positions if isinstance(p, dict) and p.get("conditionId")
    }
    to_import = []
    resolved_by_cond, exited_by_cond, predicted_by_cond = index_history_by_condition(history)

    for pos in positions:
        cond_id = pos.get("conditionId")
        if not cond_id:
            continue
        c_id_lower = cond_id.lower()
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

        cond_trades = trades_by_cond.get(c_id_lower, [])
        latest_trade_ts = max((t["ts"] for t in cond_trades), default=int(pos.get("timestamp") or 0))

        if c_id_lower in resolved_by_cond:
            updated_existing += reconcile_resolved_match(resolved_by_cond[c_id_lower], realized_pnl, total_bankroll)
            continue

        if is_from_closed and c_id_lower in exited_by_cond:
            updated_existing += reconcile_exited_match(
                exited_by_cond[c_id_lower], pos, avg_price, cur_price, size, realized_pnl, total_bankroll
            )
            continue

        if c_id_lower in predicted_by_cond:
            m = None
            market_id = predicted_by_cond[c_id_lower][0][0]
            p_ref = predicted_by_cond[c_id_lower][0][1]
            question = p_ref.get("question") or pos.get("title") or "unknown"
            slug = p_ref.get("slug") or pos.get("slug") or "unknown"
            end_date = p_ref.get("endDate") or pos.get("endDate")
        else:
            m, market_id, question, slug, end_date = resolve_position_market_metadata(pos, event_cache)

        is_market_resolved = bool(m and (m.get("closed") or m.get("umaResolutionStatus") == "resolved"))
        is_resolved_pos = bool(pos.get("redeemable")) or is_market_resolved
        is_exited_open_pos = is_from_closed and not is_resolved_pos
        pos["redeemable"] = is_resolved_pos
        pos["_exited"] = is_exited_open_pos

        _, _, days_until = parse_market_date(end_date)
        days_until = days_until or 0.0

        already_stored = False
        for m_id in list(history.get("predicted", {}).keys()):
            preds_list = ensure_list(history["predicted"][m_id])
            matching_preds = [
                p for p in preds_list
                if (c_id_lower and (p.get("conditionId") or "").lower() == c_id_lower)
                or (market_id and str(m_id) == str(market_id))
            ]
            if not matching_preds:
                continue

            already_stored = True
            wallet_preds = [p for p in matching_preds if p.get("source") == "wallet"]
            local_preds = [p for p in matching_preds if p.get("source") != "wallet"]

            delta_usdc, delta_pm = compute_tranche_delta(
                wallet_preds, cond_trades, initial_val, size, avg_price, prev_bankroll, latest_trade_ts
            )

            unmatched_delta_for_import = False
            if local_preds and delta_usdc > 0.05:
                sel_idx = _prompt_local_tranche_match(local_preds, delta_usdc, delta_pm, slug, question, auto_sync)
                if 1 <= sel_idx <= len(local_preds):
                    chosen_lp = local_preds.pop(sel_idx - 1)
                    _rebalance_wallet_costs(wallet_preds, initial_val, delta_usdc, prev_bankroll)
                    _apply_local_tranche_match(
                        chosen_lp, pos, m, delta_usdc, delta_pm, avg_price, cur_price, size,
                        realized_pnl, days_until, base_ego, total_bankroll, wallet_address,
                        pos_outcome_idx, is_exited_open_pos, latest_trade_ts, question, slug,
                    )
                    wallet_preds.append(chosen_lp)
                    if is_exited_open_pos:
                        imported_exited += 1
                    else:
                        imported_predicted += 1
                else:
                    unmatched_delta_for_import = True
                    _rebalance_wallet_costs(wallet_preds, initial_val, delta_usdc, prev_bankroll)

            wallet_target_usdc = max(0.0, initial_val - delta_usdc) if unmatched_delta_for_import else initial_val
            if wallet_preds:
                updated_existing += _update_wallet_tranches(
                    wallet_preds, pos, m, wallet_target_usdc, delta_usdc, delta_pm, avg_price,
                    cur_price, size, realized_pnl, days_until, base_ego, prev_bankroll,
                    total_bankroll, wallet_address, pos_outcome_idx, is_exited_open_pos,
                    latest_trade_ts, question, slug, end_date, bool(local_preds),
                )

            for lp in local_preds:
                lp["source"] = "local"

            if unmatched_delta_for_import:
                to_import.append({
                    "pos": pos, "m": m, "market_id": market_id, "question": question,
                    "slug": slug, "end_date": end_date, "avg_price": delta_pm,
                    "cur_price": cur_price, "size": size, "initial_val": delta_usdc,
                    "realized_pnl": realized_pnl, "days_until": days_until,
                    "cond_id": cond_id, "latest_trade_ts": latest_trade_ts,
                })

            target_key = market_id
            if m_id != target_key:
                history["predicted"].pop(m_id, None)
                if preds_list:
                    history["predicted"][target_key] = ensure_list(history["predicted"].get(target_key, [])) + preds_list
            elif preds_list:
                history["predicted"][target_key] = preds_list
            else:
                history["predicted"].pop(target_key, None)

            _transition_matched_market(
                history, target_key, pos, m, cond_id, pos_outcome_str, pos_outcome_idx,
                avg_price, cur_price, size, realized_pnl, total_bankroll, latest_trade_ts,
                is_exited_open_pos, trades_by_cond, event_cache,
            )
            break

        if not already_stored and not is_from_closed:
            already_stored = _revive_exited_position(
                history, c_id_lower, pos, pos_outcome_str, initial_val, size, cur_price
            )

        if already_stored:
            continue

        to_import.append({
            "pos": pos, "m": m, "market_id": market_id, "question": question,
            "slug": slug, "end_date": end_date, "avg_price": avg_price,
            "cur_price": cur_price, "size": size, "initial_val": initial_val,
            "realized_pnl": realized_pnl, "days_until": days_until,
            "cond_id": cond_id, "latest_trade_ts": latest_trade_ts,
        })

    _archive_missing_wallet_positions(history, closed_positions, wallet_cond_ids, trades_by_cond, total_bankroll)

    imp_p, imp_e, imp_r = _prompt_and_import_new_positions(
        history, to_import, wallet_address, base_ego, total_bankroll, auto_sync
    )
    imported_predicted += imp_p
    imported_exited += imp_e
    imported_resolved += imp_r

    local_unexecuted = []
    for m_id in list(history.get("predicted", {}).keys()):
        for p in ensure_list(history["predicted"][m_id]):
            if p.get("source") != "wallet":
                p["source"] = "local"
                local_unexecuted.append(p)

    print(
        f"\n[+] Sync complete: {imported_predicted} active imported, "
        f"{imported_exited} exited imported, {imported_resolved} resolved imported, "
        f"{updated_existing} existing synced."
    )
    if local_unexecuted:
        print(f"[!] Unexecuted Local Predictions (Not in Polymarket Wallet): {len(local_unexecuted)}")
        for lp in local_unexecuted:
            q_short = lp.get("question", "Unknown")
            if len(q_short) > 65:
                q_short = q_short[:62] + "..."
            q_linked = ansi_link(f"https://polymarket.com/event/{lp.get('slug', 'unknown')}", q_short)
            print(
                f"    - [LOCAL] ({lp.get('date', 'unknown')}) {q_linked} "
                f"| pu: {float(lp.get('pu', 0.5))*100:.1f}% | pm: {float(lp.get('pm', 0.5))*100:.1f}% "
                f"| Target Kelly: {float(lp.get('kelly', 0.0))*100:.2f}%"
            )
    return history


def resolve_market_entries(history, market_id, market_data, bankroll=None):
    """Resolves a closed/resolved market's predicted and exited entries into history['resolved'].

    Returns:
        bool: True if the market was closed/resolved and moved, False otherwise.
    """
    if not isinstance(market_data, dict):
        return False
    is_closed = market_data.get("closed", False)
    uma_resolved = market_data.get("umaResolutionStatus") == "resolved"
    if not (is_closed or uma_resolved):
        return False

    prices = parse_outcome_prices(market_data)
    if len(prices) < 2:
        prices = [0.5, 0.5]

    pred_list = ensure_list(history["predicted"].get(market_id, [])) + ensure_list(history["exited"].get(market_id, []))
    history["resolved"].setdefault(market_id, [])
    is_wallet_prof = bool(str(history.get("profile_id") or history.get("wallet_address") or "").startswith("0x"))
    active_b = bankroll if bankroll is not None else get_effective_bankroll(history, cfg.BANKROLL)

    for pred_data in pred_list:
        is_exited_p = bool(pred_data.get("exited"))
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
        resolved_entry.setdefault("resolved_ts", int(time.time()))
        if is_exited_p:
            pass  # Preserve already-realized exit PnL
        elif is_unexec_local or is_watch_pred:
            resolved_entry["realized_pnl"] = None
        else:
            new_pnl = calculate_resolved_pnl(
                resolved_entry,
                outcome,
                bankroll=active_b,
                allow_kelly_fallback=(not is_wallet_prof),
                use_entry_cost_fallback=True,
                preserve_exited=False,
            )
            if new_pnl is not None:
                resolved_entry["realized_pnl"] = new_pnl
        history["resolved"][market_id].append(resolved_entry)

    if not history["resolved"][market_id]:
        history["resolved"].pop(market_id, None)

    history["predicted"].pop(market_id, None)
    history["exited"].pop(market_id, None)
    return True


def update_resolutions(history):
    """Audits unresolved predictions in predicted and exited buckets against Gamma API."""
    resolved_count = 0
    history.setdefault("predicted", {})
    history.setdefault("exited", {})
    history.setdefault("resolved", {})
    market_ids = list({
        str(mid) for mid in list(history.get("predicted", {}).keys()) + list(history.get("exited", {}).keys())
        if not str(mid).startswith("custom_")
    })
    if market_ids:
        print(f"[*] Auditing {len(market_ids)} unresolved market(s) for resolution...")

    chunk_size = 40
    for i in range(0, len(market_ids), chunk_size):
        chunk = market_ids[i:i + chunk_size]
        resp_map = {}
        try:
            url = f"{cfg.GAMMA_API}/markets?" + "&".join(f"id={mid}" for mid in chunk)
            batch_res = api.get(url, timeout=6).json()
            if isinstance(batch_res, list):
                resp_map = {str(item.get("id")): item for item in batch_res if item.get("id")}
        except Exception:
            resp_map = {}

        for market_id in chunk:
            try:
                m_data = resp_map.get(market_id)
                if not m_data:
                    try:
                        m_data = api.get(f"{cfg.GAMMA_API}/markets/{market_id}", timeout=5).json()
                    except Exception:
                        continue
                if resolve_market_entries(history, market_id, m_data):
                    resolved_count += 1
            except Exception:
                continue

    if resolved_count > 0:
        print(f"[+] Auto-resolved and scored {resolved_count} closed market(s).")
    return history
