"""Synchronization of Polymarket wallet positions and automatic market resolution auditing."""
from datetime import datetime, timezone
import config as cfg
from config import api
from utils import ensure_list, parse_outcome_prices
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
    linked_wallet = f"\033]8;;{profile_url}\033\\{wallet_address}\033]8;;\033\\"
    base_ego = calculate_base_ego(history)

    print(f"[*] Syncing Polymarket wallet {linked_wallet}...")
    try:
        url = f"https://data-api.polymarket.com/positions?user={wallet_address}&sizeThreshold=0&limit=500"
        open_positions = api.get(url, timeout=10).json()
        if not isinstance(open_positions, list):
            open_positions = []
    except Exception as e:
        print(f"[!] Error fetching open positions: {e}")
        open_positions = []

    try:
        closed_url = f"https://data-api.polymarket.com/closed-positions?user={wallet_address}&limit=500"
        closed_positions = api.get(closed_url, timeout=10).json()
        if not isinstance(closed_positions, list):
            closed_positions = []
    except Exception as e:
        print(f"[!] Error fetching closed positions: {e}")
        closed_positions = []

    open_keys = {(p.get("conditionId"), p.get("outcome")) for p in open_positions if isinstance(p, dict)}
    for cp in closed_positions:
        if not isinstance(cp, dict):
            continue
        cp["_from_closed"] = True
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
        if isinstance(cp, dict) and (cp.get("conditionId"), cp.get("outcome")) not in open_keys
    ]

    trades_by_cond = {}
    try:
        act_url = f"https://data-api.polymarket.com/activity?user={wallet_address}&limit=500"
        act_resp = api.get(act_url, timeout=10).json()
        if isinstance(act_resp, list):
            for ev in act_resp:
                if not isinstance(ev, dict) or ev.get("type") != "TRADE":
                    continue
                c_id = ev.get("conditionId")
                if not c_id:
                    continue
                trades_by_cond.setdefault(c_id, []).append({
                    "ts": int(ev.get("timestamp") or 0),
                    "side": str(ev.get("side") or "BUY").upper(),
                    "usdc": float(ev.get("usdcSize") or 0.0),
                    "shares": float(ev.get("size") or 0.0),
                    "price": float(ev.get("price") or 0.5)
                })
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

    cfg.BANKROLL = round(cash_balance + portfolio_value, 4)
    history["wallet_summary"] = {
        "cash": round(cash_balance, 4),
        "portfolio": round(portfolio_value, 4),
        "bankroll": cfg.BANKROLL,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    }
    print(f"[*] Wallet Profile Updated: ${cfg.BANKROLL:,.2f} (Cash: ${cash_balance:,.2f} | Positions: ${portfolio_value:,.2f})")

    if not positions:
        print("[*] No positions returned from Polymarket endpoints.")
    else:
        print(f"Found {len(open_positions)} active/unclosed + {len(closed_positions)} closed/redeemed positions. Syncing...")

    imported_predicted = 0
    imported_exited = 0
    imported_resolved = 0
    updated_existing = 0
    event_cache = {}
    wallet_cond_ids = {p.get("conditionId") for p in open_positions if isinstance(p, dict) and p.get("conditionId")}
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
                if bool(pos.get("redeemable")) or cur_price >= 0.99 or cur_price <= 0.01:
                    out_v = 1.0 if cur_price >= 0.99 else 0.0
                    for e in matching_ex:
                        e["cur_pm"] = cur_price
                        e["realized_pnl"] = realized_pnl
                        e["outcome"] = out_v
                        updated_existing += 1
                    rem_ex = [e for e in ex_list if e not in matching_ex]
                    if rem_ex:
                        history["exited"][m_id] = rem_ex
                    else:
                        history["exited"].pop(m_id, None)
                    history["resolved"].setdefault(m_id, []).extend(matching_ex)
                else:
                    for e in matching_ex:
                        e["cur_pm"] = cur_price
                        e["realized_pnl"] = realized_pnl
                        updated_existing += 1
                break
            if already_in_exited:
                continue

        m, market_id, question, slug, end_date = resolve_position_market_metadata(pos, event_cache)
        cond_trades = trades_by_cond.get(cond_id, [])
        latest_trade_ts = max((t["ts"] for t in cond_trades), default=int(pos.get("timestamp") or 0))

        is_market_resolved = bool(m and (m.get("closed") or m.get("umaResolutionStatus") == "resolved"))
        is_resolved_pos = bool(pos.get("redeemable")) or is_market_resolved or (is_from_closed and (cur_price >= 0.99 or cur_price <= 0.01))
        is_exited_open_pos = is_from_closed and not is_resolved_pos
        pos["redeemable"] = is_resolved_pos
        pos["_exited"] = is_exited_open_pos

        try:
            res_dt = datetime.fromisoformat(str(end_date).replace('Z', '+00:00'))
            if res_dt.tzinfo is None:
                res_dt = res_dt.replace(tzinfo=timezone.utc)
            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            days_until = seconds_left / 86400.0
        except Exception:
            days_until = 0.0

        already_stored = False
        for m_id in list(history.get("predicted", {}).keys()):
            preds_list = ensure_list(history["predicted"][m_id])
            matching_preds = [
                p for p in preds_list
                if p.get("conditionId") == cond_id and (
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
                linked_q = f"\033]8;;{web_link}\033\\{question}\033]8;;\033\\"
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
                    lm_lp = calculate_logistic_metrics(chosen_lp.get("pu", delta_pm), delta_pm, ego_lp)
                    chosen_lp["question"] = question
                    chosen_lp["slug"] = slug
                    chosen_lp["tranche_pm"] = delta_pm
                    chosen_lp["pm"] = delta_pm
                    chosen_lp["cur_pm"] = cur_price
                    chosen_lp["cost_basis"] = round(0.0 if is_exited_open_pos else delta_usdc, 4)
                    chosen_lp["realized_pnl"] = realized_pnl
                    chosen_lp["kelly"] = lm_lp["model_kelly"]
                    chosen_lp["edge"] = lm_lp["active_edge"]
                    _, chosen_lp["apy"] = calculate_annualized_yield(chosen_lp["edge"], lm_lp["win_prob_m"], days_until)
                    chosen_lp["position"] = pos.get("outcome", "Yes")
                    chosen_lp["outcomeIndex"] = pos_outcome_idx
                    chosen_lp["source"] = "wallet"
                    chosen_lp["wallet"] = wallet_address
                    chosen_lp["exited"] = is_exited_open_pos
                    if latest_trade_ts > 0:
                        chosen_lp["last_synced_ts"] = latest_trade_ts
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
                raw_weights = [
                    float(wp.get("cost_basis", float(wp.get("kelly", 0.0)) * prev_bankroll))
                    for wp in wallet_preds
                ]
                total_w = sum(raw_weights)
                for wp, w_i in zip(wallet_preds, raw_weights):
                    share = (w_i / total_w) if total_w > 0 else (1.0 / len(wallet_preds))
                    tranche_pm = float(wp.get("tranche_pm", wp.get("pm", avg_price)))
                    ego_wp = float(wp.get("dynamic_ego", base_ego))
                    lm_wp = calculate_logistic_metrics(wp.get("pu", tranche_pm), tranche_pm, ego_wp)
                    wp["question"] = question
                    wp["slug"] = slug
                    wp["pm"] = tranche_pm
                    wp["cur_pm"] = cur_price
                    wp["cost_basis"] = round(0.0 if is_exited_open_pos else (wallet_target_usdc * share), 4)
                    wp["realized_pnl"] = round(realized_pnl * share, 4)
                    wp["kelly"] = lm_wp["model_kelly"]
                    wp["edge"] = lm_wp["active_edge"]
                    _, wp["apy"] = calculate_annualized_yield(wp["edge"], lm_wp["win_prob_m"], days_until)
                    wp["position"] = pos.get("outcome", "Yes")
                    wp["outcomeIndex"] = pos_outcome_idx
                    wp["source"] = "wallet"
                    wp["wallet"] = wallet_address
                    wp["exited"] = is_exited_open_pos
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
                for p in to_exit:
                    p["exited"] = True
                    p["cost_basis"] = 0.0
                    p["realized_pnl"] = realized_pnl
                if to_exit:
                    history["exited"].setdefault(target_key, []).extend(to_exit)
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

    for m_id in list(history.get("predicted", {}).keys()):
        preds_list = ensure_list(history["predicted"][m_id])
        kept_preds = []
        moved_exits = []
        for p in preds_list:
            c_id = p.get("conditionId")
            if p.get("source") == "wallet" and c_id and c_id not in wallet_cond_ids:
                p["cost_basis"] = 0.0
                p["exited"] = True
                sells = [t for t in trades_by_cond.get(c_id, []) if t["side"] == "SELL"]
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
        linked_question = f"\033]8;;{web_link}\033\\{question}\033]8;;\033\\"

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
        dynamic_ego = compute_dynamic_ego(base_ego, upper - lower, 0.0)
        lm = calculate_logistic_metrics(pu_val, avg_price, dynamic_ego)
        edge = lm["active_edge"]
        kelly_val = lm["model_kelly"]
        _, apy = calculate_annualized_yield(edge, lm["win_prob_m"], days_until)

        is_exited_flag = bool(pos.get("_exited"))
        pred_entry = {
            "question": question,
            "slug": slug,
            "date": pos.get("endDate", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
            "pu": pu_val,
            "pm": avg_price,
            "cur_pm": cur_price,
            "dynamic_ego": dynamic_ego,
            "kelly": kelly_val,
            "cost_basis": round(0.0 if is_exited_flag else initial_val, 4),
            "realized_pnl": realized_pnl,
            "edge": edge,
            "apy": apy,
            "conditionId": cond_id,
            "position": outcome_str,
            "outcomeIndex": outcome_idx,
            "source": "wallet",
            "wallet": wallet_address,
            "exited": is_exited_flag
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
            q_linked = f"\033]8;;{lp_link}\033\\{q_short}\033]8;;\033\\"
            k_pct = float(lp.get("kelly", 0.0)) * 100.0
            pu_pct = float(lp.get("pu", 0.5)) * 100.0
            pm_pct = float(lp.get("pm", 0.5)) * 100.0
            print(f"    - [LOCAL] ({lp.get('date', 'unknown')}) {q_linked} | pu: {pu_pct:.1f}% | pm: {pm_pct:.1f}% | Target Kelly: {k_pct:.2f}%")
    return history


def update_resolutions(history):
    """Audits unresolved predictions in predicted and exited buckets against Gamma API."""
    resolved_count = 0
    history.setdefault("exited", {})
    history.setdefault("resolved", {})
    open_buckets = [("predicted", list(history.get("predicted", {}).keys())), ("exited", list(history.get("exited", {}).keys()))]
    total_to_audit = sum(len(k_list) for _, k_list in open_buckets)
    if total_to_audit:
        print(f"[*] Auditing {total_to_audit} unresolved market(s) (active + exited) for resolution...")

    for bucket_name, market_ids in open_buckets:
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

                    pred_list = ensure_list(history[bucket_name][market_id])
                    history["resolved"].setdefault(market_id, [])

                    for pred_data in pred_list:
                        if pred_data.get("source") == "wallet":
                            try:
                                out_idx = int(pred_data.get("outcomeIndex", 0))
                            except Exception:
                                out_idx = 0
                            if len(prices) > out_idx:
                                outcome = 1.0 if prices[out_idx] >= 0.99 else (0.0 if prices[out_idx] <= 0.01 else 0.5)
                            else:
                                outcome = 0.5
                        else:
                            if prices[0] >= 0.99: outcome = 1.0
                            elif prices[1] >= 0.99: outcome = 0.0
                            else: outcome = 0.5
                        resolved_entry = dict(pred_data)
                        resolved_entry["outcome"] = outcome
                        history["resolved"][market_id].append(resolved_entry)

                    del history[bucket_name][market_id]
                    resolved_count += 1
            except Exception:
                continue

    if resolved_count > 0:
        print(f"[+] Auto-resolved and scored {resolved_count} closed market(s).")
    return history

