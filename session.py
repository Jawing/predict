"""session.py - Prediction session and custom calculator execution.

Handles interactive market streaming, Kelly calculations, VWAP orderbook walking,
and custom hypothetical market calculations.
"""

import math
from datetime import datetime, timezone, timedelta
import pandas as pd

from config import api
import config as cfg
from utils import (
    ensure_list,
    format_time_remaining,
    parse_market_date,
    calculate_position_holding,
)
from storage import load_history, save_history
from model import (
    calculate_base_ego,
    calculate_allocation,
    calculate_annualized_yield,
    calculate_logistic_metrics,
    calculate_dynamic_kelly_action,
    parse_user_input,
    calculate_vwap,
)
from market import (
    generate_market_stream,
    extract_prices,
    refresh_prices_from_clob,
    stream_clob_book,
    get_event_sub_markets,
    get_market_clob_tokens,
)


def format_velocity_display(edge, price, days):
    """Formats daily capital velocity and daily ROI with dynamic precision."""
    if days <= 0 or price <= 0 or abs(edge) < 1e-9:
        return "Immediate / N/A"
    daily_edge = edge / max(0.05, days)
    daily_roi = (edge / price) / max(0.05, days)
    d_edge_pct = daily_edge * 100.0
    d_roi_pct = daily_roi * 100.0
    edge_str = f"{d_edge_pct:+.4f}%" if abs(d_edge_pct) < 0.01 else f"{d_edge_pct:+.2f}%"
    roi_str = f"{d_roi_pct:+.4f}%" if abs(d_roi_pct) < 0.01 else f"{d_roi_pct:+.2f}%"
    return f"{edge_str} edge/day ({roi_str} ROI/day)"


def get_available_sub_markets(current_m, sub_markets, history, session_candidates=None):
    """Filters sibling sub-markets to only unpredicted, active, open markets (excluding current)."""
    if not sub_markets or len(sub_markets) <= 1:
        return []
    cur_id = str(current_m.get("id"))
    staged_ids = {str(c.get("market_id")) for c in (session_candidates or [])}
    pred_ids = set(str(k) for k in history.get("predicted", {}).keys())
    skip_ids = set(str(k) for k in history.get("skipped", {}).keys())

    available = []
    for sm in sub_markets:
        sm_id = str(sm.get("id"))
        if sm_id == cur_id:
            continue
        if sm.get("closed") or sm.get("umaResolutionStatus") == "resolved" or sm.get("active") is False:
            continue
        if sm_id in pred_ids or sm_id in staged_ids or sm_id in skip_ids:
            continue
        available.append(sm)
    return available


def select_sub_market(current_m, sub_markets, history, session_candidates=None, page_size=10):
    """Displays unanchored list of available sibling sub-markets with page scrolling and instant search."""
    available = get_available_sub_markets(current_m, sub_markets, history, session_candidates)
    if not available:
        print("[!] No other available sub-markets found for this event.\n")
        return None

    total_items = len(available)
    total_pages = math.ceil(total_items / page_size)
    parent_title = current_m.get("parent_name") or "Event Sub-Markets"
    current_page = 0

    while True:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_items)

        print(f"\n" + "=" * 76)
        if total_pages > 1:
            print(f" [AVAILABLE SUB-MARKETS: {parent_title} (Page {current_page + 1}/{total_pages} - {total_items} total)]")
        else:
            print(f" [AVAILABLE SUB-MARKETS: {parent_title} ({total_items} available)]")
        print("-" * 76)
        print(f"  # | Market / Sub-Question")
        print(f"----+-------------------------------------------------------------------")

        for idx in range(start_idx, end_idx):
            sm = available[idx]
            title = sm.get("groupItemTitle") or sm.get("question") or "unknown"
            title_disp = (title[:62] + "...") if len(title) > 65 else title
            print(f" {idx + 1:2d} | {title_disp}")

        print("=" * 76)

        prompt_str = f"Select # (1-{total_items}), [n]ext, [p]rev, [a]ll, or search [Enter to cancel]: " if total_pages > 1 else f"Select # (1-{total_items}), or search [Enter to cancel]: "
        choice = input(prompt_str).strip()
        if not choice:
            return None

        choice_lower = choice.lower()
        if total_pages > 1 and choice_lower in ("n", "next", ">"):
            current_page = current_page + 1 if current_page < total_pages - 1 else 0
            continue

        if total_pages > 1 and choice_lower in ("p", "prev", "<"):
            current_page = current_page - 1 if current_page > 0 else total_pages - 1
            continue

        if total_pages > 1 and choice_lower in ("a", "all"):
            print(f"\n" + "=" * 76)
            print(f" [ALL SUB-MARKETS: {parent_title} ({total_items} available)]")
            print("-" * 76)
            print(f"  # | Market / Sub-Question")
            print(f"----+-------------------------------------------------------------------")
            for i, sm in enumerate(available, 1):
                title = sm.get("groupItemTitle") or sm.get("question") or "unknown"
                title_disp = (title[:62] + "...") if len(title) > 65 else title
                print(f" {i:2d} | {title_disp}")
            print("=" * 76)
            sel = input(f"Select # to predict (1-{total_items}) [Enter to cancel]: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= total_items:
                chosen = available[int(sel) - 1].copy()
                chosen["parent_slug"] = current_m.get("parent_slug")
                chosen["parent_name"] = current_m.get("parent_name")
                chosen["event_markets"] = sub_markets
                return chosen
            return None

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < total_items:
                chosen = available[idx].copy()
                chosen["parent_slug"] = current_m.get("parent_slug")
                chosen["parent_name"] = current_m.get("parent_name")
                chosen["event_markets"] = sub_markets
                return chosen
            else:
                print(f"[!] Invalid selection. Please enter a number between 1 and {total_items}.\n")
                continue

        # Instant text search
        matches = [
            (i, sm) for i, sm in enumerate(available)
            if choice_lower in (sm.get("groupItemTitle") or sm.get("question") or "").lower()
        ]
        if len(matches) == 1:
            idx, chosen = matches[0]
            m_title = chosen.get("groupItemTitle") or chosen.get("question")
            print(f"[+] Matched #{idx + 1}: {m_title}")
            chosen_ret = chosen.copy()
            chosen_ret["parent_slug"] = current_m.get("parent_slug")
            chosen_ret["parent_name"] = current_m.get("parent_name")
            chosen_ret["event_markets"] = sub_markets
            return chosen_ret
        elif len(matches) > 1:
            print(f"\n[+] Found {len(matches)} matches for '{choice}':")
            for idx, sm in matches:
                t = sm.get("groupItemTitle") or sm.get("question") or "unknown"
                t_disp = (t[:62] + "...") if len(t) > 65 else t
                print(f" {idx + 1:2d} | {t_disp}")
            sel = input(f"\nSelect # to predict [Enter to cancel]: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= total_items:
                chosen = available[int(sel) - 1].copy()
                chosen["parent_slug"] = current_m.get("parent_slug")
                chosen["parent_name"] = current_m.get("parent_name")
                chosen["event_markets"] = sub_markets
                return chosen
            continue
        else:
            print(f"[!] No matches for '{choice}'. Please enter a number (1-{total_items}), 'n', 'p', or Enter to cancel.\n")
            continue


def _capture_market_state(m, history, portfolio_len):
    """Snapshots market presence in history for redo/rollback support."""
    m_id = m.get("id")
    return {
        "in_pred": m_id in history.get("predicted", {}),
        "pred_data": history.get("predicted", {}).get(m_id),
        "in_skip": m_id in history.get("skipped", {}),
        "skip_data": history.get("skipped", {}).get(m_id),
        "portfolio_len": portfolio_len,
    }


def _check_and_enqueue_sibling_submarket(m, sub_markets, history, session_candidates, session_markets, current_index, portfolio_len):
    """Offers to predict another sibling sub-market for the parent event."""
    avail_after = get_available_sub_markets(m, sub_markets, history, session_candidates)
    if avail_after:
        another = input(f"Predict another sub-market for '{m.get('parent_name')}'? [y/N]: ").strip().lower()
        if another in ("y", "yes"):
            chosen = select_sub_market(m, sub_markets, history, session_candidates)
            if chosen:
                chosen["original_state"] = _capture_market_state(chosen, history, portfolio_len)
                session_markets.insert(current_index + 1, chosen)


def _print_market_card(m, exact_date_str, days_str, num_avail, event_url):
    """Renders the standard terminal market header card."""
    sub_tag = f" ({num_avail} other available)" if num_avail > 0 else ""
    print(f"==================================================")
    print(f"Event    : {m.get('parent_name')}{sub_tag}")
    print(f"URL      : {event_url}")
    print(f"Market   : {m.get('question', 'unknown')}")
    print(f"Resolves : {exact_date_str} ({days_str})\n")


def run_prediction_session(
    mode="discover",
    sub_mode="all",
    target_slugs=None,
    category="All",
    max_offset=0,
    exclude_mode="none",
    discovery_order="volume",
    target_market_id=None,
):
    """Main interactive terminal prediction loop for discovery or targeted markets."""
    history = load_history()
    save_history(history)

    base_ego = calculate_base_ego(history)

    # Resolve live bankroll and past exposure
    current_bankroll = cfg.BANKROLL
    ws = history.get("wallet_summary", {})
    if isinstance(ws, dict) and ws.get("bankroll") and float(ws.get("bankroll")) > 0:
        current_bankroll = float(ws.get("bankroll"))

    cumulative_exposure = 0.0
    for pred_value in history.get("predicted", {}).values():
        preds = ensure_list(pred_value)
        for p in preds:
            lm_p = calculate_logistic_metrics(
                p.get("pu", 0.5), p.get("pm", 0.5), float(p.get("dynamic_ego", base_ego)),
                held_kelly=float(p.get("kelly", 0.0) or 0.0), cur_pm=p.get("cur_pm")
            )
            cumulative_exposure += lm_p["eff_kelly"]

    vec_note = f" | {discovery_order.upper()}" if mode == "discover" else ""
    cat_note = f" [{category.upper()}]" if mode == "discover" and category.upper() != "ALL" else ""
    print(f"\n--- SESSION: {mode.upper()}{cat_note}{vec_note} [Bankroll: ${current_bankroll:,.2f} | Base Ego: {base_ego:.3f} | Exposure: {cumulative_exposure*100:.1f}%] ---\n")

    market_stream = generate_market_stream(
        mode,
        sub_mode,
        target_slugs,
        history,
        category,
        max_offset,
        exclude_mode,
        discovery_order,
        target_market_id=target_market_id,
    )

    portfolio_data = []
    session_trades = {}
    session_markets = []
    session_candidates = []
    current_index = 0

    while True:
        if current_index >= len(session_markets):
            try:
                new_m = next(market_stream)
                pm_bid, pm_ask, pm_mid = extract_prices(new_m)
                if pm_mid is None:
                    continue
                new_m["cached_bid"], new_m["cached_ask"], new_m["cached_mid"] = pm_bid, pm_ask, pm_mid
                new_m["original_state"] = _capture_market_state(new_m, history, len(portfolio_data))
                session_markets.append(new_m)
            except (StopIteration, TypeError):
                print("\n[*] Data stream ended or exhausted.")
                break

        m = session_markets[current_index]
        m = refresh_prices_from_clob(m)
        pm_bid, pm_ask, pm_mid = extract_prices(m)
        m["cached_bid"], m["cached_ask"], m["cached_mid"] = pm_bid, pm_ask, pm_mid

        sub_markets = get_event_sub_markets(m)
        avail_subs = get_available_sub_markets(m, sub_markets, history, session_candidates)
        num_avail = len(avail_subs)

        market_id = m.get("id")
        question = m.get("question", "unknown")
        event_slug = m.get("parent_slug", "unknown")
        event_url = f"https://polymarket.com/event/{event_slug}"
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        res_dt, seconds_left, _ = parse_market_date(m.get("endDate"))
        exact_date_str = res_dt.strftime("%B %d, %Y") if res_dt else "unknown"
        days_str = format_time_remaining(seconds_left) if seconds_left is not None else "unknown"

        _print_market_card(m, exact_date_str, days_str, num_avail, event_url)

        prompt_sub = f", 'sub' (list {num_avail} mkts)" if num_avail > 0 else ""
        while True:
            user_input = input(
                f"Enter % bounds ('36-48' or '42'){prompt_sub}, 'l' (live), 's' (skip), 'r' (redo), 'q' (quit): "
            ).strip()
            cmd_parts = user_input.split()
            if cmd_parts and cmd_parts[0].lower() in ("l", "live"):
                custom_tok = cmd_parts[1] if len(cmd_parts) > 1 else None
                follow_up = stream_clob_book(m, custom_token_id=custom_tok)
                pm_bid, pm_ask, pm_mid = m["cached_bid"], m["cached_ask"], m["cached_mid"]
                if follow_up:
                    user_input = follow_up
                    break
                continue
            if user_input.lower() in ("sub", "m", "markets", "list") and num_avail > 0:
                chosen = select_sub_market(m, sub_markets, history, session_candidates)
                if chosen:
                    chosen = refresh_prices_from_clob(chosen)
                    b_p, a_p, m_p = extract_prices(chosen)
                    chosen["cached_bid"], chosen["cached_ask"], chosen["cached_mid"] = b_p, a_p, m_p
                    chosen["original_state"] = _capture_market_state(chosen, history, len(portfolio_data))
                    session_markets[current_index] = chosen
                    m = chosen
                    market_id = m.get("id")
                    question = m.get("question", "unknown")

                    res_dt, seconds_left, _ = parse_market_date(m.get("endDate"))
                    exact_date_str = res_dt.strftime("%B %d, %Y") if res_dt else "unknown"
                    days_str = format_time_remaining(seconds_left) if seconds_left is not None else "unknown"

                    pm_bid, pm_ask, pm_mid = m["cached_bid"], m["cached_ask"], m["cached_mid"]
                    avail_subs = get_available_sub_markets(m, sub_markets, history, session_candidates)
                    num_avail = len(avail_subs)
                    prompt_sub = f", 'sub' (list {num_avail} mkts)" if num_avail > 0 else ""
                    _print_market_card(m, exact_date_str, days_str, num_avail, event_url)
                    continue
                else:
                    continue
            break

        if user_input.lower() == "q":
            print("\nSaving and quitting session...")
            break

        elif user_input.lower() == "r":
            if current_index > 0:
                print("\n[!] Rolling back to previous market...")
                current_index -= 1
                prev_m = session_markets[current_index]
                prev_market_id = prev_m.get("id")

                if session_candidates and session_candidates[-1].get("market_id") == prev_market_id:
                    session_candidates.pop()

                if prev_market_id in session_trades:
                    refund_amount = session_trades.pop(prev_market_id)
                    cumulative_exposure -= (refund_amount / current_bankroll) if current_bankroll > 0 else 0.0

                orig = prev_m["original_state"]
                if orig["in_pred"]:
                    history["predicted"][prev_market_id] = orig["pred_data"]
                else:
                    history["predicted"].pop(prev_market_id, None)

                if orig["in_skip"]:
                    history["skipped"][prev_market_id] = orig["skip_data"]
                else:
                    history["skipped"].pop(prev_market_id, None)

                portfolio_data = portfolio_data[: orig["portfolio_len"]]
            else:
                print("\n[!] Cannot redo. This is the first market of the session.\n")
            continue

        elif user_input.lower() == "s":
            print("Status: Skipped.\n")
            if market_id not in history["predicted"]:
                history["skipped"][market_id] = {
                    "question": question,
                    "slug": event_slug,
                    "date": today_str,
                }
            _check_and_enqueue_sibling_submarket(
                m, sub_markets, history, session_candidates, session_markets, current_index, len(portfolio_data)
            )
            current_index += 1
            continue

        try:
            lower, upper = parse_user_input(user_input)
            fee_rate = (m.get("feeSchedule") or {}).get("rate")
            if fee_rate is None:
                try:
                    f = float(m.get("fee") or 0)
                    fee_rate = f / 1e18 if f > 1 else f
                except Exception:
                    fee_rate = 0.0

            existing_preds = ensure_list(history.get("predicted", {}).get(market_id, []))
            is_existing = len(existing_preds) > 0
            live_held_val = 0.0
            held_shares = 0.0
            if is_existing:
                for p in existing_preds:
                    is_p_no = (p.get("position", "Yes") == "No")
                    p_live = (1.0 - pm_mid) if is_p_no else pm_mid
                    v, _, sh = calculate_position_holding(p, p_live, current_bankroll)
                    live_held_val += v
                    held_shares += sh

            held_mtm_k = (live_held_val / current_bankroll) if (current_bankroll > 0 and live_held_val > 0) else sum(float(p.get("kelly", 0.0) or 0.0) for p in existing_preds)

            action, true_price, raw_kelly, eff_kelly, full_alloc, dynamic_ego, edge = calculate_allocation(
                lower, upper, pm_bid, pm_ask, fee_rate, current_bankroll, base_ego
            )

            # Incremental tranche allocation if position already held
            if is_existing and held_mtm_k > 0:
                delta_k = max(0.0, eff_kelly - held_mtm_k)
                final_alloc = delta_k * current_bankroll
            else:
                delta_k = eff_kelly
                final_alloc = full_alloc

            _, _, days_until = parse_market_date(m.get("endDate"))
            days_until = days_until or 0.0

            roi, apy = calculate_annualized_yield(edge, true_price, days_until)

            # Orderbook VWAP calculation
            vwap_true = true_price
            slippage = 0.0
            adjusted_edge = edge
            adjusted_apy = apy

            if action in ["YES", "NO"] and final_alloc > 0:
                yes_token, no_token = get_market_clob_tokens(m)
                token_id = yes_token if action == "YES" else no_token
                asks_list = []
                if token_id:
                    try:
                        r = api.get(f"https://clob.polymarket.com/book?token_id={token_id}", timeout=5)
                        if r.status_code == 200:
                            asks_list = r.json().get("asks", [])
                    except Exception:
                        pass
                if asks_list:
                    vwap, _ = calculate_vwap(asks_list, final_alloc)
                    if vwap > 0:
                        vwap_true = vwap + (fee_rate * vwap * (1.0 - vwap))
                        slippage = max(0.0, vwap_true - true_price)
                        win_prob = true_price + edge
                        adjusted_edge = win_prob - vwap_true
                        _, adjusted_apy = calculate_annualized_yield(adjusted_edge, vwap_true, days_until)
                        if adjusted_edge < cfg.MIN_EDGE:
                            action = "THIN_EDGE"

            pu_mid_val = (lower + upper) / 2.0
            lm = calculate_logistic_metrics(
                pu_mid_val, pm_mid, dynamic_ego, held_kelly=eff_kelly if is_existing else raw_kelly, fee_rate=fee_rate
            )

            pos_side = action.capitalize() if action in ("YES", "NO") else ("Yes" if lm["is_yes"] else "No")
            is_pos_no = (pos_side == "No")
            stored_pu = round(1.0 - pu_mid_val, 6) if is_pos_no else round(pu_mid_val, 6)
            stored_pm = round(true_price if true_price > 0 else (1.0 - pm_mid), 6) if is_pos_no else round(pm_mid, 6)
            stored_cur_pm = round(1.0 - pm_mid, 6) if is_pos_no else round(pm_mid, 6)
            if is_pos_no:
                lm = calculate_logistic_metrics(
                    stored_pu, stored_pm, dynamic_ego, held_kelly=eff_kelly if is_existing else raw_kelly,
                    cur_pm=stored_cur_pm, fee_rate=fee_rate
                )

            if not is_existing and lm["regime"] == "AGAINST":
                lm["regime"] = "ACTIVE"

            if is_existing:
                primary_p = existing_preds[0].copy()
                primary_p["pu"] = stored_pu
                primary_p["cur_pm"] = stored_cur_pm
                primary_p["dynamic_ego"] = dynamic_ego
                primary_p["edge"] = adjusted_edge
                primary_p["apy"] = adjusted_apy
                primary_p["date"] = today_str
                if delta_k >= 0.005:
                    primary_p["kelly"] = raw_kelly
                    new_shares = final_alloc / true_price if true_price > 0 else 0.0
                    old_cb = float(primary_p.get("cost_basis", 0.0) or 0.0)
                    old_sh = float(primary_p.get("shares", 0.0) or (old_cb / float(primary_p.get("pm", stored_pm))))
                    tot_sh = old_sh + new_shares
                    tot_cb = old_cb + final_alloc
                    primary_p["cost_basis"] = round(tot_cb, 4)
                    primary_p["shares"] = round(tot_sh, 4)
                    primary_p["pm"] = round(tot_cb / tot_sh, 6) if tot_sh > 0 else stored_pm
                    primary_p["tranche_added"] = final_alloc
                    if m.get("endDate"):
                        primary_p["endDate"] = m.get("endDate")
                    if m.get("volume") is not None:
                        primary_p["volume"] = float(m.get("volume") or 0.0)
                    if m.get("volume24hr") is not None:
                        primary_p["volume24hr"] = float(m.get("volume24hr") or 0.0)
                elif primary_p.get("kelly") is None or float(primary_p.get("kelly", 0.0)) <= 0:
                    primary_p["kelly"] = raw_kelly
                candidate_to_commit = primary_p
            else:
                candidate_to_commit = {
                    "question": question,
                    "slug": event_slug,
                    "date": today_str,
                    "pu": stored_pu,
                    "pm": stored_pm,
                    "cur_pm": stored_cur_pm,
                    "dynamic_ego": dynamic_ego,
                    "kelly": raw_kelly,
                    "edge": adjusted_edge,
                    "apy": adjusted_apy,
                    "conditionId": m.get("conditionId"),
                    "position": pos_side,
                    "source": "local",
                    "cost_basis": round(final_alloc, 4),
                    "shares": round(final_alloc / true_price, 4) if true_price > 0 else 0.0,
                    "endDate": m.get("endDate"),
                    "volume": float(m.get("volume") or 0.0),
                    "volume24hr": float(m.get("volume24hr") or 0.0),
                }

            norm_target = lm["target_win_m"]
            side_tag = f" ({pos_side})" if is_pos_no else ""
            print(f"\n--- MARKET ANALYSIS ---")
            print(f"Spread       : Bid {pm_bid*100:.1f}% | Ask {pm_ask*100:.1f}% (Spread: {(pm_ask - pm_bid)*100:.1f}%)")
            conv_str = f"User {lower*100:.1f}%" if lower == upper else f"User [{lower*100:.1f}%-{upper*100:.1f}%]"
            print(f"Conviction   : {conv_str} -> Active Target: {norm_target*100:.1f}%{side_tag} (Ego: {dynamic_ego:.2f})")
            ref_px = stored_pm if not is_existing else stored_cur_pm
            dist_tgt = norm_target - ref_px
            print(f"Runway       : {dist_tgt*100:+.1f}% | Efficiency: {lm['eff_live']*100:.1f}% ({lm['regime']})")

            vel_str = format_velocity_display(adjusted_edge, vwap_true, days_until) if (action in ["YES", "NO"] and final_alloc > 0 and adjusted_edge > 0) else "N/A (No trade / 0 allocation)"
            print(f"Velocity     : {vel_str}")

            if action in ["YES", "NO"]:
                if is_existing and (held_mtm_k > 0 or live_held_val > 0):
                    spread_val = (pm_ask - pm_bid) if (pm_ask > pm_bid) else 0.01
                    half_sp = max(0.0025, spread_val / 2.0)
                    dyn_action = calculate_dynamic_kelly_action(
                        win_prob_bal=norm_target,
                        cur_p=ref_px,
                        held_frac=held_mtm_k,
                        bankroll=current_bankroll,
                        fee_rate=fee_rate,
                        half_spread=half_sp
                    )
                    action_label = dyn_action["action_label"]
                    print(f"Action       : {action_label} | Net Edge: {adjusted_edge*100:+.2f}%")
                    sh_tag = f" | {held_shares:,.1f} shares" if held_shares > 0 else ""
                    dyn_target_usd = dyn_action["f_star_hold"] * current_bankroll
                    print(f"Position     : Held {held_mtm_k*100:.2f}% (${live_held_val:,.2f}{sh_tag}) -> Dynamic Optimal {dyn_action['f_star_hold']*100:.2f}% (${dyn_target_usd:,.2f})")
                else:
                    slip_note = f" (Slippage: +{slippage*100:.2f}%)" if slippage > 0.0001 else ""
                    print(f"Action       : BUY {action} @ {vwap_true*100:.1f}% VWAP{slip_note} | Net Edge: {adjusted_edge*100:+.2f}%")
                    print(f"Allocation   : ${final_alloc:,.2f} ({eff_kelly*100:.2f}% Eff-Kelly | {raw_kelly*100:.2f}% Raw)")

                    ws_curr = history.get("wallet_summary", {})
                    wallet_cash = float(ws_curr.get("cash", current_bankroll) or current_bankroll) if isinstance(ws_curr, dict) else current_bankroll
                    if final_alloc > wallet_cash + 0.50:
                        shortfall = final_alloc - wallet_cash
                        print(f"Cash Alert   : Target allocation exceeds cash by ${shortfall:,.2f} (Available Cash: ${wallet_cash:,.2f})")
                        rebal_candidates = []
                        for m_id, p_list in history.get("predicted", {}).items():
                            for p_item in ensure_list(p_list):
                                if p_item.get("source") == "wallet":
                                    p_ed = float(p_item.get("edge", 0.0) or 0.0)
                                    p_sh = float(p_item.get("shares", 0.0) or 0.0)
                                    p_cp = float(p_item.get("cur_pm", p_item.get("pm", 0.5)))
                                    p_val = p_sh * p_cp
                                    if p_val > 1.0:
                                        rebal_candidates.append((p_val, p_ed, p_item.get("question", "")))
                        rebal_candidates.sort(key=lambda x: x[1])
                        if rebal_candidates:
                            c_val, c_edge, c_q = rebal_candidates[0]
                            c_q_disp = (c_q[:35] + "...") if len(c_q) > 38 else c_q
                            print(f"  -> Rebalance Suggestion: Liquidate/Trim {c_q_disp} [Edge: {c_edge*100:+.1f}% | Equity: ${c_val:,.2f}]")
            else:
                if lm["live_active_edge"] <= 0:
                    reason = f"Price {pm_mid*100:.1f}% at/above target {lm['pu_norm']*100:.1f}% (Exit Signal: SELL)"
                elif action == "THIN_EDGE":
                    if edge > 0 and adjusted_edge <= 0:
                        reason = f"Orderbook slippage on ${final_alloc:,.2f} order (Ask {true_price*100:.1f}% -> VWAP {vwap_true*100:.1f}%) wiped raw edge"
                    elif edge > 0 and adjusted_edge < cfg.MIN_EDGE:
                        reason = f"Net edge {adjusted_edge*100:.2f}% below {cfg.MIN_EDGE*100:.1f}% threshold after slippage"
                    else:
                        reason = f"Edge below {cfg.MIN_EDGE*100:.1f}% threshold"
                elif action == "NONE":
                    reason = "Trapped inside bid-ask spread"
                else:
                    reason = "Unknown"
                print(f"Action       : $0 ({reason}) | Edge: {adjusted_edge*100:+.2f}%")
                if is_existing and live_held_val > 0:
                    sh_tag = f" | {held_shares:,.1f} shares" if held_shares > 0 else ""
                    print(f"Existing Pos : {held_mtm_k*100:.2f}% MTM (${live_held_val:,.2f}{sh_tag}) | Target: 0.00% ($0.00)")

            if action in ["YES", "NO"] and adjusted_edge > 0 and final_alloc > 0:
                k_disp = delta_k if (is_existing and held_mtm_k > 0) else eff_kelly
                cand = {
                    "market_id": market_id,
                    "question": question,
                    "slug": event_slug,
                    "action": action,
                    "side": pos_side,
                    "price": vwap_true,
                    "raw_kelly": raw_kelly,
                    "eff_kelly": k_disp,
                    "delta_k": delta_k,
                    "alloc": final_alloc,
                    "edge": adjusted_edge,
                    "apy": adjusted_apy,
                    "dynamic_ego": dynamic_ego,
                    "is_existing": is_existing,
                    "held_val": live_held_val,
                    "held_mtm_k": held_mtm_k,
                    "held_shares": held_shares,
                    "payload": candidate_to_commit,
                    "days_until": days_until,
                    "regime": lm["regime"],
                    "target": norm_target,
                    "selected": True,
                }
                session_candidates = [c for c in session_candidates if c["market_id"] != market_id]
                session_candidates.append(cand)

                staged_alloc = sum(c["alloc"] for c in session_candidates)
                staged_k = sum(c["eff_kelly"] for c in session_candidates)
                c_cnt = len(session_candidates)
                print(f"Status       : [+] Staged ({c_cnt} candidate{'s' if c_cnt > 1 else ''} | ${staged_alloc:,.2f} / ${current_bankroll:,.2f} [{staged_k*100:.1f}% Kelly])\n")

                if staged_alloc >= current_bankroll:
                    print("=" * 80)
                    print(f"[!] CAPITAL POOL SATURATED: Staged candidates reach ${staged_alloc:,.2f} ({staged_alloc/current_bankroll*100:.1f}% of Bankroll).")
                    c_choice = input("    Continue scanning [Enter] | Finalize & Review candidates now [f/q]: ").strip().lower()
                    if c_choice in ("f", "q", "done", "exit"):
                        print("\n[*] Finalizing session to review candidate pool...")
                        break
                    print("=" * 80 + "\n")
            else:
                print(f"Status       : [-] Pass (Edge: {adjusted_edge*100:+.2f}%)\n")

            _check_and_enqueue_sibling_submarket(
                m, sub_markets, history, session_candidates, session_markets, current_index, len(portfolio_data)
            )
            current_index += 1

        except ValueError as e:
            print(f"\n[!] Error: {e}. Please try again.\n")

    _review_and_commit_candidates(
        session_candidates=session_candidates,
        history=history,
        current_bankroll=current_bankroll,
        portfolio_data=portfolio_data,
        session_trades=session_trades
    )

    save_history(history)
    return pd.DataFrame(portfolio_data)


def _review_and_commit_candidates(session_candidates, history, current_bankroll, portfolio_data, session_trades):
    """Presents staged session candidates sorted by edge with smart default recommendation within Kelly limits."""
    if not session_candidates:
        print("\n[*] Session completed with 0 candidates staged.")
        return

    # Sort: highest edge first
    session_candidates.sort(key=lambda c: float(c.get("edge", 0.0) or 0.0), reverse=True)

    ws = history.get("wallet_summary", {})
    wallet_cash = float(ws.get("cash", current_bankroll) or current_bankroll) if isinstance(ws, dict) else current_bankroll

    # Preselect top candidates up to 100% bankroll
    cum_k = 0.0
    cum_alloc = 0.0
    for idx, c in enumerate(session_candidates):
        alloc = float(c.get("alloc", 0.0))
        k_val = float(c.get("eff_kelly", 0.0))
        if idx == 0 or (cum_alloc + alloc <= current_bankroll + 1.0):
            c["selected"] = True
            c["recommended"] = True
            cum_alloc += alloc
            cum_k += k_val
        else:
            c["selected"] = False
            c["recommended"] = False

    while True:
        box_len = 106
        print("\n" + "=" * box_len)
        print(" " * 36 + "[ SESSION CANDIDATES REVIEW ]")
        print("-" * box_len)
        print("  # | Sel | Side | Edge    | Eff-Kelly | Allocation | Cumulative | Question")
        print("----+-----+------+---------+-----------+------------+------------+-----------------------------------------------")
        running_tot = 0.0
        for i, c in enumerate(session_candidates, 1):
            sel_mark = "[*]" if c["selected"] else "[ ]"
            running_tot += c["alloc"] if c["selected"] else 0.0
            q_txt = c["question"]
            q_disp = (q_txt[:43] + "...") if len(q_txt) > 46 else q_txt
            exceed_tag = " (Exceeds Bankroll)" if (running_tot > current_bankroll and c["selected"]) else ""
            print(f" {i:2d} | {sel_mark} | {c['side']:<4} | {c['edge']*100:+6.2f}% | {c['eff_kelly']*100:8.2f}% | ${c['alloc']:>9,.2f} | ${running_tot:>9,.2f} | {q_disp}{exceed_tag}")
        print("=" * box_len)

        sel_candidates = [c for c in session_candidates if c["selected"]]
        tot_sel_alloc = sum(c["alloc"] for c in sel_candidates)
        tot_sel_k = sum(c["eff_kelly"] for c in sel_candidates)

        cash_info = f" (Available Cash: ${wallet_cash:,.2f})" if (isinstance(ws, dict) and "cash" in ws) else ""
        print(f"Capital State  : ${current_bankroll:,.2f} Bankroll{cash_info}")
        print(f"Selected Pool  : {len(sel_candidates)} of {len(session_candidates)} candidates | ${tot_sel_alloc:,.2f} ({tot_sel_k*100:.1f}% Eff-Kelly)")
        if tot_sel_alloc > current_bankroll + 1.0:
            print(f"[!] Warning    : Selected allocation exceeds bankroll by ${tot_sel_alloc - current_bankroll:,.2f}")

        act_prompt = f"\nActions: [Enter] Commit Selected  [1-{len(session_candidates)}] Toggle  [a] Select All  [q] Discard All\n> "
        choice = input(act_prompt).strip().lower()

        if choice == "":
            if not sel_candidates:
                print("[*] No candidates selected. Session ended without commits.")
                break
            for c in sel_candidates:
                m_id = c["market_id"]
                history.setdefault("predicted", {})[m_id] = [c["payload"]]
                history.get("skipped", {}).pop(m_id, None)
                session_trades[m_id] = c["alloc"]
                portfolio_data.append({
                    "Question": c["question"][:50] + "..",
                    "Action": c["action"],
                    "Ego": f"{c['dynamic_ego']:.2f}",
                    "Price": f"{c['price']*100:.1f}%",
                    "Kelly": f"{c['eff_kelly']*100:.2f}%",
                    "EffKelly": f"{c['eff_kelly']*100:.2f}%",
                    "Alloc": f"${c['alloc']:,.2f}",
                })
            save_history(history)
            print(f"\n[+] Successfully committed {len(sel_candidates)} candidate(s) to active portfolio (${tot_sel_alloc:,.2f} total).")
            break
        elif choice == "q":
            print("\n[*] Discarded all staged candidates (0 committed).")
            break
        elif choice == "a":
            all_sel = all(c["selected"] for c in session_candidates)
            for c in session_candidates:
                c["selected"] = not all_sel
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(session_candidates):
                session_candidates[idx]["selected"] = not session_candidates[idx]["selected"]
            else:
                print(f"[!] Choice out of range (1-{len(session_candidates)}).")
        else:
            print("[!] Invalid selection.")


def custom_calculator():
    """Interactive custom calculator for hypothetical market calculations."""
    print("\n--- CUSTOM MARKET CALCULATOR ---")
    history = load_history()
    ego_mode = input("Custom Base Ego (0.0-1.0) [Enter for Historical Ego]: ").strip()
    try:
        base_ego = (
            float(ego_mode)
            if (ego_mode and 0.0 <= float(ego_mode) <= 1.0)
            else calculate_base_ego(history)
        )
    except ValueError:
        base_ego = calculate_base_ego(history)

    try:
        m_lower, m_upper = parse_user_input(input("Market Probability Bounds % (e.g., 42-52 or 47): ").strip())
        lower, upper = parse_user_input(input("User Probability Bounds % (e.g., 16-42 or 29): ").strip())
        fee_input = input("Fee Rate % [Enter for 0%]: ").strip()
        fee_rate = float(fee_input) / 100.0 if fee_input else 0.0
        days_input = input("Days until resolution [Enter for 0]: ").strip()
        days_until = float(days_input) if days_input else 0.0
        bankroll_input = input(f"Bankroll [Enter for ${cfg.BANKROLL:,.2f}]: ").strip()
        calc_bankroll = float(bankroll_input) if bankroll_input else cfg.BANKROLL

        action, true_price, raw_kelly, eff_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
            lower, upper, m_lower, m_upper, fee_rate, calc_bankroll, base_ego
        )
        _, apy = calculate_annualized_yield(edge, true_price, days_until)
        pu_mid_c = round((lower + upper) / 2.0, 6)
        pm_mid_c = round((m_lower + m_upper) / 2.0, 6)
        lm_c = calculate_logistic_metrics(
            pu_mid_c, pm_mid_c, dynamic_ego, held_kelly=raw_kelly, fee_rate=fee_rate
        )

        c_vel_str = format_velocity_display(edge, true_price, days_until)

        print(f"\n--- CALCULATION RESULTS ---")
        print(f"ACTIVE PROB  : {lm_c['pu_norm']*100:.2f}% (Eff: Market {lm_c['eff_m']*100:.1f}% | Active {lm_c['eff_norm']*100:.1f}% | Regime: {lm_c['regime']})")
        print(f"VELOCITY     : {c_vel_str}")
        if action in ["YES", "NO"]:
            print(f"ACTION       : BUY {action} @ {true_price*100:.1f}% | Edge: {edge*100:+.2f}% | Ego: {dynamic_ego:.2f}")
            print(f"KELLY %      : {raw_kelly*100:.2f}% (${raw_kelly*calc_bankroll:,.2f}) | Eff-Kelly: {eff_kelly*100:.2f}% (${final_alloc:,.2f})")
        else:
            print(f"ACTION       : $0 Allocation | Edge: {edge*100:+.2f}% | Ego: {dynamic_ego:.2f}")

        if input("\nSave custom prediction to history? (y/n) [Enter=n]: ").strip().lower() == "y":
            q_title = input("Enter title: ").strip() or "Custom Market"
            custom_id = f"custom_{int(datetime.now().timestamp())}"
            pos_side_c = action.capitalize() if action in ("YES", "NO") else "Yes"
            is_no_c = (pos_side_c == "No")
            history.setdefault("predicted", {}).setdefault(custom_id, []).append({
                "question": f"[CUSTOM] {q_title}",
                "slug": "custom",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "pu": round(1.0 - pu_mid_c, 6) if is_no_c else pu_mid_c,
                "pm": round(true_price if true_price > 0 else (1.0 - pm_mid_c), 6) if is_no_c else pm_mid_c,
                "cur_pm": round(1.0 - pm_mid_c, 6) if is_no_c else pm_mid_c,
                "dynamic_ego": dynamic_ego,
                "kelly": raw_kelly,
                "edge": edge,
                "apy": apy,
                "conditionId": "custom",
                "position": pos_side_c,
                "source": "local",
                "cost_basis": round(final_alloc, 4),
                "shares": round(final_alloc / true_price, 4) if true_price > 0 else 0.0,
                "endDate": (datetime.now(timezone.utc) + timedelta(days=days_until)).strftime("%Y-%m-%dT%H:%M:%SZ") if days_until > 0 else None,
                "volume": 0.0,
                "volume24hr": 0.0,
            })
            save_history(history)
            print("[+] Saved to active predictions.")
    except ValueError as e:
        print(f"\n[!] Error: {e}.\n")


def compare_asset_market():
    """Interactive tool comparing binary prediction contracts vs. spot/perp asset holding."""
    print("\n--- ASSET vs. BINARY PREDICTION LEVERAGE & EFFICIENCY ---")
    title = input("Target / Question [e.g. 'ETH reaches $1500' or 'BTC reaches $100k']: ").strip() or "Asset Target"
    s0_in = input("Current Spot Price S0 ($) [e.g. 2250]: ").strip()
    k_in = input("Target Strike / Barrier K ($) [e.g. 1500]: ").strip()

    try:
        s0 = float(s0_in.replace("$", "").replace(",", ""))
        k = float(k_in.replace("$", "").replace(",", ""))
        if s0 <= 0 or k <= 0:
            print("[!] Spot and strike prices must be positive numbers.")
            return

        is_downside = (k < s0)
        asset_direction = "DOWNSIDE / SHORT" if is_downside else "UPSIDE / LONG"
        asset_pct_move = abs(k - s0) / s0

        side_in = input(f"Contract Side you want to buy: [1] YES (hits ${k:,.0f})  [2] NO (avoids ${k:,.0f}) [Enter=1]: ").strip()
        is_no = (side_in == "2" or side_in.lower() in ("no", "n"))
        contract_side = "NO" if is_no else "YES"

        pm_in = input(f"Polymarket {contract_side} Price % (e.g. 88 or 0.88): ").strip()
        pu_in = input(f"Your Subjective Probability % of {contract_side} winning [e.g. 98 or 0.98]: ").strip()
        days_in = input("Days to Expiry / Horizon [Enter for 30]: ").strip()

        pm_val = float(pm_in.replace("%", ""))
        pm = pm_val / 100.0 if pm_val > 1.0 else pm_val

        pu_val = float(pu_in.replace("%", ""))
        pu = pu_val / 100.0 if pu_val > 1.0 else pu_val

        days = float(days_in) if days_in else 30.0

        if pm <= 0 or pm >= 1.0 or pu <= 0 or pu > 1.0:
            print("[!] Probabilities must be between 0 and 100%.")
            return

        binary_roi = (1.0 - pm) / pm
        pct_return = binary_roi * 100.0
        edge = pu - pm
        ev_binary_roi = (edge / pm) * 100.0
        bin_daily_roi = (binary_roi / max(0.1, days)) * 100.0
        ann_ev_roi = (ev_binary_roi / max(0.1, days)) * 365.0

        AAVE_LT_LONG = 0.825
        AAVE_LT_SHORT = 0.850

        is_short_strat = (not is_no and is_downside)

        if not is_no:
            leverage_mult = binary_roi / asset_pct_move if asset_pct_move > 0 else 0.0
            bin_lev_str = f"{leverage_mult:.1f}x"
            bin_req_str = f"{'-' if is_downside else '+'}{asset_pct_move*100:.1f}%"
            bin_tgt_str = f"${k:,.2f}"
            bin_liq_str = "NONE"
            bin_risk_str = "Zero liquidation / No margin calls"
        else:
            leverage_mult = 1.0 / pm if pm > 0 else 1.0
            bin_lev_str = f"{leverage_mult:.1f}x"
            bin_req_str = f"Cush {asset_pct_move*100:.1f}%"
            bin_tgt_str = f"${k:,.2f}"
            bin_liq_str = "NONE"
            bin_risk_str = f"Pays +{pct_return:.1f}% if flat/down/cushion"

        table_rows = [
            {
                "strategy": f"Binary {contract_side} (Poly)",
                "lev": bin_lev_str,
                "req_move": bin_req_str,
                "target_px": bin_tgt_str,
                "liq_px": bin_liq_str,
                "risk": bin_risk_str,
            }
        ]

        strat_configs = [
            ("Spot (Unleveraged)", 1.0),
            ("Aave v4 Recollat (2x)", 2.0),
            ("Aave v4 Recollat (3x)", 3.0),
        ]

        for name, lev in strat_configs:
            req_move_val = binary_roi / lev
            if not is_short_strat:
                s_star_val = s0 * (1.0 + req_move_val)
                req_move_disp = f"+{req_move_val*100:.1f}%"
            else:
                s_star_val = s0 * max(0.0, 1.0 - req_move_val)
                req_move_disp = f"-{req_move_val*100:.1f}%"

            if lev == 1.0:
                liq_disp = "NONE"
                risk_disp = "Zero liquidation (loss only at $0)"
            else:
                if not is_short_strat:
                    s_liq_val = s0 * (lev - 1.0) / (lev * AAVE_LT_LONG)
                    drop_pct = (s_liq_val - s0) / s0 * 100.0
                    liq_disp = f"${s_liq_val:,.2f} ({drop_pct:+.1f}%)"
                    risk_tag = "HIGH RISK: " if lev >= 3.0 else ""
                    risk_disp = f"{risk_tag}Liquidated at {drop_pct:.1f}% dip (82.5% LT)"
                else:
                    s_liq_val = s0 * (lev * AAVE_LT_SHORT) / (lev - 1.0)
                    rally_pct = (s_liq_val - s0) / s0 * 100.0
                    liq_disp = f"${s_liq_val:,.2f} ({rally_pct:+.1f}%)"
                    risk_tag = "HIGH RISK: " if lev >= 3.0 else ""
                    risk_disp = f"{risk_tag}Liquidated at {rally_pct:+.1f}% rally (85% LT)"

            table_rows.append({
                "strategy": name,
                "lev": f"{lev:.1f}x",
                "req_move": req_move_disp,
                "target_px": f"${s_star_val:,.2f}",
                "liq_px": liq_disp,
                "risk": risk_disp,
            })

        box_w = 125
        print(f"\n" + "=" * box_w)
        print(" " * 35 + "[ ASSET vs. PREDICTION MARKET LEVERAGE & EFFICIENCY ]")
        print("=" * box_w)
        print(f"Target Event       : {title}")
        print(f"Spot Reference     : ${s0:,.2f} -> ${k:,.2f} ({asset_pct_move*100:.2f}% {asset_direction} Move)")
        print(f"Binary Position    : BUY {contract_side} @ {pm*100:.1f}c -> $1.00 (+{pct_return:.2f}% Payout)")
        print(f"Conviction         : Your Prob: {pu*100:.1f}% vs Market: {pm*100:.1f}% (Edge: {edge*100:+.2f}%)")
        print(f"Horizon            : {days:.1f} days ({bin_daily_roi:+.2f}%/day max ROI | EV Ann: {ann_ev_roi:+.1f}%)")
        print("-" * box_w)
        print(f"Breakeven Win Rate : {pm*100:.1f}% (Requires {contract_side} to resolve > {pm*100:.1f}% of the time)")
        print(f"Expected EV ROI    : {ev_binary_roi:+.2f}% per dollar committed (EV Payout: ${1.0 * (pu/pm):.2f} per $1.00)")
        print("-" * box_w)
        print("  Strategy                 | Lev   | Req Move   | Target Px ($S*) | Liq Price (Aave v4) | Margin & Liquidation Risk")
        print("---------------------------+-------+------------+-----------------+----------------------+-----------------------------------")
        for r in table_rows:
            print(f"  {r['strategy']:<25}| {r['lev']:>5} | {r['req_move']:>10} | {r['target_px']:>15} | {r['liq_px']:>20} | {r['risk']:<34}")
        print("-" * box_w)
        print("  * Aave v4 leverage models buy -> borrow -> recollateralize looping (ETH 82.5% Liquidation Threshold).")
        print("  * Verdict & allocation decision left to operator based on risk tolerance, liquidation buffer & payout asymmetry.")
        print("=" * box_w + "\n")

    except ValueError as e:
        print(f"\n[!] Input error: {e}.\n")
