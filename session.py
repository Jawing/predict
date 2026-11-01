"""
session.py - Prediction session and custom calculator execution.
Handles interactive market streaming, Kelly calculations, VWAP orderbook walking,
and custom hypothetical market calculations.
"""

import json
from datetime import datetime, timezone
import pandas as pd

from config import BANKROLL, MIN_EDGE, api
import config as cfg
from utils import ensure_list, format_time_remaining
from storage import load_history, save_history, stringify_overflow
from model import (
    calculate_base_ego,
    calculate_allocation,
    calculate_annualized_yield,
    calculate_logistic_metrics,
    parse_user_input,
    calculate_vwap,
)
from market import (
    generate_market_stream,
    extract_prices,
    refresh_prices_from_clob,
    stream_clob_book,
)
from views import print_review_table, print_prediction_stats


def run_prediction_session(
    mode="discover",
    sub_mode="all",
    target_slugs=None,
    category="All",
    max_offset=0,
    exclude_mode="none",
    discovery_order="volume",
):
    global BANKROLL

    history = load_history()
    save_history(stringify_overflow(history))

    if mode == "review":
        print_review_table(history, sub_mode)
        return None

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
    )

    portfolio_data = []
    session_trades = {}
    session_markets = []
    current_index = 0

    while True:
        if current_index >= len(session_markets):
            try:
                new_m = next(market_stream)

                pm_bid, pm_ask, pm_mid = extract_prices(new_m)
                if pm_mid is None:
                    continue
                new_m["cached_bid"], new_m["cached_ask"], new_m["cached_mid"] = (
                    pm_bid,
                    pm_ask,
                    pm_mid,
                )

                new_m["original_state"] = {
                    "in_pred": new_m.get("id") in history["predicted"],
                    "pred_data": history["predicted"].get(new_m.get("id")),
                    "in_skip": new_m.get("id") in history["skipped"],
                    "skip_data": history["skipped"].get(new_m.get("id")),
                    "portfolio_len": len(portfolio_data),
                }
                session_markets.append(new_m)

            except StopIteration:
                print("\n[*] Data stream ended or exhausted.")
                break
            except TypeError:
                break

        m = session_markets[current_index]
        m = refresh_prices_from_clob(m)
        pm_bid, pm_ask, pm_mid = extract_prices(m)
        m["cached_bid"], m["cached_ask"], m["cached_mid"] = pm_bid, pm_ask, pm_mid

        market_id = m.get("id")
        question = m.get("question", "unknown")
        event_slug = m.get("parent_slug", "unknown")
        event_url = f"https://polymarket.com/event/{event_slug}"
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        try:
            res_dt = datetime.fromisoformat(m.get("endDate").replace("Z", "+00:00"))
            exact_date_str = res_dt.strftime("%B %d, %Y")

            seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
            days_str = format_time_remaining(seconds_left)
        except:
            exact_date_str, days_str = "unknown", "unknown"

        pm_bid, pm_ask, pm_mid = m["cached_bid"], m["cached_ask"], m["cached_mid"]

        print(f"==================================================")
        print(f"Event    : {m.get('parent_name')}")
        print(f"URL      : {event_url}")
        print(f"Market   : {question}")
        print(f"Resolves : {exact_date_str} ({days_str})\n")

        while True:
            user_input = input(
                "Enter % bounds ('36-48' or '42'), 'l' (live), 's' (skip), 'r' (redo), 'q' (quit): "
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
            current_index += 1
            continue

        try:
            lower, upper = parse_user_input(user_input)
            fee_rate = (m.get("feeSchedule") or {}).get("rate")
            if fee_rate is None:
                try:
                    f = float(m.get("fee") or 0)
                    fee_rate = f / 1e18 if f > 1 else f
                except:
                    fee_rate = 0.0

            action, true_price, raw_kelly, eff_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
                lower, upper, pm_bid, pm_ask, fee_rate, current_bankroll, base_ego
            )

            try:
                res_dt = datetime.fromisoformat(m.get("endDate", "").replace("Z", "+00:00"))
                seconds_left = (res_dt - datetime.now(timezone.utc)).total_seconds()
                days_until = max(0.0, seconds_left / 86400.0)
            except:
                days_until = 0.0

            roi, apy = calculate_annualized_yield(edge, true_price, days_until)

            # Orderbook VWAP calculation
            vwap_true = true_price
            slippage = 0.0
            adjusted_edge = edge
            adjusted_apy = apy

            if action in ["YES", "NO"] and final_alloc > 0:
                clob_ids_str = m.get("clobTokenIds")
                asks_list = []
                if clob_ids_str:
                    try:
                        clob_ids = (
                            json.loads(clob_ids_str)
                            if isinstance(clob_ids_str, str)
                            else clob_ids_str
                        )
                        token_id = None
                        tokens = m.get("tokens", [])
                        if tokens and len(tokens) == len(clob_ids):
                            for idx, tok in enumerate(tokens):
                                if tok.get("outcome", "").upper() == action:
                                    token_id = clob_ids[idx]
                                    break
                        if not token_id:
                            token_id = clob_ids[0] if action == "YES" else clob_ids[1]
                        r = api.get(
                            f"https://clob.polymarket.com/book?token_id={token_id}", timeout=5
                        )
                        if r.status_code == 200:
                            book = r.json()
                            asks_list = book.get("asks", [])
                    except Exception:
                        pass
                if asks_list:
                    vwap, total_spent = calculate_vwap(asks_list, final_alloc)
                    if vwap > 0:
                        vwap_true = vwap + (fee_rate * vwap * (1.0 - vwap))
                        slippage = max(0.0, vwap_true - true_price)

                        win_prob = true_price + edge
                        adjusted_edge = win_prob - vwap_true
                        _, adjusted_apy = calculate_annualized_yield(
                            adjusted_edge, vwap_true, days_until
                        )
                        if adjusted_edge < MIN_EDGE:
                            action = "THIN_EDGE"

            if market_id not in history["predicted"]:
                history["predicted"][market_id] = []
            else:
                history["predicted"][market_id] = ensure_list(history["predicted"][market_id])

            pu_mid_val = (lower + upper) / 2.0
            lm = calculate_logistic_metrics(
                pu_mid_val, pm_mid, dynamic_ego, held_kelly=raw_kelly, fee_rate=fee_rate
            )

            history["predicted"][market_id].append({
                "question": question,
                "slug": event_slug,
                "date": today_str,
                "pu": pu_mid_val,
                "pm": pm_mid,
                "dynamic_ego": dynamic_ego,
                "kelly": raw_kelly,
                "edge": adjusted_edge,
                "apy": adjusted_apy,
                "conditionId": m.get("conditionId"),
                "position": action.capitalize() if action in ("YES", "NO") else ("Yes" if lm["is_yes"] else "No"),
                "source": "local",
            })

            history["skipped"].pop(market_id, None)

            print(f"\n--- MARKET ANALYSIS ---")
            print(
                f"Spread       : Bid {pm_bid*100:.1f}% | Ask {pm_ask*100:.1f}% (Spread: {(pm_ask - pm_bid)*100:.1f}%)"
            )
            print(
                f"Conviction   : User [{lower*100:.1f}%-{upper*100:.1f}%] -> Active Target: {lm['pu_norm']*100:.1f}% (Ego: {dynamic_ego:.2f})"
            )
            print(
                f"Efficiency   : {lm['eff_m']*100:.1f}% [4p(1-p)] | Regime: {lm['regime']} ({lm['dist_to_decel']*100:+.1f}% to 78.9% Decel)"
            )

            if action in ["YES", "NO"]:
                cumulative_exposure += final_alloc / current_bankroll
                session_trades[market_id] = final_alloc

                print(f"Action       : BUY {action} @ {true_price*100:.1f}% | Edge: {adjusted_edge*100:+.2f}%" + (f" | APY: {adjusted_apy*100:.1f}%" if days_until > 0 else ""))
                if slippage > 0.0001:
                    print(f"Execution    : VWAP {vwap_true*100:.2f}% (Slippage: +{slippage*100:.2f}%)")
                print(f"Kelly Target : {raw_kelly*100:.2f}% (${raw_kelly*current_bankroll:,.2f}) | Eff-Kelly: {eff_kelly*100:.2f}% (${final_alloc:,.2f})")
                print(f"Allocation   : ${final_alloc:,.2f} ({eff_kelly*100:.2f}% Eff-Kelly) | Exposure: {cumulative_exposure*100:.1f}%")

                portfolio_data.append({
                    "Question": question[:50] + "..",
                    "Action": action,
                    "Ego": f"{dynamic_ego:.2f}",
                    "Price": f"{vwap_true*100:.1f}%",
                    "Kelly": f"{raw_kelly*100:.2f}%",
                    "EffKelly": f"{eff_kelly*100:.2f}%",
                    "Alloc": f"${final_alloc:,.2f}",
                })
            else:
                if action == "THIN_EDGE":
                    reason = f"Edge below {MIN_EDGE*100}% threshold"
                elif action == "NONE":
                    reason = "Trapped inside bid-ask spread"
                else:
                    reason = "Unknown"

                print(f"Action       : $0 ({reason}) | Edge: {adjusted_edge*100:+.2f}%")

            print(
                f"Session      : {len(session_trades)} placed (${sum(session_trades.values()):,.2f} total) | Bankroll: ${current_bankroll:,.2f}\n"
            )
            print_prediction_stats(history, sub_mode="predicted", base_ego=base_ego)

            current_index += 1

        except ValueError as e:
            print(f"\n[!] Error: {e}. Please try again.\n")

    save_history(stringify_overflow(history))
    return pd.DataFrame(portfolio_data)


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
        m_lower, m_upper = parse_user_input(
            input("Market Probability Bounds % (e.g., 42-52 or 47): ").strip()
        )
        lower, upper = parse_user_input(
            input("User Probability Bounds % (e.g., 16-42 or 29): ").strip()
        )
        fee_input = input("Fee Rate % [Enter for 0%]: ").strip()
        fee_rate = float(fee_input) / 100.0 if fee_input else 0.0
        days_input = input("Days until resolution [Enter for 0]: ").strip()
        days_until = float(days_input) if days_input else 0.0
        bankroll_input = input(f"Bankroll [Enter for ${BANKROLL:,.2f}]: ").strip()
        calc_bankroll = float(bankroll_input) if bankroll_input else BANKROLL

        action, true_price, raw_kelly, eff_kelly, final_alloc, dynamic_ego, edge = calculate_allocation(
            lower, upper, m_lower, m_upper, fee_rate, calc_bankroll, base_ego
        )
        _, apy = calculate_annualized_yield(edge, true_price, days_until)
        pu_mid_c = round((lower + upper) / 2.0, 6)
        pm_mid_c = round((m_lower + m_upper) / 2.0, 6)
        lm_c = calculate_logistic_metrics(
            pu_mid_c, pm_mid_c, dynamic_ego, held_kelly=raw_kelly, fee_rate=fee_rate
        )

        print(f"\n--- CALCULATION RESULTS ---")
        print(
            f"ACTIVE PROB  : {lm_c['pu_norm']*100:.2f}% (Eff: Market {lm_c['eff_m']*100:.1f}% | Active {lm_c['eff_norm']*100:.1f}% | Regime: {lm_c['regime']})"
        )
        if action in ["YES", "NO"]:
            print(
                f"ACTION       : BUY {action} @ {true_price*100:.1f}% | Edge: {edge*100:+.2f}% | Ego: {dynamic_ego:.2f}"
            )
            if days_until > 0:
                print(f"APY          : {apy*100:.2f}%")
            print(
                f"KELLY %      : {raw_kelly*100:.2f}% (${raw_kelly*calc_bankroll:,.2f}) | Eff-Kelly: {eff_kelly*100:.2f}% (${final_alloc:,.2f})"
            )
        else:
            print(f"ACTION       : $0 Allocation | Edge: {edge*100:+.2f}% | Ego: {dynamic_ego:.2f}")

        if input("\nSave custom prediction to history? (y/n) [Enter=n]: ").strip().lower() == "y":
            q_title = input("Enter title: ").strip() or "Custom Market"
            custom_id = f"custom_{int(datetime.now().timestamp())}"
            history.setdefault("predicted", {}).setdefault(custom_id, []).append({
                "question": f"[CUSTOM] {q_title}",
                "slug": "custom",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "pu": pu_mid_c,
                "pm": pm_mid_c,
                "dynamic_ego": dynamic_ego,
                "kelly": raw_kelly,
                "edge": edge,
                "apy": apy,
                "conditionId": "custom",
                "source": "local",
            })
            save_history(stringify_overflow(history))
            print("[+] Saved to active predictions.")
    except ValueError as e:
        print(f"\n[!] Error: {e}.\n")
