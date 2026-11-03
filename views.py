"""Portfolio and history display tables, aggregate variance metrics, and top-of-menu dashboard."""
import math
import config as cfg
from utils import ensure_list
from model import (
    LOGISTIC_DECEL_LOW, LOGISTIC_DECEL_HIGH, LOGISTIC_HALF_LOW, LOGISTIC_HALF_HIGH,
    calculate_logistic_metrics, calculate_base_ego
)


def calculate_prediction_stats(history, sub_mode="predicted", base_ego=None):
    """Calculates weighted portfolio stats: balance percentage, effective N, expected win rate, variance."""
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

    items = []
    for p, state in preds_list:
        pos_side = p.get("position", "Yes")
        pu = float(p.get("pu", 0.5))
        if pos_side == "No" and pu < 0.5:
            pu = 1.0 - pu
        pm = float(p.get("pm", 0.5))
        cur_pm = p.get("cur_pm")
        kelly = max(0.0, float(p.get("kelly", 0.0)))
        if kelly < 1e-9:
            kelly = 0.0
        ego = float(p.get("dynamic_ego", base_ego))
        outcome = p.get("outcome")
        is_exited = (state == "exited") or bool(p.get("exited"))
        is_res = (state == "resolved")

        lm = calculate_logistic_metrics(pu, pm, ego, held_kelly=kelly, cur_pm=cur_pm)
        win_prob_bal = lm["win_prob_bal"]
        var_i = win_prob_bal * (1.0 - win_prob_bal)
        alloc_i = (kelly * cfg.BANKROLL) if state == "active" else 0.0
        eff_alloc_i = (lm["eff_kelly"] * cfg.BANKROLL) if state == "active" else 0.0
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
            "eff_kelly": lm["eff_kelly"],
            "active_edge": lm["active_edge"],
            "live_active_edge": lm["live_active_edge"],
            "eff_m": lm["eff_m"],
            "eff_live": lm["eff_live"],
            "eff_norm": lm["eff_norm"],
            "regime": "EXITED" if is_exited else lm["regime"],
            "is_exited": is_exited,
            "realized_pnl": float(p.get("realized_pnl", 0.0) or 0.0),
            "alloc": alloc_i,
            "eff_alloc": eff_alloc_i,
            "win_prob_bal": win_prob_bal,
            "win_prob_u": lm["win_prob_u"],
            "win_prob_m": lm["win_prob_m"],
            "var_i": var_i,
            "source": p.get("source", "local"),
            "is_resolved": is_res,
            "outcome": float(outcome) if (is_res and outcome is not None) else None
        })

    sized_items = [x for x in items if x["kelly"] > 0]
    total_kelly = sum(x["kelly"] for x in items)
    total_eff_kelly = sum(x["eff_kelly"] for x in items)
    total_alloc = sum(x["alloc"] for x in items)
    total_eff_alloc = sum(x["eff_alloc"] for x in items)
    total_pnl = sum(x["realized_pnl"] for x in items)
    local_count = sum(1 for x in items if x["source"] != "wallet")
    wallet_count = sum(1 for x in items if x["source"] == "wallet")

    if total_kelly > 0 and sized_items:
        weights = [x["kelly"] / total_kelly for x in sized_items]
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
    avg_active_edge = sum(w * x["active_edge"] for w, x in zip(weights, used_items))

    norm_variance = sum((w ** 2) * x["var_i"] for w, x in zip(weights, used_items))
    norm_std_dev = math.sqrt(norm_variance)
    sum_w_sq = sum(w ** 2 for w in weights)
    effective_n = (1.0 / sum_w_sq) if sum_w_sq > 0 else 0.0

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
        "bankroll": cfg.BANKROLL,
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
        "items": items
    }


def print_prediction_stats(history, sub_mode="predicted", base_ego=None):
    """Prints balanced portfolio summary block."""
    stats = calculate_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego)
    if not stats:
        return
    n_denom = stats['active_count'] if stats['active_count'] > 0 else stats['total_count']
    bal_pct = (stats['effective_n'] / n_denom * 100.0) if n_denom > 0 else 0.0
    status_lbl = {"predicted": "active", "exited": "exited", "resolved": "resolved", "all": "total"}.get(sub_mode, "active")
    sized_note = f" ({stats['active_count']} sized)" if stats['active_count'] != stats['total_count'] else ""

    print(f"--- PORTFOLIO & PREDICTION STATS [{status_lbl.upper()}] ---")
    print(f"Predictions      : {stats['total_count']} {status_lbl}{sized_note} ({stats['local_count']} LOCAL, {stats['wallet_count']} WALLET)")
    print(f"Effective N      : {stats['effective_n']:.2f} / {n_denom} equal-sized bets ({bal_pct:.1f}% size-balanced)")
    w_sum = history.get("wallet_summary")
    w_addr = history.get("wallet_address") or cfg.POLYGON_WALLET
    if isinstance(w_sum, dict) and "cash" in w_sum and "portfolio" in w_sum:
        w_label = f"\033]8;;https://polymarket.com/profile/{w_addr}\033\\Wallet Bankroll\033]8;;\033\\" if w_addr else "Wallet Bankroll"
        print(f"{w_label}  : ${stats['bankroll']:,.2f} (Cash: ${w_sum['cash']:,.2f} | Open Positions: ${w_sum['portfolio']:,.2f})")

    if sub_mode == "predicted":
        print(f"Model Exposure   : ${stats['total_eff_alloc']:,.2f} / ${stats['bankroll']:,.2f} ({stats['total_eff_kelly']*100:.1f}% Eff-Kelly)")
    else:
        avg_k = stats['total_kelly'] / max(1, stats['active_count'])
        wins = sum(1 for x in stats["items"] if x["outcome"] == 1.0)
        losses = sum(1 for x in stats["items"] if x["outcome"] == 0.0)
        wl_str = f" | Record: {wins}W - {losses}L" if sub_mode == "resolved" else ""
        print(f"Realized PnL     : ${stats['total_pnl']:+,.2f}{wl_str} | Avg Sizing: {avg_k*100:.2f}% Kelly")

    print(f"Expected Win Rate: {stats['exp_win_bal']*100:.2f}% Balanced (User: {stats['exp_win_u']*100:.2f}% | Market: {stats['exp_win_m']*100:.2f}%)")
    if sub_mode == "resolved":
        print(f"Logistic Eff.    : Entry {stats['avg_eff_m']*100:.1f}% | Pooled (p_norm) {stats['avg_eff_norm']*100:.1f}% | Avg Entry Edge: {stats['avg_active_edge']*100:+.2f}%\n")
    else:
        print(f"Logistic Eff.    : Entry {stats['avg_eff_m']*100:.1f}% | Live {stats['avg_eff_live']*100:.1f}% | Pooled (p_norm) {stats['avg_eff_norm']*100:.1f}% | Avg Edge: {stats['avg_active_edge']*100:+.2f}%\n")


def print_review_table(history, sub_mode):
    """Renders formatted history table with ANSI hyperlinks and sorting."""
    base_ego = calculate_base_ego(history, verbose=False)
    TABLE_CONFIGS = {
        "predicted": {
            "headers": ["#", "Question", "Action / Signal", "Held Pos", "Kelly", "Edge", "Side", "Live", "p_norm", "Date", "Src"],
            "widths": [3, 34, 22, 13, 7, 7, 4, 6, 6, 10, 8],
            "aligns": ["right", "left", "left", "right", "right", "right", "left", "right", "right", "left", "left"],
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
            "widths": [3, 34, 8, 22, 13, 7, 7, 4, 9, 6, 10],
            "aligns": ["right", "left", "left", "left", "right", "right", "right", "left", "right", "right", "left"],
            "q_idx": 1
        }
    }

    rows = []

    if sub_mode in ["predicted", "predicted_skipped", "all"]:
        for market_id, value in history.get("predicted", {}).items():
            for pred in ensure_list(value):
                side_val = pred.get("position", "Yes")
                pu_v = float(pred.get("pu", 0.5))
                if side_val == "No" and pu_v < 0.5:
                    pu_v = 1.0 - pu_v
                pm_v = float(pred.get("pm", 0.5))
                cur_pm_v = pred.get("cur_pm")
                k_v = float(pred.get("kelly", 0.0))
                ego_v = float(pred.get("dynamic_ego", base_ego))

                # Mark-to-Market holding fraction of live bankroll
                sh = float(pred.get("shares", 0.0) or 0.0)
                cur_p = float(cur_pm_v if cur_pm_v is not None else pm_v)
                is_wallet = (pred.get("source", "local") == "wallet")
                if is_wallet:
                    if sh > 0 and cur_p > 0:
                        live_val = sh * cur_p
                    elif pred.get("cost_basis"):
                        cb = float(pred.get("cost_basis", 0.0) or 0.0)
                        entry_p = float(pred.get("pm", cur_p) or cur_p)
                        live_val = cb * (cur_p / entry_p if entry_p > 0 else 1.0)
                    else:
                        live_val = k_v * cfg.BANKROLL
                    held_mtm_k = (live_val / cfg.BANKROLL) if cfg.BANKROLL > 0 else k_v
                else:
                    live_val = 0.0
                    held_mtm_k = 0.0

                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v, cur_pm=cur_pm_v)
                tgt_v = lm["target_win_m"]
                delta_k = max(0.0, lm["eff_kelly"] - held_mtm_k) if held_mtm_k > 0 else lm["eff_kelly"]

                delta_usd = delta_k * cfg.BANKROLL if cfg.BANKROLL > 0 else 0.0
                buy_usd = lm["eff_kelly"] * cfg.BANKROLL if cfg.BANKROLL > 0 else 0.0

                if not is_wallet:
                    regime_str = f"EXEC BUY ${buy_usd:,.0f} ({lm['eff_kelly']*100:.1f}%)" if lm["live_active_edge"] > 0 else "SKIP: EDGE GONE"
                elif lm["regime"] == "OVER-TGT" or lm["live_active_edge"] <= 0:
                    regime_str = f"SELL: OVER-TGT({lm['live_active_edge']*100:+.1f}%)"
                elif delta_k >= 0.005:
                    tag = "DISC " if lm["regime"] == "AGAINST" else ""
                    regime_str = f"ADD +${delta_usd:,.0f} (+{delta_k*100:.1f}%)"
                else:
                    regime_str = f"HOLD (+{lm['dist_to_target']*100:.1f}% to exit)"
                w_tag = pred.get("wallet")
                alias_tag = history.get("alias")
                if pred.get("source") == "wallet" and alias_tag:
                    src_val = alias_tag[:8]
                elif w_tag and len(w_tag) >= 6:
                    src_val = f"0x..{w_tag[-4:]}"
                else:
                    src_val = pred.get("source", "local").upper()
                is_pos_yes = (side_val == "Yes")
                pu_disp = pu_v
                p_norm_disp = lm["win_prob_bal"]
                pm_disp = pm_v
                live_disp = float(cur_pm_v) if cur_pm_v is not None else pm_v

                rows.append({
                    "Mode": "active",
                    "Source": src_val,
                    "Side": side_val,
                    "Market ID": market_id,
                    "Date": str(pred.get("date", "unknown"))[:10],
                    "Question": pred.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{pred.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "HeldVal": live_val,
                    "HeldFrac": held_mtm_k,
                    "Shares": sh,
                    "Edge": float(pred.get("edge", lm["live_active_edge"]) or lm["live_active_edge"]),
                    "DistDecel": lm["dist_to_target"],
                    "pu": pu_disp,
                    "p_norm": p_norm_disp,
                    "pm": pm_disp,
                    "cur_pm": live_disp,
                    "Target": tgt_v,
                    "PnL": pred.get("realized_pnl"),
                    "Regime": regime_str,
                    "Outcome": ""
                })

    if sub_mode in ["exited", "all"]:
        for market_id, value in history.get("exited", {}).items():
            for pred in ensure_list(value):
                pu_v = float(pred.get("pu", 0.5))
                pm_v = float(pred.get("pm", 0.5))
                cur_pm_v = pred.get("cur_pm")
                k_v = float(pred.get("kelly", 0.0))
                ego_v = float(pred.get("dynamic_ego", base_ego))
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v, cur_pm=cur_pm_v)
                is_pos_yes = lm["is_yes"]
                side_val = pred.get("position", "Yes" if is_pos_yes else "No")
                if side_val == "No":
                    pu_disp = lm["win_prob_u"]
                    p_norm_disp = lm["win_prob_bal"]
                    pm_disp = lm["win_prob_m"]
                    live_disp = lm["eval_win_m"]
                else:
                    pu_disp = pu_v
                    p_norm_disp = lm["pu_norm"]
                    pm_disp = pm_v
                    live_disp = float(cur_pm_v) if cur_pm_v is not None else pm_v

                w_tag = pred.get("wallet")
                alias_tag = history.get("alias")
                if pred.get("source") == "wallet" and alias_tag:
                    src_val = alias_tag[:8]
                elif w_tag and len(w_tag) >= 6:
                    src_val = f"0x..{w_tag[-4:]}"
                else:
                    src_val = pred.get("source", "wallet").upper()

                rows.append({
                    "Mode": "exited",
                    "Source": src_val,
                    "Side": side_val,
                    "Market ID": market_id,
                    "Date": str(pred.get("date", "unknown"))[:10],
                    "Question": pred.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{pred.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "Edge": lm["active_edge"],
                    "DistDecel": lm["dist_to_target"],
                    "pu": pu_disp,
                    "p_norm": p_norm_disp,
                    "pm": pm_disp,
                    "cur_pm": live_disp,
                    "Target": lm["target_win_m"],
                    "PnL": float(pred.get("realized_pnl", 0.0) or 0.0),
                    "Regime": "EXITED (Await Res)",
                    "Outcome": ""
                })

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


    if sub_mode in ["resolved", "all"]:
        for market_id, value in history.get("resolved", {}).items():
            for res in ensure_list(value):
                pu_v = float(res.get("pu", 0.5))
                pm_v = float(res.get("pm", 0.5))
                k_v = float(res.get("kelly", 0.0))
                ego_v = float(res.get("dynamic_ego", base_ego))
                lm = calculate_logistic_metrics(pu_v, pm_v, ego_v, held_kelly=k_v)
                w_tag = res.get("wallet")
                alias_tag = history.get("alias")
                if res.get("source") == "wallet" and alias_tag:
                    src_val = alias_tag[:8]
                elif w_tag and len(w_tag) >= 6:
                    src_val = f"0x..{w_tag[-4:]}"
                else:
                    src_val = res.get("source", "local").upper()
                is_pos_yes = lm["is_yes"]
                side_val = res.get("position", "Yes" if is_pos_yes else "No")
                if side_val == "No":
                    pu_disp = lm["win_prob_u"]
                    p_norm_disp = lm["win_prob_bal"]
                    pm_disp = lm["win_prob_m"]
                else:
                    pu_disp = pu_v
                    p_norm_disp = lm["pu_norm"]
                    pm_disp = pm_v

                rows.append({
                    "Mode": "resolved",
                    "Source": src_val,
                    "Side": side_val,
                    "Market ID": market_id,
                    "Date": str(res.get("date", "unknown"))[:10],
                    "Question": res.get("question", "unknown"),
                    "Web Link": f"https://polymarket.com/event/{res.get('slug', 'unknown')}",
                    "Kelly": k_v,
                    "EffKelly": lm["eff_kelly"],
                    "Edge": lm["active_edge"],
                    "DistDecel": lm["dist_to_target"],
                    "pu": pu_disp,
                    "p_norm": p_norm_disp,
                    "pm": pm_disp,
                    "cur_pm": pm_disp,
                    "Target": lm["target_win_m"],
                    "PnL": res.get("realized_pnl"),
                    "Regime": lm["regime"],
                    "Outcome": str(res.get("outcome", 0.5))
                })

    if not rows:
        print(f"\n[!] No markets found in '{sub_mode}'.")
        return

    sort_by = "date"
    if sub_mode in ["predicted", "exited", "resolved", "all"]:
        print(f"\nSort '{sub_mode}' by: [1] Date (default)  [2] Kelly  [3] Edge  [4] Exit Proximity")
        sort_choice = input("> ").strip()
        sort_by = {"2": "kelly", "3": "edge", "4": "exit_prox"}.get(sort_choice, "date")

    if sort_by == "date":
        rows.sort(key=lambda x: x.get("Date", "") if x.get("Date", "") != "unknown" else "", reverse=True)
    elif sort_by == "kelly":
        rows.sort(key=lambda x: (0 if x.get("Kelly") is None else 1, x.get("Kelly") or 0.0), reverse=True)
    elif sort_by == "edge":
        rows.sort(key=lambda x: (0 if x.get("Edge") is None else 1, x.get("Edge") or 0.0), reverse=True)
    elif sort_by == "exit_prox":
        rows.sort(key=lambda x: (1 if x.get("DistDecel") is None else 0, x.get("DistDecel") if x.get("DistDecel") is not None else float("inf")))

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
        eff_kelly_val = r.get("EffKelly")
        pu_val = r.get("pu")
        p_norm_val = r.get("p_norm")
        pm_val = r.get("pm")
        cur_pm_val = r.get("cur_pm")
        tgt_val = r.get("Target")
        pnl_val = r.get("PnL")
        regime_val = r.get("Regime", "-")
        side_str = r.get("Side", "-")
        src_str = r.get("Source", "-")

        edge_v = r.get("Edge")

        kelly_str = f"{kelly_val*100:.2f}%" if kelly_val is not None else "-"
        eff_kelly_str = f"{eff_kelly_val*100:.2f}%" if eff_kelly_val is not None else "-"
        edge_str = f"{edge_v*100:+.1f}%" if edge_v is not None else "-"
        pu_str = f"{pu_val*100:.1f}%" if pu_val is not None else "-"
        p_norm_str = f"{p_norm_val*100:.1f}%" if p_norm_val is not None else "-"
        pm_str = f"{pm_val*100:.1f}%" if pm_val is not None else "-"
        live_str = f"{cur_pm_val*100:.1f}%" if cur_pm_val is not None else "-"
        tgt_str = f"{tgt_val*100:.1f}%" if tgt_val is not None else "-"
        pnl_str = f"${pnl_val:+,.2f}" if pnl_val is not None else "-"

        outcome_val = r.get("Outcome")
        is_wallet_row = "0x" in src_str or src_str == "WALLET"
        if outcome_val == "1.0": outcome_str = "WON" if is_wallet_row else "YES"
        elif outcome_val == "0.0": outcome_str = "LOST" if is_wallet_row else "NO"
        elif outcome_val == "0.5": outcome_str = "HALF"
        else: outcome_str = "-"

        row_num = str(row_i)
        if sub_mode == "predicted":
            held_v = r.get("HeldVal", 0.0)
            held_f = r.get("HeldFrac", 0.0)
            held_str = f"${held_v:,.0f} ({held_f*100:.1f}%)" if held_v > 0 else "-"
            vals = [row_num, r["Question"], regime_val, held_str, kelly_str, edge_str, side_str, live_str, p_norm_str, r["Date"], src_str]
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
                row_parts.append(f"\033]8;;{r['Web Link']}\033\\{q_padded}\033]8;;\033\\")
            else:
                row_parts.append(format_cell(v, w, a))
        print(" | ".join(row_parts))

    print("=" * len(header_str) + "\n")
    if sub_mode in ["predicted", "exited", "resolved", "all"]:
        print_prediction_stats(history, sub_mode=sub_mode, base_ego=base_ego)

    # Pure Voluntary Action Bar
    if sub_mode == "predicted":
        sell_signals = [r for r in rows if "SELL" in r.get("Regime", "")]
        exit_tag = f"  [*] Active Exit: {len(sell_signals)} position{'s' if len(sell_signals) > 1 else ''} reached target (Market Sell on Polymarket recommended)\n" if sell_signals else ""
        if exit_tag:
            print(exit_tag, end="")
        act = input(f"Select row # to predict / add tranche (1-{len(rows)}) [Enter for Main Menu]: ").strip()
        if act.isdigit():
            _handle_targeted_tranche_predict(rows, preselected_idx=int(act) - 1)

    elif sub_mode in ["exited", "resolved", "skipped", "all"] and rows:
        act = input(f"Select row # to predict / add tranche (1-{len(rows)}) [Enter for Main Menu]: ").strip()
        if act.isdigit():
            _handle_targeted_tranche_predict(rows, preselected_idx=int(act) - 1)


def _handle_targeted_tranche_predict(rows, preselected_idx=None):
    """Launches targeted prediction / tranche addition on a specific table row."""
    if preselected_idx is not None:
        sel_idx = preselected_idx
    else:
        sel_str = input(f"Select row # to predict / add tranche (1-{len(rows)}) [Enter to cancel]: ").strip()
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


def _strip_ansi(text):
    import re
    return re.sub(r'\033\]8;;.*?\033\\|\033\[[0-9;]*m', '', text)


def print_main_menu_dashboard(history):
    """Renders the top-level status bar and summary metrics for the main menu without redundancies."""
    base_ego = calculate_base_ego(history, verbose=False)
    n_pred = sum(len(ensure_list(v)) for v in history.get("predicted", {}).values())
    n_exit = sum(len(ensure_list(v)) for v in history.get("exited", {}).values())
    n_res = sum(len(ensure_list(v)) for v in history.get("resolved", {}).values())
    n_skip = len(history.get("skipped", {}))
    
    from storage import get_active_profile, get_profile_display_name
    active_pid = get_active_profile()
    is_wallet = active_pid.startswith("0x")
    ws = history.get("wallet_summary", {})
    cash_val = float(ws.get("cash", 0.0) or 0.0)
    pos_val = float(ws.get("portfolio", 0.0) or 0.0)

    alias = history.get("alias", "")
    if not alias:
        if active_pid.startswith("0xbC73"): alias = "Main"
        elif active_pid == "local": alias = "Paper"

    if is_wallet:
        prof_link = f"\033]8;;https://polymarket.com/profile/{active_pid}\033\\{active_pid[:6]}...{active_pid[-4:]}\033]8;;\033\\"
        alias_str = f"{alias} [{prof_link}]" if alias else prof_link
        cap_line = f"  Capital State  : ${cfg.BANKROLL:,.2f} Total (Cash ${cash_val:,.2f} | Open Positions ${pos_val:,.2f})"
        hdr_tag = f"PROFILE: {alias_str} | Polymarket Live Portfolio"
    else:
        alias_str = f"{alias} ({active_pid})" if alias else active_pid
        cap_line = f"  Capital State  : ${cfg.BANKROLL:,.2f} Paper Bankroll (Sandbox)"
        hdr_tag = f"PROFILE: {alias_str} | Paper Trading Sandbox"
    
    active_eff_kelly = 0.0
    active_edge_w = 0.0
    active_signals = {}
    for preds in history.get("predicted", {}).values():
        for p in ensure_list(preds):
            k = max(0.0, float(p.get("kelly", 0.0) or 0.0))
            is_w = (p.get("source") == "wallet")
            sh = float(p.get("shares", 0.0) or 0.0)
            cur_p = float(p.get("cur_pm") if p.get("cur_pm") is not None else p.get("pm", 0.5))
            if is_w:
                if sh > 0 and cur_p > 0:
                    live_val = sh * cur_p
                elif p.get("cost_basis"):
                    cb = float(p.get("cost_basis", 0.0) or 0.0)
                    entry_p = float(p.get("pm", cur_p) or cur_p)
                    live_val = cb * (cur_p / entry_p if entry_p > 0 else 1.0)
                else:
                    live_val = k * cfg.BANKROLL
                held_mtm_k = live_val / cfg.BANKROLL if cfg.BANKROLL > 0 else k
            else:
                live_val = 0.0
                held_mtm_k = 0.0

            pos_side = p.get("position", "Yes")
            pu_dash = float(p.get("pu", 0.5))
            if pos_side == "No" and pu_dash < 0.5:
                pu_dash = 1.0 - pu_dash
            ego_dash = float(p.get("dynamic_ego") if p.get("dynamic_ego") is not None else base_ego)
            lm = calculate_logistic_metrics(
                pu_dash, p.get("pm"), ego_dash,
                held_kelly=k, cur_pm=p.get("cur_pm")
            )

            eff_k = lm["eff_kelly"]
            delta_k = max(0.0, eff_k - held_mtm_k) if held_mtm_k > 0 else eff_k

            active_eff_kelly += eff_k
            active_edge_w += eff_k * float(p.get("edge", lm["live_active_edge"]) or lm["live_active_edge"])

            if not is_w:
                sig = "BUY" if lm["live_active_edge"] > 0 else "PASS"
            elif lm["regime"] == "OVER-TGT" or lm["live_active_edge"] <= 0:
                sig = "SELL"
            elif delta_k >= 0.005:
                sig = "ADD"
            else:
                sig = "HOLD"
            active_signals[sig] = active_signals.get(sig, 0) + 1
            
    avg_edge = (active_edge_w / active_eff_kelly) if active_eff_kelly > 0 else 0.0
    sig_summary = " ".join(f"{s}:{cnt}" for s, cnt in sorted(active_signals.items())) if active_signals else "-"

    wins, losses = 0, 0
    pnl_total = 0.0
    for preds in history.get("resolved", {}).values():
        for p in ensure_list(preds):
            out = p.get("outcome")
            if out == 1.0: wins += 1
            elif out == 0.0: losses += 1
            pnl_total += float(p.get("realized_pnl", 0.0) or 0.0)
    for preds in history.get("exited", {}).values():
        for p in ensure_list(preds):
            pnl_total += float(p.get("realized_pnl", 0.0) or 0.0)

    total_games = wins + losses
    win_pct_str = f" ({wins/total_games*100:.1f}%)" if total_games > 0 else ""
    pnl_str = f"+${pnl_total:,.2f}" if pnl_total >= 0 else f"-${abs(pnl_total):,.2f}"
    skip_str = f" | {n_skip} skipped" if n_skip > 0 else ""

    box_w = 96
    hdr_visible_len = len(_strip_ansi(hdr_tag))
    hdr_pad = max(0, (box_w - hdr_visible_len - 4) // 2)
    hdr_line = " " * hdr_pad + f"[ {hdr_tag} ]"

    print("\n" + "=" * box_w)
    print(hdr_line)
    print("-" * box_w)
    print(cap_line)
    print(f"  Live Exposure  : {active_eff_kelly*100:.1f}% Eff-Kelly (${active_eff_kelly*cfg.BANKROLL:,.2f} across {n_pred} active) | Avg Edge: {avg_edge*100:+.2f}% | Signals: {sig_summary}")
    print(f"  Track Record   : {wins}W - {losses}L{win_pct_str} | Realized PnL: {pnl_str} across {n_res} resolved + {n_exit} exited | Base Ego: {base_ego:.3f}{skip_str}")
    print("=" * box_w)

