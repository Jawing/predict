import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
"""manage.py - History data management and manual resolution utilities.

Allows selective deletion of markets, category clears, manual resolution overrides,
and interactive settings/filters configuration.
"""

import os
from datetime import datetime, timezone, timedelta
import config as cfg
from utils import ensure_list, get_history_counts, parse_market_date
from storage import load_history, save_history, get_active_profile, get_profile_path
from wallet import update_resolutions
from model import calculate_base_ego, calculate_trade_ego_metrics, restore_snapshot_to_prediction


def delete_specific_markets(history):
    """Interactively lists markets across categories and deletes selected rows or tranches."""
    counts = get_history_counts(history)
    pred_count = counts["active_markets"]
    ex_count = counts["exited_markets"]
    res_count = counts["resolved_markets"]
    skip_count = counts["skipped"]

    print(
        f"\nFilter view: [1] All ({pred_count+ex_count+res_count+skip_count})  [2] Active ({pred_count})  [3] Exited ({ex_count})  [4] Resolved ({res_count})  [5] Skipped ({skip_count})"
    )
    f_choice = input("> ").strip()
    target_cats = ["predicted", "exited", "resolved", "skipped"]
    if f_choice == "2":
        target_cats = ["predicted"]
    elif f_choice == "3":
        target_cats = ["exited"]
    elif f_choice == "4":
        target_cats = ["resolved"]
    elif f_choice == "5":
        target_cats = ["skipped"]

    indexed_markets = []
    print(f"\n{'#':<4} | {'CAT':<9} | {'MARKET ID':<20} | {'QUESTION'}")
    print("-" * 80)

    for cat in target_cats:
        for m_id, item in history.get(cat, {}).items():
            if isinstance(item, list) and len(item) > 0:
                q = item[0].get("question", "Unknown")
                snaps_0 = item[0].get("snapshots")
                has_snaps = isinstance(snaps_0, list) and len(snaps_0) > 1
                if len(item) > 1:
                    sub_info = f" ({len(item)} tranches)"
                elif has_snaps:
                    sub_info = f" ({len(snaps_0)} updates)"
                else:
                    sub_info = ""
            elif isinstance(item, dict):
                q = item.get("question", "Unknown")
                sub_info = ""
            else:
                q = "Unknown"
                sub_info = ""

            idx = len(indexed_markets) + 1
            indexed_markets.append((cat, m_id, item))
            print(f"{idx:<4} | {cat.upper():<9} | {str(m_id):<20} | {q[:45]}{sub_info}")
            if isinstance(item, list) and len(item) > 1:
                for t_i, t_p in enumerate(item, 1):
                    src = t_p.get("source", "local").upper()
                    t_pos = t_p.get("position", "Yes")
                    t_cb = float(t_p.get("cost_basis", 0.0) or 0.0)
                    t_sh = float(t_p.get("shares", 0.0) or 0.0)
                    t_k = float(t_p.get("kelly", 0.0) or 0.0)
                    t_date = str(t_p.get("date", ""))[:16]
                    t_is = "Staged Tranche" if t_p.get("is_tranche") else ("Wallet Holding" if src == "WALLET" else "Initial Position")
                    print(f"      |-- {idx}.{t_i}: [{src}] {t_is} ({t_pos}) | Kelly: {t_k*100:.1f}% | ${t_cb:,.2f} ({t_sh:,.1f} sh) | {t_date}")
            elif isinstance(item, list) and len(item) == 1:
                p_0 = item[0]
                snaps = p_0.get("snapshots")
                if isinstance(snaps, list) and len(snaps) > 1:
                    for s_i, s in enumerate(snaps, 1):
                        s_pu = float(s.get("pu", 0.0))
                        s_pm = float(s.get("pm", 0.0))
                        s_date = str(s.get("date", ""))[:16]
                        s_k = float(s.get("kelly", 0.0) or 0.0)
                        k_str = f" | Kelly: {s_k*100:.1f}%" if s_k > 0 else ""
                        s_tag = "Initial Prediction" if s_i == 1 else f"Update #{s_i - 1}"
                        s_src = s.get("source", p_0.get("source", "local")).upper()
                        print(f"      |-- {idx}.{s_i}: [{s_src}] {s_tag} | pu: {s_pu*100:.1f}% | pm: {s_pm*100:.1f}%{k_str} | {s_date}")

    if not indexed_markets:
        print("[!] No markets found in selected filter.")
        return

    print("\nSelect market(s) or update(s) to delete / undo:")
    print(" - Single index: '4' (deletes entire market #4)")
    print(" - Multiple indices: '1, 3, 5' or '1-4'")
    print(" - Specific update / tranche: '4.2' (market #4, update #2 / tranche #2)")
    print(" - Press Enter to cancel")

    del_input = input("> ").strip()
    if not del_input:
        print("[!] Deletion cancelled.")
        return

    to_delete = []
    try:
        parts = del_input.replace(",", " ").split()
        for p in parts:
            if "-" in p and "." not in p:
                start, end = map(int, p.split("-"))
                for i in range(start, end + 1):
                    if 1 <= i <= len(indexed_markets):
                        to_delete.append((indexed_markets[i - 1], "market", None))
            elif "." in p:
                m_idx, sub_idx = map(int, p.split("."))
                if 1 <= m_idx <= len(indexed_markets):
                    entry = indexed_markets[m_idx - 1]
                    cat, m_id, item = entry
                    if isinstance(item, list) and len(item) > 1:
                        if 1 <= sub_idx <= len(item):
                            to_delete.append((entry, "tranche", sub_idx - 1))
                    elif isinstance(item, list) and len(item) == 1:
                        snaps = item[0].get("snapshots", [])
                        if isinstance(snaps, list) and len(snaps) > 1:
                            if sub_idx == 1:
                                to_delete.append((entry, "market", None))
                            elif 1 < sub_idx <= len(snaps):
                                to_delete.append((entry, "snapshot", sub_idx - 1))
            else:
                i = int(p)
                if 1 <= i <= len(indexed_markets):
                    to_delete.append((indexed_markets[i - 1], "market", None))
    except ValueError:
        print("[!] Invalid input format. Deletion cancelled.")
        return

    if not to_delete:
        print("[!] No valid markets or updates selected.")
        return

    print("\nMarkets / Updates marked for deletion / rollback:")
    for (cat, m_id, item), op_type, target_idx in to_delete:
        q = ensure_list(item)[0].get("question", "Unknown") if item else "Unknown"
        if op_type == "tranche":
            desc = f" [Tranche #{target_idx + 1}]"
        elif op_type == "snapshot":
            desc = f" [Undo Update #{target_idx} -> Rollback to prior state]"
        else:
            desc = " [Entire Market]"
        print(f" - [{cat.upper()}] {m_id}: {q[:45]}{desc}")

    confirm = input(
        f"\nAre you sure you want to proceed with these {len(to_delete)} action(s)? (y/n): "
    ).strip().lower()
    if confirm != "y":
        print("[!] Operation cancelled.")
        return

    for (cat, m_id, item), op_type, target_idx in to_delete:
        if cat in history and m_id in history[cat]:
            cat_dict = history[cat]
            if op_type == "market":
                cat_dict.pop(m_id, None)
            elif op_type == "tranche":
                curr_list = ensure_list(cat_dict[m_id])
                new_list = [p for idx, p in enumerate(curr_list) if idx != target_idx]
                if new_list:
                    cat_dict[m_id] = new_list
                else:
                    cat_dict.pop(m_id, None)
            elif op_type == "snapshot":
                curr_list = ensure_list(cat_dict[m_id])
                if curr_list:
                    p_target = curr_list[0]
                    snaps = p_target.get("snapshots", [])
                    if isinstance(snaps, list) and 0 <= target_idx < len(snaps):
                        snaps.pop(target_idx)
                        if snaps:
                            prev = snaps[-1]
                            base_ego = calculate_base_ego(history, verbose=False)
                            restore_snapshot_to_prediction(p_target, prev, base_ego=base_ego)
                        else:
                            cat_dict.pop(m_id, None)

    save_history(history)
    print(f"[+] Successfully processed {len(to_delete)} item(s) in prediction history.\n")


def clear_specific_category(history):
    """Empties all entries from a selected history category."""
    counts = get_history_counts(history)
    print("\nSelect Category to Clear:")
    print(f" 1: Active Predictions ({counts['active_markets']} mkts, {counts['active']} tranches)")
    print(f" 2: Exited Positions ({counts['exited_markets']} mkts, {counts['exited']} tranches)")
    print(f" 3: Resolved Markets ({counts['resolved_markets']} mkts, {counts['resolved']} tranches)")
    print(f" 4: Skipped Markets ({counts['skipped']})")
    print(" 5: Cancel")

    cat_sel = input("> ").strip()
    cat_map = {"1": "predicted", "2": "exited", "3": "resolved", "4": "skipped"}
    if cat_sel not in cat_map:
        print("[!] Cancelled.")
        return

    target_cat = cat_map[cat_sel]
    count = len(history.get(target_cat, {}))
    if count == 0:
        print(f"[!] '{target_cat}' is already empty.")
        return

    confirm = input(
        f"WARNING: Type 'CONFIRM' to clear all {count} '{target_cat}' markets: "
    ).strip()
    if confirm == "CONFIRM":
        history[target_cat] = {}
        save_history(history)
        print(f"[+] All '{target_cat}' markets cleared from history.\n")
    else:
        print("[!] Category clear cancelled.")




def prune_closed_skipped_markets(history, verbose=True):

    """Audits history['skipped'] against Polymarket Gamma API and removes closed/resolved markets."""

    skipped = history.get("skipped", {})

    if not skipped:

        if verbose:

            print("[*] No skipped markets in history to prune.")

        return 0



    all_ids = list(skipped.keys())

    closed_ids = set()

    chunk_size = 40



    if verbose:

        print(f"[*] Auditing {len(all_ids)} skipped markets against Polymarket Gamma API...")



    for i in range(0, len(all_ids), chunk_size):

        chunk = all_ids[i:i + chunk_size]

        q_ids = "&".join(f"id={mid}" for mid in chunk if not str(mid).startswith("custom_"))

        if not q_ids:

            continue

        try:

            url = f"{cfg.GAMMA_API}/markets?closed=true&{q_ids}"

            resp = cfg.api.get(url, timeout=8).json()

            if isinstance(resp, list):

                for item in resp:

                    if item.get("closed") or item.get("umaResolutionStatus") in ("resolved", "finalized"):

                        closed_ids.add(str(item.get("id")))

        except Exception as e:

            if verbose:

                print(f"[!] Warning checking chunk: {e}")



    # Also check end dates for any remaining

    now_utc = datetime.now(timezone.utc)

    for mid, item in list(skipped.items()):

        if str(mid) in closed_ids:

            continue

        if isinstance(item, dict):

            end_d = item.get("endDate") or item.get("end_date")

            if end_d:

                res_dt, _, _ = parse_market_date(end_d)

                if res_dt and res_dt < now_utc - timedelta(days=2):

                    closed_ids.add(str(mid))



    pruned_count = 0

    for mid in closed_ids:

        if mid in skipped:

            skipped.pop(mid, None)

            pruned_count += 1



    if pruned_count > 0:

        save_history(history)

        if verbose:

            print(f"[+] Successfully pruned {pruned_count} closed/resolved markets from Skipped ({len(skipped)} remaining active).\n")

    else:

        if verbose:

            print(f"[*] All {len(skipped)} skipped markets are currently active. None pruned.\n")



    return pruned_count





def handle_delete_data():
    """Menu interface for full reset, selective deletion, clearing category, or manual resolution."""
    history = load_history()
    counts = get_history_counts(history)
    total_markets = counts["total_markets"]

    print("\n--- DATA MANAGEMENT ---")
    print(" 1: Delete all data (Full profile reset)")
    print(f" 2: Selectively delete specific markets ({total_markets} in history)")
    print(" 3: Clear specific category (Active, Exited, Resolved, or Skipped)")
    print(" 4: Manually Resolve / Override a Specific Market")
    print(" 5: Select / Toggle Markets Included in Ego Calibration")
    print(" 0: Back to Tools Menu")

    sub_choice = input("> ").strip()

    if sub_choice == "1":
        active_pid = get_active_profile()
        confirm = input(f"WARNING: Type 'CONFIRM' to reset all data for profile '{active_pid}': ").strip()
        if confirm == "CONFIRM":
            p_path = get_profile_path(active_pid)
            if os.path.exists(p_path):
                os.remove(p_path)
            if os.path.exists(cfg.HISTORY_FILE):
                os.remove(cfg.HISTORY_FILE)
            load_history(active_pid)
            print(f"[+] All data for profile '{active_pid}' reset.")
        else:
            print("[!] Data deletion cancelled.")

    elif sub_choice == "2":
        if total_markets == 0:
            print("[!] History is empty. No data to delete.")
            return
        delete_specific_markets(history)

    elif sub_choice == "3":
        if total_markets == 0:
            print("[!] History is empty. No data to delete.")
            return
        clear_specific_category(history)

    elif sub_choice == "4":
        manual_resolve(history)

    elif sub_choice == "5":
        toggle_ego_market_inclusion(history)

    elif sub_choice in ["0", ""]:
        print("[!] Cancelled.")
    else:
        print("[!] Invalid selection.")


def toggle_ego_market_inclusion(history):
    """Allows selecting specific markets to include/exclude or custom-adjust Kelly % for Base Ego calibration."""
    sort_mode = "ego_impact"  # Default: Ego impact (detractors first)

    while True:
        ws = history.get("wallet_summary", {}) if isinstance(history, dict) else {}
        live_bankroll = float(ws.get("bankroll") or cfg.BANKROLL) if isinstance(ws, dict) else cfg.BANKROLL
        if live_bankroll <= 0:
            live_bankroll = 100.0
        is_live_w = bool(isinstance(history, dict) and str(history.get("profile_id") or history.get("wallet_address") or "").startswith("0x"))
        all_settled = []
        for cat in ["resolved", "exited"]:
            for m_id, preds in history.get(cat, {}).items():
                for idx, p in enumerate(ensure_list(preds)):
                    q = p.get("question", m_id)
                    side = p.get("position", "-")
                    pnl = float(p.get("realized_pnl") if p.get("realized_pnl") is not None else p.get("pnl", 0.0) or 0.0)
                    is_excl = bool(p.get("exclude_from_ego", False))
                    
                    # Unmask exclude_from_ego temporarily to get intrinsic metrics & weight
                    p["exclude_from_ego"] = False
                    m_ego = calculate_trade_ego_metrics(p, cat=cat, bankroll=live_bankroll, is_live_wallet=is_live_w)
                    p["exclude_from_ego"] = is_excl

                    # Concluded / settled date resolution (never in future)
                    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                    d_res = p.get("resolved_date") or p.get("exit_date")
                    if d_res and str(d_res) <= now_str:
                        settled_date = str(d_res)[:16]
                    else:
                        d_raw = p.get("date")
                        if d_raw and str(d_raw)[:16] <= now_str:
                            settled_date = str(d_raw)[:16]
                        else:
                            # Fallback if date is in future (contract endDate horizon)
                            ts = p.get("last_synced_ts")
                            if ts and int(ts) > 0:
                                settled_date = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                            else:
                                settled_date = str(d_raw)[:16] if d_raw else "unknown" 

                    all_settled.append({
                        "cat": cat,
                        "m_id": m_id,
                        "idx": idx,
                        "question": q,
                        "side": side,
                        "pnl": pnl,
                        "excluded": is_excl,
                        "ref": p,
                        "date": str(settled_date),
                        "weight": m_ego["weight"],
                        "bs_u": m_ego["bs_u"],
                        "bs_m": m_ego["bs_m"],
                        "ego_drag": m_ego["ego_drag"],
                        "net_adv": m_ego["net_adv"],
                    })

        if not all_settled:
            print("[!] No resolved or exited markets available to calibrate.")
            return

        # Exact active sums & current Base Ego
        S_m = sum(it["weight"] * it["bs_m"] for it in all_settled if not it["excluded"])
        S_u = sum(it["weight"] * it["bs_u"] for it in all_settled if not it["excluded"])
        curr_ego = (S_m / (S_u + S_m)) if (S_u + S_m) > 0 else 0.50

        # Exact counterfactual Delta Ego if toggled
        for it in all_settled:
            w = it["weight"]
            bs_u = it["bs_u"]
            bs_m = it["bs_m"]
            if not it["excluded"]:
                new_Sm = S_m - (w * bs_m)
                new_Su = S_u - (w * bs_u)
            else:
                new_Sm = S_m + (w * bs_m)
                new_Su = S_u + (w * bs_u)
            new_ego = (new_Sm / (new_Su + new_Sm)) if (new_Su + new_Sm) > 0 else 0.50
            it["delta_ego"] = new_ego - curr_ego

        # Sort settled markets
        if sort_mode == "ego_impact":
            all_settled.sort(key=lambda x: (0 if x["excluded"] else 1, x["delta_ego"]), reverse=True)
            sort_label = "Ego Impact / Delta Ego (Detractors First)"
        elif sort_mode == "adv":
            all_settled.sort(key=lambda x: (0 if x.get("net_adv") is None else 1, x.get("net_adv") or 0.0), reverse=True)
            sort_label = "Net Advantage (Benefactors First)"
        elif sort_mode == "kelly":
            all_settled.sort(key=lambda x: float(x.get("weight", 0.0) or 0.0), reverse=True)
            sort_label = "Kelly %"
        elif sort_mode == "pnl":
            all_settled.sort(key=lambda x: x["pnl"], reverse=True)
            sort_label = "Realized PnL"
        elif sort_mode == "date":
            all_settled.sort(key=lambda x: x["date"], reverse=True)
            sort_label = "Settled Date (Newest First)"

        w_mode = getattr(cfg, "EGO_WEIGHTING", "kelly").upper()

        print("\n" + "=" * 114)
        print(f"--- EGO CALIBRATION MARKET SELECTOR [Sorted by {sort_label}] ---")
        active_cnt = sum(1 for x in all_settled if not x["excluded"])
        print(f"Current Base Ego: {curr_ego:.3f} ({w_mode} mode) | Active: {active_cnt} / {len(all_settled)} markets")
        print("=" * 114)
        print(f"{'#':<3} | {'Status':<10} | {'dEgo':<8} | {'Net Adv':<9} | {'Kelly':<7} | {'Type':<8} | {'Side':<6} | {'PnL':<10} | {'Question'}")
        print("-" * 114)
        for i, it in enumerate(all_settled, 1):
            st = "[EXCLUDED]" if it["excluded"] else "[ACTIVE]"
            k_val = float(it.get("weight", 0.0) or 0.0)
            adv_str = f"{it['net_adv']:>+8.4f}"
            dego_str = f"{it['delta_ego']:>+7.4f}"
            print(f"{i:<3} | {st:<10} | {dego_str} | {adv_str} | {k_val*100:>5.2f}% | {it['cat']:<8} | {it['side'][:6]:<6} | ${it['pnl']:>+8.2f} | {it['question'][:36]}")

        sel = input(f"\nSelect row # to toggle, 'adj <#>' (edit Kelly), 'sort' (toggle view), 'reset', or Enter to finish: ").strip()
        if not sel:
            break
        elif sel.lower() == "reset":
            for it in all_settled:
                it["ref"]["exclude_from_ego"] = False
            save_history(history)
            new_ego = calculate_base_ego(history, verbose=False)
            print(f"[+] All markets restored to Active in Base Ego. Updated Base Ego: {curr_ego:.3f} -> {new_ego:.3f}")
        elif sel.lower() in ("sort", "s"):
            next_sort = {
                "ego_impact": "adv",
                "adv": "kelly",
                "kelly": "pnl",
                "pnl": "date",
                "date": "ego_impact",
            }
            sort_mode = next_sort.get(sort_mode, "ego_impact")
            continue
        elif sel.lower().startswith("sort "):
            sub_s = sel.lower().split()[1]
            if sub_s in ("ego", "impact", "drag", "dego"):
                sort_mode = "ego_impact"
            elif sub_s in ("adv", "edge"):
                sort_mode = "adv"
            elif sub_s in ("kelly", "k"):
                sort_mode = "kelly"
            elif sub_s in ("pnl", "profit"):
                sort_mode = "pnl"
            elif sub_s in ("date", "d", "time"):
                sort_mode = "date"
            continue
        elif sel.lower().startswith("adj "):
            parts = sel.split()
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1]) - 1
                if 0 <= idx < len(all_settled):
                    target_item = all_settled[idx]
                    curr_k = float(target_item.get("weight", 0.0) or 0.0)
                    new_k_raw = input(f"Enter new Kelly % for '{target_item['question'][:30]}' (currently {curr_k*100:.2f}%, e.g. 1.0 for 1%): ").strip()
                    try:
                        clean_k = new_k_raw.strip()
                        if clean_k.endswith("%"):
                            parsed_num = float(clean_k[:-1].strip()) / 100.0
                        else:
                            raw_num = float(clean_k)
                            parsed_num = (raw_num / 100.0) if raw_num >= 1.0 else raw_num
                        new_k_val = max(0.0, min(1.0, parsed_num))
                        target_item["ref"]["kelly"] = new_k_val
                        target_item["ref"]["entry_kelly"] = new_k_val
                        target_item["ref"]["actual_kelly"] = new_k_val
                        save_history(history)
                        new_ego = calculate_base_ego(history, verbose=False)
                        delta = new_ego - curr_ego
                        print(f"[+] Market #{parts[1]} Kelly updated to {new_k_val*100:.2f}%. Base Ego updated: {curr_ego:.3f} -> {new_ego:.3f} ({delta:+.3f})")
                    except ValueError:
                        print("[!] Invalid percentage entered.")
            else:
                print("[!] Format: 'adj <row#>'")
        elif sel.isdigit():
            idx = int(sel) - 1
            if 0 <= idx < len(all_settled):
                curr = all_settled[idx]["ref"].get("exclude_from_ego", False)
                all_settled[idx]["ref"]["exclude_from_ego"] = not curr
                save_history(history)
                new_st = "EXCLUDED from" if not curr else "INCLUDED in"
                new_ego = calculate_base_ego(history, verbose=False)
                delta = new_ego - curr_ego
                print(f"[+] Market #{sel} is now {new_st} Base Ego calculation.")
                print(f"[*] Base Ego updated: {curr_ego:.3f} -> {new_ego:.3f} ({delta:+.3f})")
            else:
                print("[!] Invalid row number.")
        else:
            print("[!] Unrecognized input.")

def manual_resolve(history):
    """Audit resolutions automatically or prompt user to resolve a market manually."""
    print("\n--- RESOLVE MARKETS ---")
    print(" 1: Auto-Audit Unresolved Markets via Polymarket API")
    print(" 2: Manually Resolve a Specific Market")
    print(" 3: Cancel")
    r_mode = input("> ").strip()
    if r_mode == "1":
        history = update_resolutions(history)
        save_history(history)
        return history
    elif r_mode != "2":
        return history

    combined_open = {}
    for m_id, preds in history.get("predicted", {}).items():
        combined_open[m_id] = ("predicted", preds)
    for m_id, preds in history.get("exited", {}).items():
        combined_open[m_id] = ("exited", preds)

    if not combined_open:
        print("[!] No active or exited predictions to resolve.")
        return history

    for m_id, (b_name, preds) in combined_open.items():
        q = ensure_list(preds)[0].get("question", "Unknown")
        print(f" [{b_name.upper()}] ID: {m_id} | {q}")

    sel = input("\nEnter Market ID to resolve [Enter to cancel]: ").strip()
    if not sel or sel not in combined_open:
        return history

    bucket_name, preds = combined_open[sel]
    primary_p = ensure_list(preds)[0]
    pos_side = primary_p.get("position", "Yes")
    print(f"\nResolving: {primary_p.get('question')} (Position: {pos_side})")
    if pos_side == "No":
        print(" 1: WON (1.0) - Market Resolved NO   2: LOST (0.0) - Market Resolved YES   3: HALF / SPLIT (0.5)")
    else:
        print(" 1: WON (1.0) - Market Resolved YES  2: LOST (0.0) - Market Resolved NO    3: HALF / SPLIT (0.5)")
    out_sel = input("> ").strip()
    outcome = {"1": 1.0, "2": 0.0, "3": 0.5}.get(out_sel)
    if outcome is None:
        print("[!] Resolve cancelled.")
        return history

    history.setdefault("resolved", {}).setdefault(sel, [])
    for p in ensure_list(preds):
        p["outcome"] = outcome
        if p.get("realized_pnl") is None or float(p.get("realized_pnl", 0.0) or 0.0) == 0.0:
            cb = float(p.get("cost_basis", 0.0) or 0.0)
            sh = float(p.get("shares", 0.0) or 0.0)
            ws = history.get("wallet_summary", {}) if isinstance(history, dict) else {}
            active_b = float(ws.get("bankroll") or cfg.BANKROLL) if isinstance(ws, dict) else cfg.BANKROLL
            k_eval = float(p.get("entry_kelly") or p.get("kelly", 0.0) or 0.0)
            if cb == 0.0 and k_eval > 0:
                cb = k_eval * active_b
                if sh == 0.0 and p.get("pm", 0.0):
                    sh = cb / float(p["pm"])
            if sh > 0 or cb > 0:
                if outcome == 1.0:
                    p["realized_pnl"] = round(sh * 1.0 - cb, 2)
                elif outcome == 0.0:
                    p["realized_pnl"] = round(-cb, 2)
                elif outcome == 0.5:
                    p["realized_pnl"] = round(sh * 0.5 - cb, 2)
        history["resolved"][sel].append(p)

    history[bucket_name].pop(sel, None)
    save_history(history)
    print("[+] Market manually resolved and saved.")
    return history


def settings_menu():
    """Interactive settings and thresholds configurator."""
    active_pid = get_active_profile()
    is_wallet = active_pid.startswith("0x")
    history = load_history(active_pid)

    print("\n--- SETTINGS / FILTERS ---")
    print(f" 1: Min Edge       = {cfg.MIN_EDGE*100:.2f}%  (skip bets below this edge)")
    print(f" 2: Max Days       = {cfg.MAX_DAYS}  (ignore markets beyond this horizon)")
    print(f" 3: Min Days       = {cfg.MIN_DAYS}  (ignore markets resolving sooner than this)")
    print(f" 4: Extreme Odds   = {cfg.EXTREME_ODDS*100:.2f}%  (tail-risk filter - skip near 0% or 100%)")

    if is_wallet:
        bankroll_add = float(history.get("bankroll_add", 0.0) or history.get("wallet_summary", {}).get("bankroll_add", 0.0) or 0.0)
        print(f" 5: Bankroll Add   = +${bankroll_add:,.2f}  (off-platform funds added to live wallet total, default $0)")
    else:
        print(f" 5: Paper Bankroll = ${cfg.BANKROLL:,.2f}  (sandbox capital)")

    print("\nEnter setting number to edit [Enter to cancel]:")
    s_choice = input("> ").strip()
    if s_choice == "1":
        curr_pct = cfg.MIN_EDGE * 100.0
        new_raw = input(f"Enter new Min Edge in % (e.g. '1' for 1.0%, '1.5' for 1.5%, currently {curr_pct:.2f}%): ").strip()
        if new_raw:
            try:
                val_clean = new_raw.replace("%", "").strip()
                raw_f = float(val_clean)
                new_val = max(0.0, raw_f / 100.0 if ("%" in new_raw or raw_f >= 0.05) else raw_f)
                cfg.MIN_EDGE = new_val
                history.setdefault("settings", {})["MIN_EDGE"] = new_val
                save_history(history, profile_id=active_pid)
                print(f"[+] Min Edge updated to {new_val*100:.2f}% (edge hurdle: {new_val:.4f}).")
            except ValueError:
                print("[!] Invalid percentage - setting unchanged.")
    elif s_choice == "2":
        new_raw = input(f"Enter Max Days horizon (currently {cfg.MAX_DAYS}): ").strip()
        if new_raw:
            try:
                new_val = max(0, int(float(new_raw)))
                cfg.MAX_DAYS = new_val
                history.setdefault("settings", {})["MAX_DAYS"] = new_val
                save_history(history, profile_id=active_pid)
                print(f"[+] Max Days updated to {new_val} days.")
            except ValueError:
                print("[!] Invalid integer - setting unchanged.")
    elif s_choice == "3":
        new_raw = input(f"Enter Min Days horizon (currently {cfg.MIN_DAYS}): ").strip()
        if new_raw:
            try:
                new_val = max(0, int(float(new_raw)))
                cfg.MIN_DAYS = new_val
                history.setdefault("settings", {})["MIN_DAYS"] = new_val
                save_history(history, profile_id=active_pid)
                print(f"[+] Min Days updated to {new_val} days.")
            except ValueError:
                print("[!] Invalid integer - setting unchanged.")
    elif s_choice == "4":
        curr_pct = cfg.EXTREME_ODDS * 100.0
        new_raw = input(f"Enter Extreme Odds filter in % (e.g. '5' for 5%, '0' to disable, currently {curr_pct:.2f}%): ").strip()
        if new_raw:
            try:
                val_clean = new_raw.replace("%", "").strip()
                raw_f = float(val_clean)
                new_val = max(0.0, min(0.45, raw_f / 100.0 if ("%" in new_raw or raw_f >= 0.5) else raw_f))
                cfg.EXTREME_ODDS = new_val
                history.setdefault("settings", {})["EXTREME_ODDS"] = new_val
                save_history(history, profile_id=active_pid)
                print(f"[+] Extreme Odds filter updated to {new_val*100:.2f}% (skips odds < {new_val*100:.1f}% or > {(1.0-new_val)*100:.1f}%).")
            except ValueError:
                print("[!] Invalid percentage - setting unchanged.")
    elif s_choice == "5":
        if is_wallet:
            curr_add = float(history.get("bankroll_add", 0.0) or history.get("wallet_summary", {}).get("bankroll_add", 0.0) or 0.0)
            new_raw = input(f"Enter additional off-platform bankroll in $ (currently +${curr_add:,.2f}, default 0): ").strip()
            if new_raw:
                try:
                    new_add = max(0.0, float(new_raw))
                    history["bankroll_add"] = new_add
                    cash_val = float(history.get("wallet_summary", {}).get("cash", 0.0) or 0.0)
                    pos_val = float(history.get("wallet_summary", {}).get("portfolio", 0.0) or 0.0)
                    cfg.BANKROLL = round(cash_val + pos_val + new_add, 4)
                    history.setdefault("wallet_summary", {})["bankroll_add"] = new_add
                    history["wallet_summary"]["bankroll"] = cfg.BANKROLL
                    save_history(history, profile_id=active_pid)
                    print(f"[+] Off-platform bankroll add set to +${new_add:,.2f}. Total Bankroll is now ${cfg.BANKROLL:,.2f}.")
                except ValueError:
                    print("[!] Invalid dollar amount.")
        else:
            new_raw = input(f"New paper sandbox bankroll (currently ${cfg.BANKROLL:,.2f}): ").strip()
            if new_raw:
                try:
                    new_val = max(1.0, float(new_raw))
                    cfg.BANKROLL = new_val
                    history.setdefault("wallet_summary", {})["bankroll"] = new_val
                    history["wallet_summary"]["cash"] = new_val
                    save_history(history, profile_id=active_pid)
                    print(f"[+] Paper bankroll updated to ${new_val:,.2f}.")
                except ValueError:
                    print("[!] Invalid value.")
