"""manage.py - History data management and manual resolution utilities.

Allows selective deletion of markets, category clears, manual resolution overrides,
and interactive settings/filters configuration.
"""

import os
import config
from utils import ensure_list, get_history_counts
from storage import load_history, save_history, get_active_profile, get_profile_path
from wallet import update_resolutions


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
                tranche_info = f" ({len(item)} tranches)" if len(item) > 1 else ""
            elif isinstance(item, dict):
                q = item.get("question", "Unknown")
                tranche_info = ""
            else:
                q = "Unknown"
                tranche_info = ""

            idx = len(indexed_markets) + 1
            indexed_markets.append((cat, m_id, item))
            print(f"{idx:<4} | {cat.upper():<9} | {str(m_id):<20} | {q[:45]}{tranche_info}")

    if not indexed_markets:
        print("[!] No markets found in selected filter.")
        return

    print("\nSelect market(s) to delete:")
    print(" - Single index: '4'")
    print(" - Multiple indices: '1, 3, 5' or '1-4'")
    print(" - Specific tranche of a market: '4.2' (market #4, tranche #2)")
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
                        to_delete.append((indexed_markets[i - 1], None))
            elif "." in p:
                m_idx, t_idx = map(int, p.split("."))
                if 1 <= m_idx <= len(indexed_markets):
                    to_delete.append((indexed_markets[m_idx - 1], t_idx - 1))
            else:
                i = int(p)
                if 1 <= i <= len(indexed_markets):
                    to_delete.append((indexed_markets[i - 1], None))
    except ValueError:
        print("[!] Invalid input format. Deletion cancelled.")
        return

    if not to_delete:
        print("[!] No valid markets selected.")
        return

    print("\nMarkets marked for deletion:")
    for (cat, m_id, item), t_idx in to_delete:
        q = ensure_list(item)[0].get("question", "Unknown") if item else "Unknown"
        t_str = f" [Tranche #{t_idx + 1}]" if t_idx is not None else " [Entire Market]"
        print(f" - [{cat.upper()}] {m_id}: {q[:50]}{t_str}")

    confirm = input(
        f"\nAre you sure you want to delete these {len(to_delete)} item(s)? (y/n): "
    ).strip().lower()
    if confirm != "y":
        print("[!] Deletion cancelled.")
        return

    grouped_deletions = {}
    for (cat, m_id, item), t_idx in to_delete:
        grouped_deletions.setdefault((cat, m_id), []).append(t_idx)

    for (cat, m_id), t_indices in grouped_deletions.items():
        if cat in history and m_id in history[cat]:
            cat_dict = history[cat]
            if None in t_indices:
                cat_dict.pop(m_id, None)
            else:
                curr_list = ensure_list(cat_dict[m_id])
                del_indices = set(t_indices)
                new_list = [p for idx, p in enumerate(curr_list) if idx not in del_indices]
                if new_list:
                    cat_dict[m_id] = new_list
                else:
                    cat_dict.pop(m_id, None)

    save_history(history)
    print(f"[+] Successfully deleted {len(to_delete)} item(s) from prediction history.\n")


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
            if os.path.exists(config.HISTORY_FILE):
                os.remove(config.HISTORY_FILE)
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
    """Allows selecting specific markets to include or exclude from Base Ego Brier score calibration."""
    all_settled = []
    for cat in ["resolved", "exited"]:
        for m_id, preds in history.get(cat, {}).items():
            for idx, p in enumerate(ensure_list(preds)):
                q = p.get("question", m_id)
                side = p.get("position", "-")
                pnl = float(p.get("realized_pnl", 0.0) or 0.0)
                is_excl = bool(p.get("exclude_from_ego", False))
                all_settled.append({
                    "cat": cat,
                    "m_id": m_id,
                    "idx": idx,
                    "question": q,
                    "side": side,
                    "pnl": pnl,
                    "excluded": is_excl,
                    "ref": p
                })

    if not all_settled:
        print("[!] No resolved or exited markets available to calibrate.")
        return

    print("\n--- EGO CALIBRATION MARKET SELECTOR ---")
    print("Toggle markets to INCLUDE/EXCLUDE, or custom-adjust recorded Kelly %.")
    print(f"{'#':<3} | {'Status':<10} | {'Kelly':<7} | {'Type':<8} | {'Side':<4} | {'PnL':<10} | {'Question'}")
    print("-" * 90)
    for i, it in enumerate(all_settled, 1):
        st = "[EXCLUDED]" if it["excluded"] else "[ACTIVE]"
        k_val = float(it["ref"].get("kelly", 0.0) or 0.0)
        print(f"{i:<3} | {st:<10} | {k_val*100:>5.1f}% | {it['cat']:<8} | {it['side']:<4} | ${it['pnl']:>+8.2f} | {it['question'][:40]}")

    sel = input("\nEnter row # to toggle [ACTIVE/EXCLUDED], 'adj <#>' to edit Kelly %, 'reset' all, or Enter to finish: ").strip()
    if sel.lower() == "reset":
        for it in all_settled:
            it["ref"]["exclude_from_ego"] = False
        save_history(history)
        print("[+] All markets restored to Active in Base Ego.")
    elif sel.lower().startswith("adj "):
        adj_num = sel.split()[1]
        if adj_num.isdigit():
            idx = int(adj_num) - 1
            if 0 <= idx < len(all_settled):
                target_item = all_settled[idx]
                curr_k = float(target_item["ref"].get("kelly", 0.0) or 0.0)
                new_k_raw = input(f"Enter new Kelly % for '{target_item['question'][:30]}' (currently {curr_k*100:.1f}%, e.g. 1.0 for 1%): ").strip()
                try:
                    new_k_val = float(new_k_raw) / 100.0 if float(new_k_raw) > 1.0 else float(new_k_raw)
                    target_item["ref"]["kelly"] = new_k_val
                    save_history(history)
                    print(f"[+] Market #{adj_num} Kelly updated to {new_k_val*100:.2f}%.")
                except ValueError:
                    print("[!] Invalid percentage entered.")
    elif sel.isdigit():
        idx = int(sel) - 1
        if 0 <= idx < len(all_settled):
            curr = all_settled[idx]["ref"].get("exclude_from_ego", False)
            all_settled[idx]["ref"]["exclude_from_ego"] = not curr
            save_history(history)
            new_st = "EXCLUDED from" if not curr else "INCLUDED in"
            print(f"[+] Market #{sel} is now {new_st} Base Ego calculation.")
        else:
            print("[!] Invalid row number.")


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
    print(f"\nResolving: {ensure_list(preds)[0].get('question')}")
    print(" 1: WON / YES (1.0)   2: LOST / NO (0.0)   3: HALF (0.5)")
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
            if cb == 0.0 and p.get("kelly"):
                cb = float(p.get("kelly", 0.0)) * cfg.BANKROLL
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
    print(f" 1: Min Edge       = {config.MIN_EDGE*100:.2f}%  (skip bets below this edge)")
    print(f" 2: Max Days       = {config.MAX_DAYS}  (ignore markets beyond this horizon)")
    print(f" 3: Min Days       = {config.MIN_DAYS}  (ignore markets resolving sooner than this)")
    print(f" 4: Extreme Odds   = {config.EXTREME_ODDS*100:.2f}%  (tail-risk filter - skip near 0% or 100%)")

    if is_wallet:
        bankroll_add = float(history.get("bankroll_add", 0.0) or history.get("wallet_summary", {}).get("bankroll_add", 0.0) or 0.0)
        print(f" 5: Bankroll Add   = +${bankroll_add:,.2f}  (off-platform funds added to live wallet total, default $0)")
    else:
        print(f" 5: Paper Bankroll = ${config.BANKROLL:,.2f}  (sandbox capital)")

    print("\nEnter setting number to edit [Enter to cancel]:")
    s_choice = input("> ").strip()
    setting_map = {"1": "MIN_EDGE", "2": "MAX_DAYS", "3": "MIN_DAYS", "4": "EXTREME_ODDS"}

    if s_choice in setting_map:
        param = setting_map[s_choice]
        current_val = getattr(config, param)
        new_raw = input(f"New value for {param} (currently {current_val}): ").strip()
        if new_raw:
            try:
                new_val = float(new_raw)
                setattr(config, param, new_val)
                print(f"[+] {param} updated to {new_val}.")
            except ValueError:
                print("[!] Invalid value - setting unchanged.")
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
                    config.BANKROLL = round(cash_val + pos_val + new_add, 4)
                    history.setdefault("wallet_summary", {})["bankroll_add"] = new_add
                    history["wallet_summary"]["bankroll"] = config.BANKROLL
                    save_history(history, profile_id=active_pid)
                    print(f"[+] Off-platform bankroll add set to +${new_add:,.2f}. Total Bankroll is now ${config.BANKROLL:,.2f}.")
                except ValueError:
                    print("[!] Invalid dollar amount.")
        else:
            new_raw = input(f"New paper sandbox bankroll (currently ${config.BANKROLL:,.2f}): ").strip()
            if new_raw:
                try:
                    new_val = max(1.0, float(new_raw))
                    config.BANKROLL = new_val
                    history.setdefault("wallet_summary", {})["bankroll"] = new_val
                    history["wallet_summary"]["cash"] = new_val
                    save_history(history, profile_id=active_pid)
                    print(f"[+] Paper bankroll updated to ${new_val:,.2f}.")
                except ValueError:
                    print("[!] Invalid value.")
