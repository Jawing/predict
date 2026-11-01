"""
predict.py - Antigravity Prediction Engine CLI Orchestrator.
Interactive terminal interface for discovering Polymarket opportunities,
tracking predictions, logging Kelly sizing, monitoring logistic diminishing returns,
and auditing portfolio performance.
"""

import sys
import pandas as pd

from config import (
    BANKROLL,
    POLYGON_WALLET,
)
import config
from utils import ensure_list
from storage import load_history, save_history, stringify_overflow
from views import print_main_menu_dashboard
from wallet import import_polymarket_positions, update_resolutions
from session import run_prediction_session, custom_calculator
from manage import handle_delete_data, manual_resolve, settings_menu


def main():
    while True:
        history = load_history()
        print_main_menu_dashboard(history)

        n_pred = sum(len(ensure_list(v)) for v in history.get("predicted", {}).values())
        n_exit = sum(len(ensure_list(v)) for v in history.get("exited", {}).values())
        n_res = sum(len(ensure_list(v)) for v in history.get("resolved", {}).values())
        n_skip = len(history.get("skipped", {}))
        n_lim = sum(len(ensure_list(v)) for v in history.get("limit_orders", {}).values())

        lim_str = f" | {n_lim} limits" if n_lim > 0 else ""
        print(" 1: Discover Markets (Scan & Predict)")
        print(" 2: Review Portfolio & History (Active, Exited, Resolved, Skipped, Limits)")
        print(" 3: Target Specific URLs or Slugs")
        print(" 4: Wallet Profiles & Resolutions (Sync, Switch, Multi-Wallet)")
        print(" 5: Tools & Settings (Custom Calculator, Data Management, Thresholds & Gamma)")
        print(" 0: Exit")

        choice = input("> ").strip()

        targets = None
        category = "All"
        session_max_offset = 0
        target_exclude_mode = "none"
        discovery_order = "volume"

        if choice == "0":
            print("Exiting program.")
            break

        elif choice == "1":
            category_input = input(
                "\nCategory Tag (e.g., 'Politics', 'Crypto') [Enter for 'All']: "
            ).strip()
            category = category_input if category_input else "All"
            print("\nDiscovery Vector:")
            print(" 1: Volume (Macro Dollars) [Enter]")
            print(" 2: 24h Volume (Trending Repricing)")
            print(" 3: Liquidity (Orderbook TVL)")
            print(" 4: Ending Soonest (Fast Compounding)")
            print(" 5: Far Horizon (Long-Term Compounding)")
            print(" 6: Competitive (50/50 Information Velocity)")
            print(" 7: Niche / Low Volume (Asymmetric Tail Bids)")
            print(" 8: Freshly Listed (Start Date)")
            print(" 9: Multi-Vector Aggregated (All Vectors Unified)")
            vec_choice = input("> ").strip()
            vec_map = {
                "1": "volume",
                "2": "volume24hr",
                "3": "liquidity",
                "4": "endDate",
                "5": "endDate_desc",
                "6": "competitive",
                "7": "volume_asc",
                "8": "startDate",
                "9": "multi",
            }
            discovery_order = vec_map.get(vec_choice, "volume")

            print("Reach: [1] Full Platform Depth (20 pages, 2,000 events) [Enter]  [2] High Velocity (Top 5 pages)  [3] Custom Pages (1-20)")
            reach_choice = input("> ").strip()
            if reach_choice == "2":
                session_max_offset = 400
            elif reach_choice == "3":
                pages_in = input("Enter number of pages (1-20) [default: 20]: ").strip()
                try:
                    p_cnt = max(1, min(20, int(pages_in)))
                    session_max_offset = (p_cnt - 1) * 100
                except ValueError:
                    session_max_offset = 2000
            else:
                session_max_offset = 2000

            op_mode, sub_mode = "discover", "all"

        elif choice == "2":
            print(f"\nReview Portfolio & History:")
            print(f" 1: Active Predictions ({n_pred})")
            print(f" 2: Exited Positions ({n_exit})")
            print(f" 3: Resolved Markets ({n_res})")
            print(f" 4: Skipped Markets ({n_skip})")
            print(f" 5: Limit Orders ({n_lim})")
            print(f" 6: All Predictions (Active + Exited + Resolved)")
            review_mode = input("> ").strip()
            if review_mode not in ["1", "2", "3", "4", "5", "6"]:
                review_mode = "1"
            op_mode = "review"
            sub_mode = {
                "1": "predicted",
                "2": "exited",
                "3": "resolved",
                "4": "skipped",
                "5": "limit_orders",
                "6": "all",
            }[review_mode]

        elif choice == "3":
            urls_input = (
                input("Paste Polymarket URLs/Slugs (comma or space separated):\n> ")
                .replace(",", " ")
                .split()
            )
            targets = []
            for item in urls_input:
                if not item:
                    continue
                if "/event/" in item:
                    targets.append(item.split("/event/")[1].split("/")[0].split("?")[0])
                else:
                    targets.append(item.split("?")[0])

            if not targets:
                print("[!] No valid inputs parsed.\n")
                continue
            print(f"Parsed Targets ({len(targets)}): {', '.join(targets)}")
            print(
                "Exclusion Filter: [1] None (default)  [2] Exclude Active/Exited  [3] Exclude Skipped  [4] Exclude Both"
            )
            ex_choice = input("> ").strip()
            target_exclude_mode = {"2": "predicted", "3": "skipped", "4": "both"}.get(
                ex_choice, "none"
            )
            op_mode, sub_mode = "target", "all"

        elif choice == "4":
            from storage import (
                get_active_profile, set_active_profile, list_profiles,
                delete_profile, set_profile_alias, get_profile_display_name
            )
            history = load_history()
            active_pid = get_active_profile()
            profiles = list_profiles()

            is_wallet = active_pid.startswith("0x")
            prof_type = "Polymarket Wallet" if is_wallet else "Paper Trading"
            active_disp = get_profile_display_name(active_pid)

            print(f"\n--- WALLET PROFILES & RESOLUTIONS ---")
            print(f"Active Profile : {active_disp} ({prof_type})")
            print(f"Address / ID   : {active_pid}")
            print(f"Bankroll       : ${config.BANKROLL:,.2f}")
            print("\nAvailable Profiles:")
            for idx, p in enumerate(profiles, start=1):
                marker = "*" if p["is_active"] else " "
                kind = "Wallet" if p["is_wallet"] else "Paper"
                print(f"  [{idx}] {marker} {p['display_name']:<24} | {p['active_count']} active | {p['resolved_count']} res | ${p['bankroll']:,.2f}")

            print("\nActions:")
            print(" 1: Sync Active Wallet (Positions, Cash, Bankroll & Resolutions) [Enter]")
            print(" 2: Switch Active Profile")
            print(" 3: Add & Switch to New Wallet Profile (0x...)")
            print(" 4: Create Local Paper Trading Profile")
            print(" 5: Set / Rename Profile Alias")
            print(" 6: Manually Resolve a Specific Market")
            print(" 7: Delete a Profile")
            print(" 0: Back to Main Menu")

            s_choice = input("> ").strip()
            if s_choice in ("1", ""):
                if is_wallet:
                    history = import_polymarket_positions(active_pid, history, auto_sync=True)
                history = update_resolutions(history)
                save_history(stringify_overflow(history))
            elif s_choice == "2":
                sw = input("Select Profile Number to Activate: ").strip()
                try:
                    sw_idx = int(sw) - 1
                    if 0 <= sw_idx < len(profiles):
                        chosen_pid = profiles[sw_idx]["profile_id"]
                        set_active_profile(chosen_pid)
                        history = load_history(chosen_pid)
                        print(f"[+] Switched active profile to: {get_profile_display_name(chosen_pid)}")
                    else:
                        print("[!] Invalid profile number.")
                except ValueError:
                    print("[!] Cancelled.")
            elif s_choice == "3":
                new_w = input("Enter Polymarket wallet address (0x...): ").strip()
                if new_w.startswith("0x") and len(new_w) >= 10:
                    new_alias = input("Enter friendly alias for this wallet (e.g. 'Fund B', 'Alpha') [Enter to skip]: ").strip()
                    set_active_profile(new_w)
                    history = load_history(new_w)
                    if new_alias:
                        history["alias"] = new_alias
                    history = import_polymarket_positions(new_w, history, auto_sync=True)
                    history = update_resolutions(history)
                    save_history(stringify_overflow(history))
                    print(f"[+] Created and activated wallet profile: {get_profile_display_name(new_w)}")
                else:
                    print("[!] Invalid Ethereum/Polygon address format.")
            elif s_choice == "4":
                p_name = input("Enter Paper Profile ID [default: 'sandbox']: ").strip()
                if not p_name:
                    p_name = "sandbox"
                p_alias = input(f"Enter friendly alias [default: '{p_name.capitalize()}']: ").strip()
                if not p_alias:
                    p_alias = p_name.capitalize()
                set_active_profile(p_name)
                history = load_history(p_name)
                history["alias"] = p_alias
                save_history(history, profile_id=p_name)
                print(f"[+] Switched to paper trading profile: {p_alias} ({p_name})")
            elif s_choice == "5":
                rn = input("Select Profile Number to Rename: ").strip()
                try:
                    rn_idx = int(rn) - 1
                    if 0 <= rn_idx < len(profiles):
                        target_p = profiles[rn_idx]["profile_id"]
                        curr_alias = profiles[rn_idx].get("alias", "")
                        curr_hint = f" [current: '{curr_alias}']" if curr_alias else ""
                        new_alias = input(f"Enter new alias for '{target_p}'{curr_hint}: ").strip()
                        if new_alias:
                            set_profile_alias(target_p, new_alias)
                            history = load_history()
                            print(f"[+] Alias updated to: '{new_alias}'")
                        else:
                            print("[!] Alias unchanged.")
                    else:
                        print("[!] Invalid profile number.")
                except ValueError:
                    print("[!] Cancelled.")
            elif s_choice == "6":
                manual_resolve(history)
            elif s_choice == "7":
                if len(profiles) <= 1:
                    print("[!] Cannot delete the only remaining profile.")
                else:
                    del_num = input("Select Profile Number to Delete: ").strip()
                    try:
                        del_idx = int(del_num) - 1
                        if 0 <= del_idx < len(profiles):
                            target_p = profiles[del_idx]["profile_id"]
                            conf = input(f"WARNING: Type 'CONFIRM' to delete profile '{target_p}': ").strip()
                            if conf == "CONFIRM":
                                delete_profile(target_p)
                                print(f"[+] Deleted profile '{target_p}'.")
                            else:
                                print("[!] Deletion cancelled.")
                        else:
                            print("[!] Invalid profile number.")
                    except ValueError:
                        print("[!] Cancelled.")
            continue

        elif choice == "5":
            print("\n--- TOOLS & SETTINGS ---")
            print(" 1: Custom Market Calculator (Hypothetical Odds & Kelly)")
            print(" 2: Data Management (Delete Specific Markets / Reset History)")
            print(" 3: Settings & Threshold Filters (Min Edge, Gamma, Bankroll)")
            print(" 0: Back to Main Menu")
            t_choice = input("> ").strip()
            if t_choice == "1":
                custom_calculator()
            elif t_choice == "2":
                handle_delete_data()
            elif t_choice == "3":
                settings_menu()
            continue

        else:
            print("[!] Invalid selection.")
            continue

        portfolio = run_prediction_session(
            mode=op_mode,
            sub_mode=sub_mode,
            target_slugs=targets,
            category=category,
            max_offset=session_max_offset,
            exclude_mode=target_exclude_mode,
            discovery_order=discovery_order,
        )

        if portfolio is not None:
            if not portfolio.empty:
                print("\n--- Final Session Allocations ---")
                pd.set_option("display.max_colwidth", None)
                print(portfolio.to_string(index=False))
            else:
                print("\nNo allocations made.")


if __name__ == "__main__":
    main()
