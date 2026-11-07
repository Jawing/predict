"""predict.py - Antigravity Prediction Engine CLI Orchestrator.

Interactive terminal interface for discovering Polymarket opportunities,
tracking predictions, logging Kelly sizing, monitoring logistic diminishing returns,
and auditing portfolio performance.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import pandas as pd

import config as cfg
from utils import get_history_counts
from storage import (
    load_history,
    save_history,
    get_active_profile,
    set_active_profile,
    list_profiles,
    delete_profile,
    set_profile_alias,
    get_profile_display_name,
)
from views import print_main_menu_dashboard, print_review_table
from wallet import import_polymarket_positions, update_resolutions
from session import run_prediction_session, custom_calculator, compare_asset_market
from manage import handle_delete_data, manual_resolve, settings_menu


def main():
    # Startup auto-sync if on an active online wallet profile
    active_pid = get_active_profile()
    if active_pid.startswith("0x"):
        try:
            print(f"[*] Auto-syncing live wallet '{get_profile_display_name(active_pid)}'...")
            history = load_history(active_pid)
            history = import_polymarket_positions(active_pid, history, auto_sync=True)
            history = update_resolutions(history)
            save_history(history, profile_id=active_pid)
        except Exception as e:
            print(f"[!] Startup auto-sync warning: {e}")

    try:
        from views import auto_prune_closed_markets
        h_init = load_history()
        auto_prune_closed_markets(h_init)
    except Exception:
        pass

    while True:
        history = load_history()
        print_main_menu_dashboard(history)

        counts = get_history_counts(history)
        n_pred = counts["active"]
        n_exit = counts["exited"]
        n_res = counts["resolved"]
        n_skip = counts["skipped"]

        print(" 1: Discover Markets (Scan & Predict)")
        print(" 2: Review Portfolio & History (Active, Exited, Resolved, Skipped)")
        print(" 3: Target Specific URLs or Slugs")
        print(" 4: Wallet Profiles & Sync (Switch, Add, Sync)")
        print(" 5: Tools & Settings (Calculators, Data Management, Parameters)")
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
            while True:
                category_input = input(
                    "\nCategory Tag(s) (e.g. 'Politics, Geopolitics', '-Sports', or '?' for list) [Enter for 'All']: "
                ).strip()
                if category_input in ("?", "list", "help"):
                    print("\n[+] Top Polymarket Categories & Tags:")
                    print("    - Politics (Elections, World Elections, US Politics, Geopolitics)")
                    print("    - Crypto (Bitcoin, Ethereum, Solana, DeFi, Airdrops)")
                    print("    - Sports (Soccer, Football, Basketball, Tennis, Esports)")
                    print("    - Business & Economy (Macro, Fed Rates, CPI, Stocks, Tech)")
                    print("    - Science & AI (AI Models, Tech, Space, Biotech)")
                    print("    - Pop Culture (Movies, Awards, Music, Social Media)")
                    print("    Syntax: Include multiple with comma (e.g. 'Politics, Crypto') or exclude with minus (e.g. '-Sports').")
                    continue
                break
            category = category_input if category_input else "All"
            print("\nDiscovery Vector:")
            print(" 1: Compounding Epistemic Yield (CEY: Event-Diverse EII / sqrt(Time)) [Enter]")
            print(" 2: Epistemic Impact Index (EII: Event-Diverse Pure Macro Stakes)")
            print(" 3: Volume (Macro Dollars)")
            print(" 4: 24h Volume (Trending Repricing)")
            print(" 5: Liquidity (Orderbook TVL)")
            print(" 6: Ending Soonest (Fast Compounding)")
            print(" 7: Far Horizon (Long-Term Compounding)")
            print(" 8: Competitive (50/50 Information Velocity)")
            print(" 9: Niche / Low Volume (Asymmetric Tail Bids)")
            print(" 10: Freshly Listed (Start Date)")
            print(" 11: Multi-Vector Aggregated (All Vectors Unified)")
            vec_choice = input("> ").strip()
            vec_map = {
                "1": "cey",
                "2": "eii",
                "3": "volume",
                "4": "volume24hr",
                "5": "liquidity",
                "6": "endDate",
                "7": "endDate_desc",
                "8": "competitive",
                "9": "volume_asc",
                "10": "startDate",
                "11": "multi",
            }
            discovery_order = vec_map.get(vec_choice, "cey")

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
            print(f" 1: Active Predictions ({n_pred}) [Enter]")
            print(f" 2: Exited Positions ({n_exit})")
            print(f" 3: Resolved Markets ({n_res})")
            print(f" 4: Skipped Markets ({n_skip})")
            print(f" 5: All Predictions (Active + Exited + Resolved)")
            print(f" 6: Calibrate Base Ego (Include/Exclude & Custom Kelly)")
            review_mode = input("> ").strip()
            if review_mode == "6":
                from manage import toggle_ego_market_inclusion
                toggle_ego_market_inclusion(history)
                continue
            if review_mode not in ["1", "2", "3", "4", "5"]:
                review_mode = "1"
            sub_mode = {
                "1": "predicted",
                "2": "exited",
                "3": "resolved",
                "4": "skipped",
                "5": "all",
            }[review_mode]
            print_review_table(history, sub_mode)
            continue

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
            history = load_history()
            active_pid = get_active_profile()
            profiles = list_profiles()

            is_wallet = active_pid.startswith("0x")
            prof_type = "Polymarket Wallet" if is_wallet else "Paper Trading"
            active_disp = get_profile_display_name(active_pid)

            print(f"\n--- WALLET & PROFILES ---")
            print(f"Active Profile : {active_disp} ({prof_type})")
            print(f"Address / ID   : {active_pid}")
            print(f"Bankroll       : ${cfg.BANKROLL:,.2f}")
            print("\nAvailable Profiles:")
            for idx, p in enumerate(profiles, start=1):
                marker = "*" if p["is_active"] else " "
                print(f"  [{idx}] {marker} {p['display_name']:<24} | {p['active_count']} active | {p['resolved_count']} res | ${p['bankroll']:,.2f}")

            print("\nActions:")
            print(" 1: Sync Active Wallet (Positions, Cash, Bankroll & Resolutions) [Enter]")
            print(" 2: Switch Active Profile")
            print(" 3: Add New Profile (Wallet 0x... or Paper Sandbox)")
            print(" 4: Rename Profile Alias")
            print(" 5: Delete a Profile")
            print(" 0: Back to Main Menu")

            s_choice = input("> ").strip()
            if s_choice in ("1", ""):
                if is_wallet:
                    history = import_polymarket_positions(active_pid, history, auto_sync=True)
                history = update_resolutions(history)
                save_history(history)
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
                p_in = input("Enter Polymarket wallet (0x...) or Paper Profile ID [default: 'sandbox']: ").strip()
                if not p_in:
                    p_in = "sandbox"
                is_w = p_in.startswith("0x") and len(p_in) >= 10
                if is_w:
                    new_alias = input("Enter friendly alias (e.g. 'Main', 'Fund B') [Enter to skip]: ").strip()
                    set_active_profile(p_in)
                    history = load_history(p_in)
                    if new_alias:
                        history["alias"] = new_alias
                    history = import_polymarket_positions(p_in, history, auto_sync=True)
                    history = update_resolutions(history)
                    save_history(history)
                    print(f"[+] Created and activated wallet profile: {get_profile_display_name(p_in)}")
                else:
                    new_alias = input(f"Enter friendly alias [default: '{p_in.capitalize()}']: ").strip()
                    if not new_alias:
                        new_alias = p_in.capitalize()
                    br_in = input("Enter paper sandbox bankroll ($) [default: $100.00]: ").strip()
                    try:
                        paper_br = max(1.0, float(br_in)) if br_in else 100.00
                    except ValueError:
                        paper_br = 100.00
                    set_active_profile(p_in)
                    history = load_history(p_in)
                    history["alias"] = new_alias
                    history.setdefault("wallet_summary", {})["bankroll"] = paper_br
                    history["wallet_summary"]["cash"] = paper_br
                    cfg.BANKROLL = paper_br
                    save_history(history, profile_id=p_in)
                    print(f"[+] Switched to paper trading profile: {new_alias} ({p_in}) with ${paper_br:,.2f} bankroll.")
            elif s_choice == "4":
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
            elif s_choice == "5":
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
                                history = load_history()
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
            print(" 2: Asset vs. Binary Leverage & Efficiency Calculator")
            print(" 3: Data Management (Delete Data, Reset, Manual Resolve)")
            print(" 4: Settings & Threshold Filters (Min Edge, Gamma, Bankroll)")
            print(" 5: Platform Market Analyzer & Global Category Overview")
            print(" 0: Back to Main Menu")
            t_choice = input("> ").strip()
            if t_choice == "1":
                custom_calculator()
            elif t_choice == "2":
                compare_asset_market()
            elif t_choice == "3":
                handle_delete_data()
            elif t_choice == "4":
                settings_menu()
            elif t_choice == "5":
                from analyse import run_interactive_analyzer
                run_interactive_analyzer()
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
