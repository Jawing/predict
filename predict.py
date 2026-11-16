"""predict.py - Antigravity Prediction Engine CLI Orchestrator.

Interactive terminal interface for discovering Polymarket opportunities,
tracking predictions, logging Kelly sizing, monitoring logistic diminishing returns,
and auditing portfolio performance.
"""

import argparse
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
    create_backup,
    auto_backup,
    list_backups,
    restore_backup,
)
from views import print_main_menu_dashboard, print_review_table
from wallet import import_polymarket_positions, update_resolutions
from session import run_prediction_session, custom_calculator
from manage import handle_delete_data, manual_resolve, settings_menu


def _format_portfolio_table(rows):
    """Formats session allocation dicts into an aligned plain-text table."""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    widths = {
        c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows))
        for c in cols
    }
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    lines = [header]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def main():
    try:
        auto_path = auto_backup()
        if auto_path:
            print(f"[*] Auto-backup saved to '{auto_path}'.")
    except Exception as e:
        print(f"[!] Auto-backup warning: {e}")

    # Startup auto-sync if on an active online wallet profile
    active_pid = get_active_profile()
    history = load_history(active_pid)
    if active_pid.startswith("0x"):
        try:
            print(f"[*] Auto-syncing live wallet '{get_profile_display_name(active_pid)}'...")
            history = import_polymarket_positions(active_pid, history, auto_sync=True)
            history = update_resolutions(history)
            save_history(history, profile_id=active_pid)
        except Exception as e:
            print(f"[!] Startup auto-sync warning: {e}")

    try:
        from views import auto_prune_closed_markets
        auto_prune_closed_markets(history)
    except Exception as e:
        print(f"[!] Startup prune warning: {e}")

    first_loop = True
    while True:
        if first_loop:
            first_loop = False
        else:
            history = load_history()
        print_main_menu_dashboard(history)

        counts = get_history_counts(history)
        n_pred = counts["active"]
        n_exit = counts["exited"]
        n_res = counts["resolved"]
        n_skip = counts["skipped"]

        print(" 1: Discover Markets")
        print(" 2: Review Portfolio & History")
        print(" 3: Target Specific URLs or Slugs")
        print(" 4: Wallet Profiles & Sync")
        print(" 5: Tools & Settings")
        print(" 0: Exit")

        choice = input("> ").strip()

        targets = None
        category = "All"
        session_max_offset = 0
        target_exclude_mode = "none"
        discovery_order = "volume"

        if choice in ("0", ""):
            print("Exiting program.")
            break

        elif choice == "1":
            step = 1
            cancelled = False
            while step <= 3:
                if step == 1:
                    category_input = input(
                        "\nCategory (-exclude, ?: list): "
                    ).strip()
                    if category_input.lower() == "p":
                        cancelled = True
                        break
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
                    category = category_input if category_input else "All"
                    step = 2
                elif step == 2:
                    print("\nDiscovery Vector:")
                    print(" 1: Compounding Epistemic Yield")
                    print(" 2: Epistemic Impact Index")
                    print(" 3: Volume")
                    print(" 4: 24h Volume")
                    print(" 5: Liquidity")
                    print(" 6: Ending Soonest")
                    print(" 7: Far Horizon")
                    print(" 8: Competitive")
                    print(" 9: Niche / Low Volume")
                    print(" 10: Freshly Listed")
                    print(" 11: Multi-Vector Aggregated")
                    print(" 12: Live Epistemic Pulse")
                    vec_choice = input("> ").strip()
                    if vec_choice.lower() == "p":
                        step = 1
                        continue
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
                        "12": "pulse",
                    }
                    discovery_order = vec_map.get(vec_choice, "cey")
                    step = 3
                elif step == 3:
                    print("Reach: [1] Full Platform Depth  [2] High Velocity  [3] Custom Pages")
                    reach_choice = input("> ").strip()
                    if reach_choice.lower() == "p":
                        step = 2
                        continue
                    if reach_choice == "2":
                        session_max_offset = 400
                    elif reach_choice == "3":
                        pages_in = input("Pages (1-20): ").strip()
                        if pages_in.lower() == "p":
                            continue
                        try:
                            p_cnt = max(1, min(20, int(pages_in)))
                            session_max_offset = (p_cnt - 1) * 100
                        except ValueError:
                            session_max_offset = 2000
                    else:
                        session_max_offset = 2000
                    step = 4

            if cancelled:
                continue
            op_mode, sub_mode = "discover", "all"

        elif choice == "2":
            while True:
                print(f"\nReview Portfolio & History:")
                print(f" 1: Active Markets ({n_pred + n_exit})")
                print(f" 2: Resolved Markets ({n_res})")
                print(f" 3: Skipped Markets ({n_skip})")
                print(" 4: All Predictions")
                print(" 5: Calibrate Base Ego")
                review_mode = input("> ").strip()
                if review_mode.lower() == "p":
                    break
                if review_mode == "5":
                    from manage import toggle_ego_market_inclusion
                    toggle_ego_market_inclusion(history)
                    history = load_history()
                    continue
                if review_mode not in ["1", "2", "3", "4"]:
                    review_mode = "1"
                sub_mode = {
                    "1": "predicted",
                    "2": "resolved",
                    "3": "skipped",
                    "4": "all",
                }[review_mode]
                went_prev = print_review_table(history, sub_mode)
                if went_prev == "p":
                    history = load_history()
                    continue
                break
            continue

        elif choice == "3":
            raw_urls = input("URLs / Slugs:\n> ").strip()
            if not raw_urls or raw_urls.lower() == "p":
                continue
            urls_input = raw_urls.replace(",", " ").split()
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
            print("Exclusion Filter: [1] None  [2] Exclude Active/Exited  [3] Exclude Skipped  [4] Exclude Both")
            ex_choice = input("> ").strip()
            if ex_choice.lower() == "p":
                continue
            target_exclude_mode = {"2": "predicted", "3": "skipped", "4": "both"}.get(
                ex_choice, "none"
            )
            op_mode, sub_mode = "target", "all"

        elif choice == "4":
            while True:
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
                print(" 1: Sync Active Wallet")
                print(" 2: Switch Active Profile")
                print(" 3: Add New Profile")
                print(" 4: Rename Profile Alias")
                print(" 5: Delete Profile")

                s_choice = input("> ").strip()
                if not s_choice or s_choice.lower() == "p":
                    break
                if s_choice == "1":
                    if is_wallet:
                        history = import_polymarket_positions(active_pid, history, auto_sync=True)
                    history = update_resolutions(history)
                    save_history(history)
                elif s_choice == "2":
                    sw = input("Select Profile Number to Activate: ").strip()
                    if not sw or sw.lower() == "p":
                        continue
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
                    p_in = input("Enter Polymarket wallet (0x...) or Paper Profile ID: ").strip()
                    if p_in.lower() == "p":
                        continue
                    if not p_in:
                        p_in = "sandbox"
                    is_w = p_in.startswith("0x") and len(p_in) >= 10
                    if is_w:
                        new_alias = input("Enter friendly alias: ").strip()
                        if new_alias.lower() == "p":
                            continue
                        set_active_profile(p_in)
                        history = load_history(p_in)
                        if new_alias:
                            history["alias"] = new_alias
                        history = import_polymarket_positions(p_in, history, auto_sync=True)
                        history = update_resolutions(history)
                        save_history(history)
                        print(f"[+] Created and activated wallet profile: {get_profile_display_name(p_in)}")
                    else:
                        new_alias = input(f"Enter friendly alias: ").strip()
                        if new_alias.lower() == "p":
                            continue
                        if not new_alias:
                            new_alias = p_in.capitalize()
                        br_in = input("Enter paper sandbox bankroll ($): ").strip()
                        if br_in.lower() == "p":
                            continue
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
                    if not rn or rn.lower() == "p":
                        continue
                    try:
                        rn_idx = int(rn) - 1
                        if 0 <= rn_idx < len(profiles):
                            target_p = profiles[rn_idx]["profile_id"]
                            curr_alias = profiles[rn_idx].get("alias", "")
                            curr_hint = f" [current: '{curr_alias}']" if curr_alias else ""
                            new_alias = input(f"Enter new alias for '{target_p}'{curr_hint}: ").strip()
                            if new_alias.lower() == "p":
                                continue
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
                        if not del_num or del_num.lower() == "p":
                            continue
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
            while True:
                print("\n--- TOOLS & SETTINGS ---")
                print(" 1: Custom Market Calculator")
                print(" 2: Platform Market Analyzer")
                print(" 3: Data Management")
                print(" 4: Settings & Threshold Filters")
                t_choice = input("> ").strip()
                if not t_choice or t_choice.lower() == "p":
                    break
                if t_choice == "1":
                    custom_calculator()
                elif t_choice == "2":
                    from analyse import run_interactive_analyzer
                    run_interactive_analyzer()
                elif t_choice == "3":
                    handle_delete_data()
                elif t_choice == "4":
                    settings_menu()
                else:
                    print("[!] Invalid selection.")
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
            if portfolio:
                print("\n--- Final Session Allocations ---")
                print(_format_portfolio_table(portfolio))
            else:
                print("\nNo allocations made.")


def handle_cli_args(argv=None):
    """Handles non-interactive CLI flags (backup/restore); returns True if handled."""
    parser = argparse.ArgumentParser(description="Predict CLI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--backup",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Create a .zip backup of all history and settings (optional output path).",
    )
    group.add_argument(
        "--list-backups",
        action="store_true",
        help="List available .zip backups.",
    )
    group.add_argument(
        "--restore",
        metavar="PATH",
        help="Restore history and settings from a .zip backup.",
    )
    args = parser.parse_args(argv)

    if args.backup is not None:
        b_path = create_backup(output_path=args.backup or None)
        size_kb = os.path.getsize(b_path) / 1024.0
        print(f"[+] Backup saved to '{b_path}' ({size_kb:.1f} KB).")
        return True

    if args.list_backups:
        backups = list_backups()
        if not backups:
            print("[*] No .zip backups found.")
        else:
            for i, b in enumerate(backups, 1):
                size_kb = b["size_bytes"] / 1024.0
                p_cnt = len(b.get("profiles") or [])
                print(f"{i}: {b['path']} ({size_kb:.1f} KB, {p_cnt} profile(s), {b.get('created_at') or '-'})")
        return True

    if args.restore:
        res = restore_backup(args.restore, create_safety_backup=True)
        if res.get("safety_backup"):
            print(f"[*] Safety backup saved to '{res['safety_backup']}'.")
        print(f"[+] Restored {len(res['restored_profiles'])} profile(s). Active profile: {res['active_profile']}")
        return True

    return False


if __name__ == "__main__":
    if not handle_cli_args():
        main()
