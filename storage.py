"""Persistent JSON storage operations and per-wallet profile isolation."""
import json
import os
import re
import config as cfg
from utils import ensure_list, stringify_overflow


def sanitize_profile_id(name):
    """Sanitizes profile identifier for safe file storage."""
    if not name:
        return "local"
    clean = re.sub(r'[^\w\.-]', '_', str(name).strip())
    return clean if clean else "local"


def get_active_profile():
    """Returns the identifier of the currently active wallet or paper profile."""
    if os.path.exists(cfg.ACTIVE_PROFILE_FILE):
        try:
            with open(cfg.ACTIVE_PROFILE_FILE, "r", encoding="utf-8") as f:
                p = f.read().strip()
                if p:
                    return p
        except Exception:
            pass
    if cfg.POLYGON_WALLET:
        return cfg.POLYGON_WALLET
    if os.path.exists(cfg.HISTORY_DIR):
        for fname in os.listdir(cfg.HISTORY_DIR):
            if fname.startswith("history_") and fname.endswith(".json"):
                pid = fname[len("history_"):-len(".json")]
                if pid:
                    return pid
    return "local"


def set_active_profile(profile_id):
    """Sets the active wallet/profile and updates runtime globals."""
    profile_id = sanitize_profile_id(profile_id)
    try:
        with open(cfg.ACTIVE_PROFILE_FILE, "w", encoding="utf-8") as f:
            f.write(profile_id)
    except Exception:
        pass
    if profile_id.startswith("0x"):
        cfg.POLYGON_WALLET = profile_id
    else:
        cfg.POLYGON_WALLET = ""
    return profile_id


def get_profile_path(profile_id=None):
    """Returns absolute file path for a profile."""
    if not profile_id:
        profile_id = get_active_profile()
    profile_id = sanitize_profile_id(profile_id)
    return os.path.join(cfg.HISTORY_DIR, f"history_{profile_id}.json")


def get_profile_alias(profile_id=None):
    """Returns friendly alias for a profile (e.g. 'Main', 'Paper')."""
    if not profile_id:
        profile_id = get_active_profile()
    profile_id = sanitize_profile_id(profile_id)
    fpath = get_profile_path(profile_id)
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                alias = data.get("alias", "")
                if alias:
                    return alias
        except Exception:
            pass
    if profile_id.startswith("0xbC73"):
        return "Main"
    elif profile_id == "local":
        return "Paper"
    return ""


def set_profile_alias(profile_id, alias):
    """Updates the friendly alias for a profile."""
    if not profile_id:
        profile_id = get_active_profile()
    profile_id = sanitize_profile_id(profile_id)
    history = load_history(profile_id)
    history["alias"] = alias.strip()
    save_history(history, profile_id=profile_id)
    return history["alias"]


def get_profile_display_name(profile_id=None):
    """Returns a formatted profile name, incorporating alias and short address."""
    if not profile_id:
        profile_id = get_active_profile()
    profile_id = sanitize_profile_id(profile_id)
    alias = get_profile_alias(profile_id)
    is_wallet = profile_id.startswith("0x")
    short_w = f"0x..{profile_id[-4:]}" if (is_wallet and len(profile_id) >= 6) else profile_id

    if alias and is_wallet:
        return f"{alias} ({short_w})"
    elif alias:
        return f"{alias} ({profile_id})"
    elif is_wallet:
        return short_w
    return profile_id.upper()


def migrate_legacy_history_if_needed():
    """Migrates legacy single prediction_history.json into data/history_<wallet>.json."""
    os.makedirs(cfg.HISTORY_DIR, exist_ok=True)
    existing_files = [f for f in os.listdir(cfg.HISTORY_DIR) if f.startswith("history_") and f.endswith(".json")]
    if existing_files:
        return

    if os.path.exists(cfg.HISTORY_FILE):
        try:
            with open(cfg.HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            w_addr = data.get("wallet_address") or cfg.POLYGON_WALLET or "local"
            w_addr = sanitize_profile_id(w_addr)
            target_file = os.path.join(cfg.HISTORY_DIR, f"history_{w_addr}.json")
            if "alias" not in data:
                data["alias"] = "Main" if w_addr.startswith("0xbC73") else ("Paper" if w_addr == "local" else "")
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(stringify_overflow(data), f, indent=4)
            set_active_profile(w_addr)
            print(f"[*] Migrated legacy prediction history to profile '{w_addr}'.")
        except Exception as e:
            print(f"[!] Migration error: {e}")


def load_history(profile_id=None):
    """Loads prediction history for specified profile (or active profile) and sets runtime bankroll."""
    migrate_legacy_history_if_needed()
    if not profile_id:
        profile_id = get_active_profile()

    target_path = get_profile_path(profile_id)
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                data.setdefault("predicted", {})
                data.setdefault("exited", {})
                data.setdefault("skipped", {})
                data.setdefault("resolved", {})
                data.setdefault("limit_orders", {})
                data.setdefault("wallet_summary", {"cash": 100.0, "portfolio": 0.0, "bankroll": 100.0})

                if "alias" not in data or not data["alias"]:
                    if profile_id.startswith("0xbC73"):
                        data["alias"] = "Main"
                    elif profile_id == "local":
                        data["alias"] = "Paper"
                    else:
                        data["alias"] = ""

                if profile_id.startswith("0x"):
                    data["wallet_address"] = profile_id
                    cfg.POLYGON_WALLET = profile_id

                # Auto-migrate any exited entries out of 'predicted' into 'exited'
                for m_id in list(data["predicted"].keys()):
                    preds = ensure_list(data["predicted"][m_id])
                    active_p = [p for p in preds if not p.get("exited")]
                    exited_p = [p for p in preds if p.get("exited")]
                    if exited_p:
                        data["exited"].setdefault(m_id, []).extend(exited_p)
                        if active_p:
                            data["predicted"][m_id] = active_p
                        else:
                            data["predicted"].pop(m_id, None)

                # Set runtime bankroll
                ws = data.get("wallet_summary")
                if isinstance(ws, dict):
                    saved_br = ws.get("bankroll")
                    if saved_br and float(saved_br) > 0:
                        cfg.BANKROLL = float(saved_br)

                return data
        except json.JSONDecodeError:
            pass

    # Profile does not exist yet: create skeleton
    is_wallet = profile_id.startswith("0x")
    alias_val = "Main" if profile_id.startswith("0xbC73") else ("Paper" if profile_id == "local" else "")
    data = {
        "wallet_address": profile_id if is_wallet else None,
        "alias": alias_val,
        "predicted": {},
        "exited": {},
        "skipped": {},
        "resolved": {},
        "limit_orders": {},
        "wallet_summary": {"cash": 100.0, "portfolio": 0.0, "bankroll": 100.0}
    }
    cfg.BANKROLL = 100.0
    if is_wallet:
        cfg.POLYGON_WALLET = profile_id
    save_history(data, profile_id=profile_id)
    return data


def save_history(history, profile_id=None):
    """Saves prediction history to profile file."""
    if not profile_id:
        profile_id = get_active_profile()

    os.makedirs(cfg.HISTORY_DIR, exist_ok=True)
    target_path = get_profile_path(profile_id)
    clean_data = stringify_overflow(history)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=4)


def list_profiles():
    """Returns a list of all existing profiles with metadata and aliases."""
    migrate_legacy_history_if_needed()
    active_pid = get_active_profile()
    profiles = []
    if not os.path.exists(cfg.HISTORY_DIR):
        return profiles

    for fname in sorted(os.listdir(cfg.HISTORY_DIR)):
        if fname.startswith("history_") and fname.endswith(".json"):
            pid = fname[len("history_"):-len(".json")]
            fpath = os.path.join(cfg.HISTORY_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

            n_pred = sum(len(ensure_list(v)) for v in data.get("predicted", {}).values())
            n_exit = sum(len(ensure_list(v)) for v in data.get("exited", {}).values())
            n_res = sum(len(ensure_list(v)) for v in data.get("resolved", {}).values())
            br = float(data.get("wallet_summary", {}).get("bankroll", 100.0) or 100.0)

            alias = data.get("alias", "")
            if not alias:
                if pid.startswith("0xbC73"):
                    alias = "Main"
                elif pid == "local":
                    alias = "Paper"

            is_wallet = pid.startswith("0x")
            short_w = f"0x..{pid[-4:]}" if (is_wallet and len(pid) >= 6) else pid
            if alias and is_wallet:
                display_name = f"{alias} ({short_w})"
            elif alias:
                display_name = f"{alias} ({pid})"
            elif is_wallet:
                display_name = short_w
            else:
                display_name = pid.upper()

            profiles.append({
                "profile_id": pid,
                "alias": alias,
                "display_name": display_name,
                "is_active": (pid == active_pid),
                "is_wallet": is_wallet,
                "active_count": n_pred,
                "exited_count": n_exit,
                "resolved_count": n_res,
                "bankroll": br,
                "path": fpath
            })
    return profiles


def delete_profile(profile_id):
    """Deletes a profile file. If it was active, switches to another existing profile."""
    profile_id = sanitize_profile_id(profile_id)
    fpath = get_profile_path(profile_id)
    if os.path.exists(fpath):
        os.remove(fpath)

    active_pid = get_active_profile()
    if active_pid == profile_id:
        profiles = list_profiles()
        if profiles:
            set_active_profile(profiles[0]["profile_id"])
        else:
            set_active_profile("local")
