"""Persistent JSON storage operations and per-wallet profile isolation."""
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
import config as cfg
from utils import ensure_list, stringify_overflow, get_history_counts


BACKUP_DIR_NAME = "backups"
MAX_BACKUPS_PER_PROFILE = 20


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

    # Synchronize runtime bankroll immediately to match active profile
    fpath = get_profile_path(profile_id)
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            saved_br = pdata.get("wallet_summary", {}).get("bankroll")
            if saved_br and float(saved_br) > 0:
                cfg.BANKROLL = float(saved_br)
        except Exception:
            pass
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
                data["profile_id"] = profile_id
                data.setdefault("predicted", {})
                data.setdefault("exited", {})
                data.setdefault("skipped", {})
                data.setdefault("resolved", {})
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

                # Restore profile settings if saved (env vars take precedence)
                saved_st = data.get("settings")
                if isinstance(saved_st, dict):
                    for k in ["MIN_EDGE", "MAX_DAYS", "MIN_DAYS", "EXTREME_ODDS"]:
                        if k in saved_st:
                            setattr(cfg, k, saved_st[k])
                st_dict = saved_st if isinstance(saved_st, dict) else {}
                if not os.getenv("CLOB_SIGNATURE_TYPE", "").strip():
                    saved_sig = st_dict.get("CLOB_SIGNATURE_TYPE")
                    cfg.CLOB_SIGNATURE_TYPE = int(saved_sig) if saved_sig in (0, 1, 2) else None
                if not os.getenv("EXECUTION_MODE", "").strip():
                    saved_mode = str(st_dict.get("EXECUTION_MODE") or "browser").strip().lower()
                    cfg.EXECUTION_MODE = saved_mode if saved_mode in ("browser", "metamask", "clob") else "browser"
                if not os.getenv("METAMASK_SIGNER", "").strip():
                    cfg.METAMASK_SIGNER = str(st_dict.get("METAMASK_SIGNER") or "").strip()
                if not os.getenv("SIGNING_BROWSER", "").strip():
                    cfg.SIGNING_BROWSER = str(st_dict.get("SIGNING_BROWSER") or "").strip()

                return data
        except json.JSONDecodeError as e:
            import shutil, time as _time
            backup_path = f"{target_path}.corrupt.{int(_time.time())}"
            try:
                shutil.copy2(target_path, backup_path)
                print(f"[!] Corrupt profile JSON in {target_path} ({e}). Backed up to {backup_path}.")
            except OSError as copy_err:
                print(f"[!] Corrupt profile JSON in {target_path} ({e}); backup failed: {copy_err}.")

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
        "wallet_summary": {"cash": 100.0, "portfolio": 0.0, "bankroll": 100.0}
    }
    cfg.BANKROLL = 100.0
    if is_wallet:
        cfg.POLYGON_WALLET = profile_id
    save_history(data, profile_id=profile_id)
    return data


def _backup_profile_file(target_path, profile_id, max_backups=MAX_BACKUPS_PER_PROFILE):
    """Saves a deduplicated timestamped backup in <HISTORY_DIR>/backups before overwriting."""
    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        return None
    backup_dir = os.path.join(cfg.HISTORY_DIR, BACKUP_DIR_NAME)
    os.makedirs(backup_dir, exist_ok=True)
    prefix = f"history_{profile_id}."
    existing = sorted(
        f for f in os.listdir(backup_dir)
        if f.startswith(prefix) and f.endswith(".json")
    )
    with open(target_path, "rb") as rf:
        current_bytes = rf.read()
    if existing:
        try:
            with open(os.path.join(backup_dir, existing[-1]), "rb") as lf:
                if lf.read() == current_bytes:
                    return os.path.join(backup_dir, existing[-1])
        except OSError:
            pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = os.path.join(backup_dir, f"{prefix}{stamp}.json")
    shutil.copy2(target_path, backup_path)
    existing.append(os.path.basename(backup_path))
    if max_backups and len(existing) > max_backups:
        for old_name in existing[:-max_backups]:
            try:
                os.remove(os.path.join(backup_dir, old_name))
            except OSError:
                pass
    return backup_path


def save_history(history, profile_id=None, max_backups=MAX_BACKUPS_PER_PROFILE):
    """Atomically saves history JSON with a deduplicated rolling backup."""
    if not isinstance(history, dict):
        raise TypeError("history must be a dict")
    if not profile_id:
        profile_id = get_active_profile()

    os.makedirs(cfg.HISTORY_DIR, exist_ok=True)
    target_path = get_profile_path(profile_id)
    clean_data = stringify_overflow(history)

    if os.path.exists(target_path):
        _backup_profile_file(target_path, sanitize_profile_id(profile_id), max_backups=max_backups)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cfg.HISTORY_DIR,
            prefix=f".history_{sanitize_profile_id(profile_id)}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(clean_data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


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

            counts = get_history_counts(data)
            br = float(data.get("wallet_summary", {}).get("bankroll", 100.0) or 100.0)
            alias = get_profile_alias(pid)
            is_wallet = pid.startswith("0x")

            profiles.append({
                "profile_id": pid,
                "alias": alias,
                "display_name": get_profile_display_name(pid),
                "is_active": (pid == active_pid),
                "is_wallet": is_wallet,
                "active_count": counts["active"],
                "exited_count": counts["exited"],
                "resolved_count": counts["resolved"],
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
