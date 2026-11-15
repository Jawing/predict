"""Persistent JSON storage operations and per-wallet profile isolation."""
import json
import os
import re
import time
import zipfile
from datetime import datetime, timezone
import config as cfg
from utils import ensure_list, stringify_overflow, get_history_counts


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


def default_profile_alias(profile_id):
    """Returns default friendly alias for known built-in profile IDs."""
    pid = str(profile_id or "")
    if pid.startswith("0xbC73"):
        return "Main"
    if pid == "local":
        return "Paper"
    return ""


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
    return default_profile_alias(profile_id)


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
                data["alias"] = default_profile_alias(w_addr)
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
                    data["alias"] = default_profile_alias(profile_id)

                if profile_id.startswith("0x"):
                    data["wallet_address"] = profile_id
                    cfg.POLYGON_WALLET = profile_id

                # Auto-migrate any exited entries out of 'predicted' into 'exited'
                for m_id in list(data["predicted"].keys()):
                    preds = ensure_list(data["predicted"][m_id])
                    active_p = [p for p in preds if not p.get("exited")]
                    exited_p = [p for p in preds if p.get("exited")]
                    if exited_p:
                        data["exited"][m_id] = ensure_list(data["exited"].get(m_id, [])) + exited_p
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

                # Restore profile settings if saved
                saved_st = data.get("settings")
                if isinstance(saved_st, dict):
                    for k in ["MIN_EDGE", "MAX_DAYS", "MIN_DAYS", "EXTREME_ODDS"]:
                        if k in saved_st:
                            setattr(cfg, k, saved_st[k])

                return data
        except json.JSONDecodeError as e:
            import shutil
            backup_path = f"{target_path}.corrupt.{int(time.time())}"
            try:
                shutil.copy2(target_path, backup_path)
                print(f"[!] Corrupt profile JSON in {target_path} ({e}). Backed up to {backup_path}.")
            except OSError as copy_err:
                print(f"[!] Corrupt profile JSON in {target_path} ({e}); backup failed: {copy_err}.")

    # Profile does not exist yet: create skeleton
    is_wallet = profile_id.startswith("0x")
    alias_val = default_profile_alias(profile_id)
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


def save_history(history, profile_id=None):
    """Saves prediction history to profile file, automatically sanitizing overflow floats."""
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


def _default_backup_dir():
    return os.path.join(cfg.HISTORY_DIR, "backups")


def _is_safe_history_member(name):
    """Validates zip member names to prevent path traversal and reject unexpected entries."""
    if not name or "\\" in name or name.startswith("/") or ":" in name:
        return False
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    if name == "active_profile.txt":
        return True
    if len(parts) == 2 and parts[0] == "data" and parts[1].startswith("history_") and parts[1].endswith(".json"):
        pid = parts[1][len("history_"):-len(".json")]
        return bool(pid) and sanitize_profile_id(pid) == pid
    return False


def _prune_old_backups(target_dir, keep):
    """Deletes oldest predict_backup_*.zip archives when count exceeds keep."""
    if not keep or keep <= 0 or not os.path.isdir(target_dir):
        return
    zips = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.startswith("predict_backup_") and f.lower().endswith(".zip")
    ]
    zips = [p for p in zips if os.path.isfile(p)]
    if len(zips) <= keep:
        return
    zips.sort(key=lambda p: (os.path.getmtime(p), p))
    for old_path in zips[:-keep]:
        try:
            os.remove(old_path)
        except OSError:
            pass


def create_backup(backup_dir=None, output_path=None, keep=10):
    """Creates a timestamped .zip backup of all profile histories and per-profile settings."""
    migrate_legacy_history_if_needed()

    if output_path:
        dest_path = output_path
        dest_parent = os.path.dirname(dest_path)
        if dest_parent:
            os.makedirs(dest_parent, exist_ok=True)
        target_dir = None
    else:
        target_dir = backup_dir or _default_backup_dir()
        os.makedirs(target_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = os.path.join(target_dir, f"predict_backup_{stamp}.zip")
        suffix = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(target_dir, f"predict_backup_{stamp}_{suffix}.zip")
            suffix += 1

    history_files = []
    if os.path.exists(cfg.HISTORY_DIR):
        for fname in sorted(os.listdir(cfg.HISTORY_DIR)):
            if fname.startswith("history_") and fname.endswith(".json"):
                fpath = os.path.join(cfg.HISTORY_DIR, fname)
                if os.path.isfile(fpath):
                    history_files.append((fname, fpath))

    tmp_path = f"{dest_path}.tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, fpath in history_files:
                zf.write(fpath, arcname=f"data/{fname}")
            if cfg.ACTIVE_PROFILE_FILE and os.path.isfile(cfg.ACTIVE_PROFILE_FILE):
                zf.write(cfg.ACTIVE_PROFILE_FILE, arcname="active_profile.txt")
        os.replace(tmp_path, dest_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if target_dir and keep:
        _prune_old_backups(target_dir, keep)

    return dest_path


def auto_backup(max_age_hours=24, keep=10, backup_dir=None):
    """Creates an automatic backup if existing history has no backup newer than max_age_hours."""
    has_history = os.path.isdir(cfg.HISTORY_DIR) and any(
        f.startswith("history_") and f.endswith(".json") for f in os.listdir(cfg.HISTORY_DIR)
    )
    if not has_history:
        return None

    target_dir = backup_dir or _default_backup_dir()
    if os.path.isdir(target_dir):
        existing = [
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.startswith("predict_backup_") and f.lower().endswith(".zip")
            and os.path.isfile(os.path.join(target_dir, f))
        ]
        if existing:
            newest_mtime = max(os.path.getmtime(p) for p in existing)
            if (time.time() - newest_mtime) < max_age_hours * 3600:
                return None

    return create_backup(backup_dir=target_dir, keep=keep)


def list_backups(backup_dir=None):
    """Lists available .zip backup archives sorted newest-first."""
    target_dir = backup_dir or _default_backup_dir()
    if not os.path.isdir(target_dir):
        return []

    entries = []
    for fname in os.listdir(target_dir):
        if not fname.lower().endswith(".zip"):
            continue
        fpath = os.path.join(target_dir, fname)
        if not os.path.isfile(fpath):
            continue

        profiles = []
        try:
            with zipfile.ZipFile(fpath, "r") as zf:
                for name in zf.namelist():
                    if name.startswith("data/history_") and name.endswith(".json"):
                        profiles.append(name[len("data/history_"):-len(".json")])
        except Exception:
            profiles = []

        mtime = os.path.getmtime(fpath)
        entries.append((
            (mtime, fname),
            {
                "filename": fname,
                "path": fpath,
                "size_bytes": os.path.getsize(fpath),
                "created_at": datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "profiles": sorted(profiles),
            },
        ))

    entries.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in entries]


def restore_backup(backup_path, create_safety_backup=True):
    """Restores profile histories and per-profile settings from a .zip backup archive."""
    if not backup_path or not os.path.isfile(backup_path):
        raise FileNotFoundError(f"Backup file not found: {backup_path}")
    if not zipfile.is_zipfile(backup_path):
        raise ValueError(f"Not a valid zip archive: {backup_path}")

    staged_history = {}
    staged_active = None

    with zipfile.ZipFile(backup_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not _is_safe_history_member(name):
                raise ValueError(f"Unsupported or unsafe backup entry: {name}")

            raw = zf.read(name).decode("utf-8")
            if name == "active_profile.txt":
                staged_active = raw.strip()
            elif name.startswith("data/"):
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Invalid history JSON in backup archive: {name}")
                staged_history[os.path.basename(name)] = raw

    if not staged_history and staged_active is None:
        raise ValueError("Backup archive contains no restorable history or profile data.")

    safety_backup_path = None
    has_existing_data = (
        (os.path.isdir(cfg.HISTORY_DIR) and any(
            f.startswith("history_") and f.endswith(".json") for f in os.listdir(cfg.HISTORY_DIR)
        ))
        or (cfg.ACTIVE_PROFILE_FILE and os.path.isfile(cfg.ACTIVE_PROFILE_FILE))
    )
    if create_safety_backup and has_existing_data:
        safety_backup_path = create_backup()

    if staged_history:
        os.makedirs(cfg.HISTORY_DIR, exist_ok=True)
        for fname, raw in staged_history.items():
            with open(os.path.join(cfg.HISTORY_DIR, fname), "w", encoding="utf-8") as f:
                f.write(raw)

    if staged_active and cfg.ACTIVE_PROFILE_FILE:
        set_active_profile(staged_active)
        load_history(staged_active)
    elif cfg.ACTIVE_PROFILE_FILE and os.path.isfile(cfg.ACTIVE_PROFILE_FILE):
        load_history()

    return {
        "restored_profiles": sorted(
            f[len("history_"):-len(".json")] for f in staged_history.keys()
        ),
        "active_profile": staged_active or get_active_profile(),
        "safety_backup": safety_backup_path,
    }
