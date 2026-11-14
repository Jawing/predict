"""Configuration parameters and API constants for the prediction pipeline."""
import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

HISTORY_DIR = "data"
HISTORY_FILE = "prediction_history.json"
ACTIVE_PROFILE_FILE = "active_profile.txt"
BANKROLL = 100.00
GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_WALLET = "0xbC730b41A9aaAd603dDA9473DcB8a059dF2c9fa7"  # Default / primary Polymarket wallet address

# --- OPTIONAL CLOB API CREDENTIALS (FOR AUTOMATED LIVE EXECUTION) ---
# Populated via .env / environment variables or directly here for py-clob-client.
# Signature types: None = auto-detect, 0 = EOA, 1 = POLY_PROXY (Magic/Email), 2 = POLY_GNOSIS_SAFE (Browser)
CLOB_HOST = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CLOB_CHAIN_ID = int(os.getenv("CLOB_CHAIN_ID", "137"))
CLOB_PRIVATE_KEY = os.getenv("CLOB_PRIVATE_KEY") or os.getenv("POLYMARKET_PRIVATE_KEY", "")
CLOB_API_KEY = os.getenv("CLOB_API_KEY", "")
CLOB_API_SECRET = os.getenv("CLOB_API_SECRET", "")
CLOB_API_PASSPHRASE = os.getenv("CLOB_API_PASSPHRASE", "")
_sig_raw = os.getenv("CLOB_SIGNATURE_TYPE", "").strip()
CLOB_SIGNATURE_TYPE = int(_sig_raw) if _sig_raw in ("0", "1", "2") else None
CLOB_FUNDER = os.getenv("CLOB_FUNDER", "")
_max_ord_raw = os.getenv("CLOB_MAX_ORDER_USD", "").strip()
CLOB_MAX_ORDER_USD = float(_max_ord_raw) if _max_ord_raw else 250.0  # Safety cap per live order ($)
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "browser").strip().lower() or "browser"  # browser | metamask | clob
METAMASK_SIGNER = os.getenv("METAMASK_SIGNER", "").strip()
SIGNING_BROWSER = os.getenv("SIGNING_BROWSER", "").strip()

# --- FILTERING PARAMETERS ---
MIN_EDGE = 0.00      # Minimum mathematical edge required (e.g., 0.01 = 1%)
MAX_DAYS = 420       # Ignore markets resolving beyond X days
MIN_DAYS = 0         # Ignore markets resolving within X days
EXTREME_ODDS = 0.00  # Tail-odds filter: skip near 0% or 100%
KELLY_GAMMA = 0.5    # 0.5 = Balanced (sqrt, Fisher geodesic), 1.0 = Full Eff, 0.0 = Normal Kelly
EGO_WEIGHTING = "kelly"  # Kelly capital allocation fraction (pure, authorized capital weighting)

api = requests.Session()
