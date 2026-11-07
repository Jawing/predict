"""Configuration parameters and API constants for the prediction pipeline."""
import requests

HISTORY_DIR = "data"
HISTORY_FILE = "prediction_history.json"
ACTIVE_PROFILE_FILE = "active_profile.txt"
BANKROLL = 100.00
GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_WALLET = "0xbC730b41A9aaAd603dDA9473DcB8a059dF2c9fa7"  # Default / primary Polymarket wallet address

# --- OPTIONAL CLOB API CREDENTIALS (FOR FUTURE AUTOMATED LIVE EXECUTION) ---
# When populated, enables automated signing and limit order execution via py-clob-client
CLOB_API_KEY = ""
CLOB_API_SECRET = ""
CLOB_API_PASSPHRASE = ""
CLOB_PRIVATE_KEY = ""

# --- FILTERING PARAMETERS ---
MIN_EDGE = 0.00      # Minimum mathematical edge required (e.g., 0.01 = 1%)
MAX_DAYS = 420       # Ignore markets resolving beyond X days
MIN_DAYS = 0         # Ignore markets resolving within X days
EXTREME_ODDS = 0.00  # Tail-odds filter: skip near 0% or 100%
KELLY_GAMMA = 0.5    # 0.5 = Balanced (sqrt, Fisher geodesic), 1.0 = Full Eff, 0.0 = Normal Kelly
EGO_WEIGHTING = "kelly"  # Kelly capital allocation fraction (pure, authorized capital weighting)

api = requests.Session()
