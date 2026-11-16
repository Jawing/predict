"""Configuration parameters and API constants for the prediction pipeline."""
import requests
from requests.adapters import HTTPAdapter

HISTORY_DIR = "data"
HISTORY_FILE = "prediction_history.json"
ACTIVE_PROFILE_FILE = "active_profile.txt"
BANKROLL = 100.00
GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_WALLET = "0xbC730b41A9aaAd603dDA9473DcB8a059dF2c9fa7"  # Default / primary Polymarket wallet address

# --- FILTERING PARAMETERS ---
MIN_EDGE = 0.00      # Minimum mathematical edge required (e.g., 0.01 = 1%)
MAX_DAYS = 420       # Ignore markets resolving beyond X days
MIN_DAYS = 0         # Ignore markets resolving within X days
EXTREME_ODDS = 0.00  # Tail-odds filter: skip near 0% or 100%
KELLY_GAMMA = 0.5    # 0.5 = Balanced (sqrt, Fisher geodesic), 1.0 = Full Eff, 0.0 = Normal Kelly
EGO_WEIGHTING = "uniform"  # Equal 1.0 weight per blind pu forecast (full accountability; "kelly" weights by backed capital)

api = requests.Session()
_http_adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
api.mount("https://", _http_adapter)
api.mount("http://", _http_adapter)
