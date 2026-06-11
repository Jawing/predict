import json
import math

def ensure_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]

def parse_outcome_prices(m):
    """Safely extracts outcomePrices as a list of floats from a market or event response."""
    try:
        prices = m.get('outcomePrices')
        if isinstance(prices, str):
            return [float(x) for x in json.loads(prices)]
        elif isinstance(prices, list):
            return [float(x) for x in prices]
    except Exception:
        pass
    return []

def format_time_remaining(seconds):
    if seconds is None:
        return "unknown"
    if seconds <= 0:
        return "Expired"
    d = int(seconds // 86400)
    rem = seconds % 86400
    h = int(rem // 3600)
    rem %= 3600
    mins = int(rem // 60)
    s = int(rem % 60)
    if d > 0:
        return f"{d}d {h:02d}h"
    return f"{h:02d}:{mins:02d}:{s:02d}"

def stringify_overflow(obj):
    """Recursively converts infinite floats into strings."""
    if isinstance(obj, float):
        if obj == float('inf'):
            return "Infinity"
        elif obj == float('-inf'):
            return "-Infinity"
        elif math.isnan(obj):
            return "NaN"
    elif isinstance(obj, dict):
        return {k: stringify_overflow(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [stringify_overflow(x) for x in obj]
    return obj
