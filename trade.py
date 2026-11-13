"""Live Polymarket CLOB order preview, signing, and execution via py-clob-client."""
import math
import webbrowser
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
import config as cfg
from config import api
from storage import get_active_profile
from model import calculate_optimal_market_buy, calculate_optimal_market_sell


POLYGON_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://1rpc.io/matic",
    "https://polygon.drpc.org",
]

SIGNATURE_TYPE_LABELS = {
    0: "EOA (Direct Wallet)",
    1: "POLY_PROXY (Email/Magic Proxy)",
    2: "POLY_GNOSIS_SAFE (Browser Proxy)",
}

_SIG_TYPE_CACHE = {}


def detect_wallet_signature_type(wallet_address):
    """
    Auto-detects Polymarket CLOB signature_type for an address via Polygon RPC:
      - 0 (EOA): no contract bytecode at address
      - 2 (POLY_GNOSIS_SAFE): contract responds to Gnosis Safe getOwners()
      - 1 (POLY_PROXY): other proxy contract (e.g. Polymarket/Magic proxy with owner())
    Returns (signature_type, source_note).
    """
    if not wallet_address or not str(wallet_address).startswith("0x"):
        return 0, "paper/default"

    addr = str(wallet_address).strip()
    key = addr.lower()
    if key in _SIG_TYPE_CACHE:
        return _SIG_TYPE_CACHE[key]

    batch = [
        {"jsonrpc": "2.0", "method": "eth_getCode", "params": [addr, "latest"], "id": 1},
        {"jsonrpc": "2.0", "method": "eth_call", "params": [{"to": addr, "data": "0xa0e67e2b"}, "latest"], "id": 2},
        {"jsonrpc": "2.0", "method": "eth_call", "params": [{"to": addr, "data": "0x8da5cb5b"}, "latest"], "id": 3},
    ]

    for rpc in POLYGON_RPCS:
        try:
            resp = api.post(rpc, json=batch, timeout=5).json()
            if not isinstance(resp, list):
                continue
            by_id = {item.get("id"): item for item in resp if isinstance(item, dict)}
            code = (by_id.get(1) or {}).get("result", "")
            if not code:
                continue
            if code in ("0x", "0x0"):
                res = (0, "auto-detected EOA")
            elif (by_id.get(2) or {}).get("result") and "error" not in (by_id.get(2) or {}):
                res = (2, "auto-detected Gnosis Safe")
            elif (by_id.get(3) or {}).get("result") and "error" not in (by_id.get(3) or {}):
                res = (1, "auto-detected Polymarket Proxy")
            else:
                res = (2, "auto-detected contract wallet")
            _SIG_TYPE_CACHE[key] = res
            return res
        except Exception:
            continue

    return 1, "fallback proxy (RPC unavailable)"


def get_clob_config(profile_id=None, detect_remote=True):
    """Resolves active wallet, private key, signature_type, and funder settings."""
    pid = str(profile_id or get_active_profile() or "").strip()
    is_live_wallet = pid.startswith("0x") and len(pid) >= 10
    priv_key = str(getattr(cfg, "CLOB_PRIVATE_KEY", "") or "").strip()
    if priv_key and not priv_key.startswith("0x"):
        priv_key = "0x" + priv_key

    sig_override = getattr(cfg, "CLOB_SIGNATURE_TYPE", None)
    if sig_override in (0, 1, 2):
        sig_type = int(sig_override)
        sig_note = "configured"
    elif is_live_wallet and detect_remote:
        sig_type, sig_note = detect_wallet_signature_type(pid)
    else:
        sig_type, sig_note = 0, "default"

    funder_override = str(getattr(cfg, "CLOB_FUNDER", "") or "").strip()
    funder = funder_override if funder_override else (pid if (is_live_wallet and sig_type in (1, 2)) else None)

    return {
        "profile_id": pid,
        "is_live_wallet": is_live_wallet,
        "host": getattr(cfg, "CLOB_HOST", "https://clob.polymarket.com"),
        "chain_id": int(getattr(cfg, "CLOB_CHAIN_ID", 137)),
        "private_key": priv_key,
        "has_private_key": bool(priv_key),
        "api_key": str(getattr(cfg, "CLOB_API_KEY", "") or "").strip(),
        "api_secret": str(getattr(cfg, "CLOB_API_SECRET", "") or "").strip(),
        "api_passphrase": str(getattr(cfg, "CLOB_API_PASSPHRASE", "") or "").strip(),
        "signature_type": sig_type,
        "signature_label": SIGNATURE_TYPE_LABELS.get(sig_type, f"Type {sig_type}"),
        "signature_note": sig_note,
        "funder": funder,
    }


def is_live_trading_ready(profile_id=None):
    """Returns True when active profile is a 0x wallet and a signing key is configured."""
    conf = get_clob_config(profile_id=profile_id, detect_remote=False)
    return bool(conf["is_live_wallet"] and conf["has_private_key"])


def get_clob_client(profile_id=None):
    """Initializes an authenticated py-clob-client ClobClient for the active wallet."""
    conf = get_clob_config(profile_id=profile_id, detect_remote=True)
    if not conf["is_live_wallet"]:
        raise ValueError(f"Active profile '{conf['profile_id']}' is not a live 0x wallet.")
    if not conf["has_private_key"]:
        raise ValueError("CLOB_PRIVATE_KEY is not configured. Set it in .env or config.py.")

    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    creds = None
    if conf["api_key"] and conf["api_secret"] and conf["api_passphrase"]:
        creds = ApiCreds(
            api_key=conf["api_key"],
            api_secret=conf["api_secret"],
            api_passphrase=conf["api_passphrase"],
        )

    client_kwargs = {
        "host": conf["host"],
        "chain_id": conf["chain_id"],
        "key": conf["private_key"],
        "signature_type": conf["signature_type"],
    }
    if creds is not None:
        client_kwargs["creds"] = creds
    if conf["funder"]:
        client_kwargs["funder"] = conf["funder"]

    client = ClobClient(**client_kwargs)
    if creds is None:
        derived = client.create_or_derive_api_creds()
        client.set_api_creds(derived)
    return client


def round_price_to_tick(price, tick_size="0.01", mode="nearest"):
    """Rounds a probability price cleanly to the market's CLOB tick_size within [tick, 1 - tick]."""
    tick_d = Decimal(str(tick_size or "0.01"))
    if tick_d <= 0:
        tick_d = Decimal("0.01")
    rounding = {
        "up": ROUND_UP,
        "down": ROUND_DOWN,
        "nearest": ROUND_HALF_UP,
    }.get(mode, ROUND_HALF_UP)

    px_d = Decimal(str(float(price)))
    rounded = (px_d / tick_d).quantize(Decimal("1"), rounding=rounding) * tick_d
    min_px = tick_d
    max_px = Decimal("1") - tick_d
    clamped = max(min_px, min(max_px, rounded))
    return float(clamped)


def fetch_orderbook(token_id):
    """Fetches live L2 orderbook and market execution constraints from Polymarket CLOB."""
    if not token_id:
        return None
    r = api.get(f"{cfg.CLOB_HOST}/book?token_id={token_id}", timeout=6)
    if r.status_code != 200:
        raise ValueError(f"CLOB orderbook HTTP {r.status_code} for token {token_id}")
    book = r.json()
    if not isinstance(book, dict) or book.get("error"):
        raise ValueError(book.get("error") if isinstance(book, dict) else "Invalid orderbook response")

    asks = sorted(
        [(float(a["price"]), float(a["size"])) for a in book.get("asks", []) if float(a.get("price", 0)) > 0 and float(a.get("size", 0)) > 0],
        key=lambda x: x[0],
    )
    bids = sorted(
        [(float(b["price"]), float(b["size"])) for b in book.get("bids", []) if float(b.get("price", 0)) > 0 and float(b.get("size", 0)) > 0],
        key=lambda x: x[0],
        reverse=True,
    )
    return {
        "token_id": str(token_id),
        "asks": asks,
        "bids": bids,
        "best_ask": asks[0][0] if asks else None,
        "best_bid": bids[0][0] if bids else None,
        "tick_size": str(book.get("tick_size") or "0.01"),
        "min_order_size": float(book.get("min_order_size") or 1.0),
        "neg_risk": bool(book.get("neg_risk", False)),
    }


def _build_base_preview(book, side, order_type, profile_id):
    conf = get_clob_config(profile_id=profile_id, detect_remote=False)
    return {
        "ok": True,
        "reason": "",
        "side": side,
        "order_type": order_type,
        "token_id": book["token_id"],
        "tick_size": book["tick_size"],
        "min_order_size": book["min_order_size"],
        "neg_risk": book["neg_risk"],
        "best_bid": book["best_bid"],
        "best_ask": book["best_ask"],
        "profile_id": conf["profile_id"],
        "is_live_wallet": conf["is_live_wallet"],
        "has_private_key": conf["has_private_key"],
    }


def _floor_shares(val):
    """Floors share quantities to 2 decimals so orders never exceed held balance."""
    return math.floor(max(0.0, float(val or 0.0)) * 100.0 + 1e-9) / 100.0


def preview_buy_order(
    token_id,
    usdc_amount,
    p_norm=None,
    fee_rate=0.0,
    min_edge=0.0,
    order_type="FAK",
    limit_price=None,
    profile_id=None,
):
    """Builds and validates a BUY order preview (FAK market or GTC limit) against the live book."""
    ot = str(order_type or "FAK").upper()
    book = fetch_orderbook(token_id)
    prev = _build_base_preview(book, "BUY", ot, profile_id)
    prev["amount_usdc"] = round(float(usdc_amount or 0.0), 2)
    prev["est_shares"] = 0.0
    prev["est_vwap"] = 0.0
    prev["price_cap"] = None

    if not prev["is_live_wallet"]:
        prev["ok"] = False
        prev["reason"] = "Active profile is a Paper Sandbox; live orders require a 0x wallet profile."
        return prev

    if prev["amount_usdc"] <= 0:
        prev["ok"] = False
        prev["reason"] = "Buy amount must be greater than $0."
        return prev

    max_order_usd = float(getattr(cfg, "CLOB_MAX_ORDER_USD", 0.0) or 0.0)
    if prev["has_private_key"] and max_order_usd > 0 and prev["amount_usdc"] > max_order_usd:
        prev["ok"] = False
        prev["reason"] = f"Buy amount ${prev['amount_usdc']:,.2f} exceeds CLOB_MAX_ORDER_USD (${max_order_usd:,.2f})."
        return prev

    tick_size = book["tick_size"]
    min_shares = book["min_order_size"]

    if ot == "GTC":
        raw_px = float(limit_price) if (limit_price is not None and limit_price > 0) else (book["best_ask"] or book["best_bid"] or 0.5)
        px = round_price_to_tick(raw_px, tick_size, mode="nearest")
        shares = round(prev["amount_usdc"] / px, 2)
        prev["price_cap"] = px
        prev["est_vwap"] = px
        prev["est_shares"] = shares
        if shares < min_shares:
            prev["ok"] = False
            prev["reason"] = f"Order size {shares:.2f} shares is below market minimum ({min_shares:g} shares)."
        return prev

    asks = book["asks"]
    if not asks:
        prev["ok"] = False
        prev["reason"] = "No asks available on orderbook."
        return prev

    has_hurdle = p_norm is not None and p_norm > 0
    fill_usd, vwap, _ = calculate_optimal_market_buy(
        asks,
        prev["amount_usdc"],
        p_norm=float(p_norm) if has_hurdle else 1.0,
        fee_rate=float(fee_rate or 0.0) if has_hurdle else 0.0,
        min_edge=float(min_edge or 0.0) if has_hurdle else 0.0,
    )
    if fill_usd <= 0:
        prev["ok"] = False
        prev["reason"] = f"Best ask {asks[0][0]*100:.1f}% exceeds belief hurdle."
        return prev
    prev["amount_usdc"] = round(fill_usd, 2)
    prev["est_vwap"] = vwap

    # Find highest ask level touched for worst-price cap
    cum = 0.0
    worst_ask = asks[0][0]
    for p, s in asks:
        cum += p * s
        worst_ask = p
        if cum >= prev["amount_usdc"] - 1e-6:
            break

    cap_px = round_price_to_tick(worst_ask, tick_size, mode="up")
    est_shares = round(prev["amount_usdc"] / max(prev["est_vwap"], 1e-4), 2)
    prev["price_cap"] = cap_px
    prev["est_shares"] = est_shares
    if est_shares < min_shares:
        prev["ok"] = False
        prev["reason"] = f"Estimated size {est_shares:.2f} shares is below market minimum ({min_shares:g} shares)."
    return prev


def preview_sell_order(
    token_id,
    shares,
    min_price=0.0,
    order_type="FAK",
    limit_price=None,
    profile_id=None,
):
    """Builds and validates a SELL order preview (FAK market or GTC limit) against the live book."""
    ot = str(order_type or "FAK").upper()
    book = fetch_orderbook(token_id)
    prev = _build_base_preview(book, "SELL", ot, profile_id)
    req_shares = _floor_shares(shares)
    prev["shares"] = req_shares
    prev["est_vwap"] = 0.0
    prev["est_proceeds"] = 0.0
    prev["price_cap"] = None

    if not prev["is_live_wallet"]:
        prev["ok"] = False
        prev["reason"] = "Active profile is a Paper Sandbox; live orders require a 0x wallet profile."
        return prev

    min_shares = book["min_order_size"]
    if req_shares < min_shares:
        prev["ok"] = False
        prev["reason"] = f"Sell size {req_shares:.2f} shares is below market minimum ({min_shares:g} shares)."
        return prev

    tick_size = book["tick_size"]

    if ot == "GTC":
        raw_px = float(limit_price) if (limit_price is not None and limit_price > 0) else (book["best_ask"] or book["best_bid"] or 0.5)
        px = round_price_to_tick(raw_px, tick_size, mode="nearest")
        prev["price_cap"] = px
        prev["est_vwap"] = px
        prev["est_proceeds"] = round(req_shares * px, 2)
        return prev

    bids = book["bids"]
    if not bids:
        prev["ok"] = False
        prev["reason"] = "No bids available on orderbook."
        return prev

    fill_sh, vwap, proceeds, worst_bid = calculate_optimal_market_sell(bids, req_shares, min_price=float(min_price or 0.0))
    if fill_sh < min_shares:
        prev["ok"] = False
        prev["reason"] = f"Available bid depth ({fill_sh:.2f} sh) above floor is below minimum ({min_shares:g} sh)."
        return prev

    floor_px = round_price_to_tick(max(float(min_price or 0.0), worst_bid), tick_size, mode="down")
    prev["shares"] = _floor_shares(fill_sh)
    prev["est_vwap"] = vwap
    prev["est_proceeds"] = proceeds
    prev["price_cap"] = floor_px
    return prev


def execute_order(preview, profile_id=None, client=None):
    """Signs and posts a validated order preview via py-clob-client."""
    if not isinstance(preview, dict) or not preview.get("ok"):
        return {
            "ok": False,
            "error": (preview or {}).get("reason", "Invalid order preview"),
        }

    from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType, PartialCreateOrderOptions
    from py_clob_client.order_builder.constants import BUY, SELL

    try:
        clob = client or get_clob_client(profile_id=profile_id)
        side_const = BUY if preview["side"] == "BUY" else SELL
        ot_str = str(preview.get("order_type", "FAK")).upper()
        ot_enum = getattr(OrderType, ot_str, OrderType.FAK)
        options = PartialCreateOrderOptions(
            tick_size=str(preview["tick_size"]),
            neg_risk=bool(preview.get("neg_risk", False)),
        )

        if ot_str == "GTC":
            size_sh = float(preview["est_shares"] if preview["side"] == "BUY" else preview["shares"])
            args = OrderArgs(
                token_id=str(preview["token_id"]),
                price=float(preview["price_cap"]),
                size=round(size_sh, 2),
                side=side_const,
            )
            signed = clob.create_order(args, options=options)
            resp = clob.post_order(signed, OrderType.GTC)
        else:
            amount_val = float(preview["amount_usdc"] if preview["side"] == "BUY" else preview["shares"])
            args = MarketOrderArgs(
                token_id=str(preview["token_id"]),
                amount=round(amount_val, 2),
                side=side_const,
                price=float(preview["price_cap"] or 0.0),
                order_type=ot_enum,
            )
            signed = clob.create_market_order(args, options=options)
            resp = clob.post_order(signed, ot_enum)

        if isinstance(resp, dict):
            err = resp.get("errorMsg") or resp.get("error") or ""
            success = bool(resp.get("success", True)) and not err
            return {
                "ok": success,
                "order_id": resp.get("orderID") or resp.get("id") or "",
                "status": resp.get("status") or ("matched" if success else "rejected"),
                "taking_amount": resp.get("takingAmount"),
                "making_amount": resp.get("makingAmount"),
                "error": err,
                "raw": resp,
            }
        return {"ok": True, "order_id": "", "status": "submitted", "error": "", "raw": resp}
    except Exception as e:
        return {"ok": False, "order_id": "", "status": "error", "error": str(e), "raw": None}


def fetch_open_orders(profile_id=None, client=None):
    """Returns open resting CLOB orders for the active wallet."""
    clob = client or get_clob_client(profile_id=profile_id)
    res = clob.get_orders()
    return res if isinstance(res, list) else (res.get("data", []) if isinstance(res, dict) else [])


def cancel_open_order(order_id, profile_id=None, client=None):
    """Cancels a specific open CLOB order by order ID."""
    clob = client or get_clob_client(profile_id=profile_id)
    return clob.cancel(order_id)


def format_browser_ticket(preview, question="", outcome="Yes", event_url=""):
    """Formats a zero-key pre-flight order ticket for manual confirmation in Polymarket UI."""
    side = str(preview.get("side", "BUY")).upper()
    ot = str(preview.get("order_type", "FAK")).upper()
    ui_order_type = "Limit" if ot == "GTC" else "Market"
    out_label = str(outcome or "Yes").upper()
    cap_cents = float(preview.get("price_cap") or 0.0) * 100.0
    vwap_pct = float(preview.get("est_vwap") or 0.0) * 100.0

    lines = [
        "--- BROWSER ORDER TICKET (Zero-Key Pre-Flight) ---",
    ]
    if question:
        lines.append(f"Market     : {question}")
    if event_url:
        lines.append(f"URL        : {event_url}")
    lines.append(f"Action     : {side} {out_label}  ({ui_order_type} Order)")
    if side == "BUY":
        amt = float(preview.get("amount_usdc") or 0.0)
        sh = float(preview.get("est_shares") or 0.0)
        px_label = "Limit Price" if ot == "GTC" else "Est VWAP / Max"
        px_val = f"{cap_cents:.1f}c" if ot == "GTC" else f"{vwap_pct:.1f}% (cap {cap_cents:.1f}c)"
        lines.append(f"Amount     : ${amt:,.2f} USDC  (~{sh:,.2f} shares)")
        lines.append(f"{px_label:<10} : {px_val}")
    else:
        sh = float(preview.get("shares") or 0.0)
        usd = float(preview.get("est_proceeds") or 0.0)
        px_label = "Limit Price" if ot == "GTC" else "Est VWAP / Min"
        px_val = f"{cap_cents:.1f}c" if ot == "GTC" else f"{vwap_pct:.1f}% (floor {cap_cents:.1f}c)"
        lines.append(f"Shares     : {sh:,.2f} shares  (~${usd:,.2f} USDC)")
        lines.append(f"{px_label:<10} : {px_val}")
    return "\n".join(lines)


def open_polymarket_url(url):
    """Opens a Polymarket event URL in the default browser."""
    if not url or not str(url).startswith("https://"):
        return False
    try:
        return bool(webbrowser.open(str(url)))
    except Exception:
        return False


def sync_wallet_after_trade(active_pid, history):
    """Re-syncs live wallet balances and positions into history in-place."""
    try:
        from storage import save_history
        from wallet import import_polymarket_positions

        print("[*] Re-syncing live wallet balances & positions...")
        updated = import_polymarket_positions(active_pid, history, auto_sync=True)
        if isinstance(updated, dict) and updated is not history:
            history.clear()
            history.update(updated)
        save_history(history, profile_id=active_pid)
        return True
    except Exception as e:
        print(f"[!] Post-trade sync warning: {e}")
        return False


def prompt_browser_confirm_and_sync(urls, active_pid, history):
    """Shared Zero-Key action prompt: open browser URL(s) and sync wallet afterward."""
    act = input("\nActions: [o] Open in Browser & Sync  [s] Sync Wallet Now  [Enter] Done\n> ").strip().lower()
    if act == "o":
        unique_urls = list(dict.fromkeys(u for u in (urls or []) if u))
        opened = sum(1 for u in unique_urls if open_polymarket_url(u))
        if opened:
            print(f"[+] Opened {opened} market tab(s) in your browser.")
        elif unique_urls:
            print(f"[*] Open manually: {', '.join(unique_urls)}")
        if input("Press Enter after confirming order(s) in browser to sync wallet (or 'q' to skip): ").strip().lower() != "q":
            sync_wallet_after_trade(active_pid, history)
            return True
        return False
    if act == "s":
        sync_wallet_after_trade(active_pid, history)
        return True
    return False
