"""Localhost MetaMask EIP-712 browser signer bridge for zero-key Polymarket CLOB execution."""
import html
import json
import secrets
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from eth_utils import to_checksum_address
from py_clob_client.clob_types import ApiCreds, OrderType, RequestArgs
from py_clob_client.config import get_contract_config
from py_clob_client.constants import ZERO_ADDRESS
from py_clob_client.endpoints import CREATE_API_KEY, DERIVE_API_KEY, GET_FEE_RATE, POST_ORDER
from py_clob_client.headers.headers import (
    POLY_ADDRESS,
    POLY_NONCE,
    POLY_SIGNATURE,
    POLY_TIMESTAMP,
    create_level_2_headers,
)
from py_clob_client.order_builder.builder import ROUNDING_CONFIG, OrderBuilder
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.signing.eip712 import MSG_TO_SIGN, get_clob_auth_domain
from py_clob_client.signing.model import ClobAuth
from py_clob_client.utilities import order_to_json
from py_order_utils.builders.base_builder import BaseBuilder
from py_order_utils.model.order import Order, SignedOrder
from py_order_utils.utils import generate_seed

from config import api


_UINT256_ORDER_FIELDS = ("salt", "tokenId", "makerAmount", "takerAmount", "expiration", "nonce", "feeRateBps")
_SESSION_L2_CREDS = {}
_CACHE_LOCK = threading.Lock()


class _AddressSigner:
    """Minimal Signer adapter exposing address/chain_id without holding a private key."""

    def __init__(self, address, chain_id=137):
        self._address = to_checksum_address(address)
        self._chain_id = int(chain_id)

    def address(self):
        return self._address

    def get_chain_id(self):
        return self._chain_id


def clear_cached_l2_creds():
    """Clears in-memory session CLOB API credentials."""
    with _CACHE_LOCK:
        _SESSION_L2_CREDS.clear()


def get_cached_l2_creds(host, signer_address, chain_id=137):
    """Returns cached in-memory ApiCreds for (host, signer_address, chain_id), if derived."""
    key = (str(host).rstrip("/"), str(signer_address).lower(), int(chain_id))
    with _CACHE_LOCK:
        return _SESSION_L2_CREDS.get(key)


def set_cached_l2_creds(host, signer_address, creds, chain_id=137):
    """Stores derived ApiCreds in memory for the current CLI session."""
    key = (str(host).rstrip("/"), str(signer_address).lower(), int(chain_id))
    with _CACHE_LOCK:
        _SESSION_L2_CREDS[key] = creds


def resolve_initial_l2_creds(conf, signer_address):
    """Returns env/cached L2 ApiCreds if present without prompting MetaMask L1 auth."""
    host = str(conf.get("host") or "https://clob.polymarket.com").rstrip("/")
    chain_id = int(conf.get("chain_id") or 137)
    cached = get_cached_l2_creds(host, signer_address, chain_id)
    if cached is not None:
        return cached
    if conf.get("api_key") and conf.get("api_secret") and conf.get("api_passphrase"):
        return ApiCreds(
            api_key=conf["api_key"],
            api_secret=conf["api_secret"],
            api_passphrase=conf["api_passphrase"],
        )
    return None


def build_auth_typed_data(signer_address, chain_id=137, timestamp=None, nonce=0):
    """Builds EIP-712 ClobAuthDomain payload compatible with MetaMask eth_signTypedData_v4."""
    addr = to_checksum_address(signer_address)
    ts = int(timestamp if timestamp is not None else datetime.now().timestamp())
    auth = ClobAuth(
        address=addr,
        timestamp=str(ts),
        nonce=int(nonce),
        message=MSG_TO_SIGN,
    )
    msg = auth.to_message(get_clob_auth_domain(int(chain_id)))
    return msg, ts, int(nonce)


def derive_l2_creds_with_signature(host, signer_address, signature, timestamp, nonce=0, chain_id=137, timeout=10):
    """Exchanges a MetaMask L1 EIP-712 signature for CLOB L2 API creds and caches in memory."""
    base = str(host).rstrip("/")
    addr = to_checksum_address(signer_address)
    headers = {
        POLY_ADDRESS: addr,
        POLY_SIGNATURE: str(signature),
        POLY_TIMESTAMP: str(timestamp),
        POLY_NONCE: str(nonce),
    }

    raw = None
    last_err = ""
    for method, ep in (("GET", DERIVE_API_KEY), ("POST", CREATE_API_KEY)):
        try:
            r = api.request(method, f"{base}{ep}", headers=headers, timeout=timeout)
            data = r.json() if r.content else {}
            if r.status_code == 200 and isinstance(data, dict) and data.get("apiKey"):
                raw = data
                break
            last_err = (data.get("error") if isinstance(data, dict) else None) or f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)

    if raw is None:
        raise ValueError(f"Failed to derive CLOB API credentials: {last_err}")

    creds = ApiCreds(
        api_key=raw["apiKey"],
        api_secret=raw["secret"],
        api_passphrase=raw["passphrase"],
    )
    set_cached_l2_creds(base, addr, creds, chain_id=chain_id)
    return creds


def fetch_token_fee_rate_bps(host, token_id, timeout=6):
    """Fetches maker/taker base fee rate in bps for a token (0 if unavailable)."""
    base = str(host).rstrip("/")
    try:
        r = api.get(f"{base}{GET_FEE_RATE}?token_id={token_id}", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return int(data.get("base_fee") or 0)
    except Exception:
        pass
    return 0


def build_order_typed_data(preview, conf, signer_address, fee_rate_bps=None, salt=None, nonce=0, expiration=0):
    """Builds Polymarket CTF Exchange EIP-712 typed data + order payload for MetaMask signing."""
    if not isinstance(preview, dict) or not preview.get("ok"):
        raise ValueError((preview or {}).get("reason") or "Invalid order preview")

    chain_id = int(conf.get("chain_id") or 137)
    sig_type = int(conf.get("signature_type") or 0)
    signer_cs = to_checksum_address(signer_address)
    if sig_type == 0:
        pid = str(conf.get("profile_id") or "").strip()
        if pid.startswith("0x") and pid.lower() != signer_cs.lower():
            raise ValueError(
                f"Active MetaMask account {signer_cs} does not match EOA profile {pid}. "
                "Switch account in MetaMask or update profile/signature type in Settings."
            )
        maker_cs = signer_cs
    else:
        funder = conf.get("funder") or conf.get("profile_id") or signer_cs
        maker_cs = to_checksum_address(funder)

    side_str = BUY if preview["side"] == "BUY" else SELL
    tick_key = str(preview.get("tick_size") or "0.01")
    round_cfg = ROUNDING_CONFIG.get(tick_key, ROUNDING_CONFIG["0.01"])
    ot_str = str(preview.get("order_type") or "FAK").upper()

    ob = OrderBuilder.__new__(OrderBuilder)
    if ot_str == "GTC":
        size_sh = float(preview["est_shares"] if side_str == BUY else preview["shares"])
        side_int, maker_amt, taker_amt = ob.get_order_amounts(side_str, size_sh, float(preview["price_cap"]), round_cfg)
    else:
        amt_val = float(preview["amount_usdc"] if side_str == BUY else preview["shares"])
        side_int, maker_amt, taker_amt = ob.get_market_order_amounts(side_str, amt_val, float(preview["price_cap"]), round_cfg)

    host = str(conf.get("host") or "https://clob.polymarket.com").rstrip("/")
    fee_bps = int(fee_rate_bps) if fee_rate_bps is not None else fetch_token_fee_rate_bps(host, str(preview["token_id"]))
    neg_risk = bool(preview.get("neg_risk", False))
    contract_cfg = get_contract_config(chain_id, neg_risk=neg_risk)

    salt_val = int(salt if salt is not None else generate_seed())
    order_struct = Order(
        salt=salt_val,
        maker=maker_cs,
        signer=signer_cs,
        taker=ZERO_ADDRESS,
        tokenId=int(str(preview["token_id"])),
        makerAmount=int(maker_amt),
        takerAmount=int(taker_amt),
        expiration=int(expiration),
        nonce=int(nonce),
        feeRateBps=fee_bps,
        side=int(side_int),
        signatureType=sig_type,
    )
    domain = BaseBuilder._get_domain_separator(None, chain_id, contract_cfg.exchange)
    typed_data = order_struct.to_message(domain)
    # MetaMask JSON.parse drops precision on 77-digit tokenIds unless sent as decimal strings.
    for field in _UINT256_ORDER_FIELDS:
        typed_data["message"][field] = str(typed_data["message"][field])

    order_dict = SignedOrder(order=order_struct, signature="").dict()
    return typed_data, order_dict


def submit_metamask_signed_order(host, chain_id, signer_address, creds, order_dict, signature, order_type="FAK", timeout=10):
    """Attaches MetaMask signature + L2 HMAC headers and posts the order to Polymarket CLOB."""
    base = str(host).rstrip("/")
    signed_payload = dict(order_dict)
    signed_payload["signature"] = str(signature)

    class _SignedDictOrder:
        def __init__(self, d):
            self._d = d

        def dict(self):
            return dict(self._d)

    ot_enum = getattr(OrderType, str(order_type or "FAK").upper(), OrderType.FAK)
    body = order_to_json(_SignedDictOrder(signed_payload), creds.api_key, ot_enum, post_only=False)
    serialized = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    req_args = RequestArgs(method="POST", request_path=POST_ORDER, body=body, serialized_body=serialized)
    headers = create_level_2_headers(_AddressSigner(signer_address, chain_id), creds, req_args)
    headers["Content-Type"] = "application/json"

    from trade import normalize_clob_response

    r = api.post(f"{base}{POST_ORDER}", headers=headers, data=serialized, timeout=timeout)
    try:
        resp = r.json()
    except Exception:
        resp = {"errorMsg": r.text or f"HTTP {r.status_code}"}

    return normalize_clob_response(resp, http_status=r.status_code)


def _render_signer_html(preview, conf, summary, csrf_token):
    """Renders the localhost MetaMask EIP-712 confirmation UI."""
    s = summary or {}
    q_txt = html.escape(str(s.get("question") or "Polymarket Order"))
    side_str = str(preview.get("side", "BUY")).upper()
    ot_str = str(preview.get("order_type", "FAK")).upper()
    out_txt = html.escape(str(s.get("outcome") or "").upper())
    cap_c = float(preview.get("price_cap") or 0.0) * 100.0
    vwap_pct = float(preview.get("est_vwap") or 0.0) * 100.0
    expected_signer = str(conf.get("metamask_signer") or "").strip()
    chain_id = int(conf.get("chain_id") or 137)

    if side_str == "BUY":
        size_line = f"${float(preview.get('amount_usdc') or 0):,.2f} USDC (~{float(preview.get('est_shares') or 0):,.2f} sh)"
    else:
        size_line = f"{float(preview.get('shares') or 0):,.2f} sh (~${float(preview.get('est_proceeds') or 0):,.2f} USDC)"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Polymarket MetaMask Signer</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background:#0f141c; color:#e6edf3; margin:0; padding:32px; display:flex; justify-content:center; }}
  .card {{ max-width:540px; width:100%; background:#161d27; border:1px solid #2b3545; border-radius:12px; padding:24px; }}
  h2 {{ margin:0 0 8px 0; font-size:18px; }}
  .badge {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:12px; font-weight:600; background:#1f3a5f; color:#7cc4ff; margin-bottom:14px; }}
  .row {{ display:flex; justify-content:space-between; gap:12px; padding:8px 0; border-bottom:1px solid #232c3b; font-size:14px; }}
  .lbl {{ color:#8b9bb4; }}
  .val {{ font-weight:600; text-align:right; word-break:break-all; }}
  .actions {{ display:flex; gap:10px; margin-top:20px; }}
  button {{ flex:1; padding:12px 16px; border:none; border-radius:8px; font-size:14px; font-weight:600; cursor:pointer; }}
  #signBtn {{ background:#2ea043; color:#fff; }}
  #signBtn:disabled {{ opacity:0.5; cursor:wait; }}
  #cancelBtn {{ flex:0.4; background:#2b3545; color:#d0d7de; }}
  #status {{ margin-top:16px; padding:12px; border-radius:8px; background:#0d1117; border:1px solid #232c3b; font-size:13px; white-space:pre-wrap; }}
  .ok {{ color:#3fb950; }} .err {{ color:#f85149; }}
</style>
</head>
<body>
<div class="card">
  <span class="badge">Zero-Key MetaMask EIP-712 Signer</span>
  <h2>{q_txt}</h2>
  <div class="row"><span class="lbl">Action</span><span class="val">{side_str} {out_txt} ({ot_str})</span></div>
  <div class="row"><span class="lbl">Size</span><span class="val">{html.escape(size_line)}</span></div>
  <div class="row"><span class="lbl">Est VWAP / Cap</span><span class="val">{vwap_pct:.1f}% (limit/cap {cap_c:.1f}c)</span></div>
  <div class="row"><span class="lbl">Profile / Funder</span><span class="val">{html.escape(str(conf.get('funder') or conf.get('profile_id') or ''))}</span></div>
  <div class="row"><span class="lbl">Signature Type</span><span class="val">{int(conf.get('signature_type') or 0)} ({html.escape(str(conf.get('signature_label') or ''))})</span></div>
  <div class="row"><span class="lbl">Token ID</span><span class="val">{html.escape(str(preview.get('token_id') or ''))}</span></div>
  {f'<div class="row"><span class="lbl">Expected Signer</span><span class="val">{html.escape(expected_signer)}</span></div>' if expected_signer else ''}
  <div class="actions">
    <button id="signBtn" onclick="runSign()">Connect MetaMask &amp; Sign Order</button>
    <button id="cancelBtn" onclick="cancelFlow()">Cancel</button>
  </div>
  <div id="status">Waiting to connect MetaMask...</div>
</div>
<script>
const TOKEN = {json.dumps(csrf_token)};
const EXPECTED_SIGNER = {json.dumps(expected_signer.lower())};
const CHAIN_HEX = "0x" + ({chain_id}).toString(16);

function setStatus(msg, cls="") {{
  const el = document.getElementById("status");
  el.className = cls;
  el.textContent = msg;
}}

async function postJson(path, payload) {{
  const res = await fetch(path, {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{...payload, token: TOKEN}})
  }});
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}}

async function ensurePolygon(eth) {{
  const cur = await eth.request({{method: "eth_chainId"}});
  if (cur && cur.toLowerCase() === CHAIN_HEX.toLowerCase()) return;
  try {{
    await eth.request({{method: "wallet_switchEthereumChain", params: [{{chainId: CHAIN_HEX}}]}});
  }} catch (e) {{
    if (e.code === 4902 && CHAIN_HEX === "0x89") {{
      await eth.request({{
        method: "wallet_addEthereumChain",
        params: [{{
          chainId: "0x89",
          chainName: "Polygon Mainnet",
          rpcUrls: ["https://polygon-bor-rpc.publicnode.com"],
          nativeCurrency: {{name: "POL", symbol: "POL", decimals: 18}},
          blockExplorerUrls: ["https://polygonscan.com"]
        }}]
      }});
    }} else {{
      throw e;
    }}
  }}
}}

async function runSign() {{
  const btn = document.getElementById("signBtn");
  btn.disabled = true;
  try {{
    if (!window.ethereum) throw new Error("MetaMask / Web3 wallet extension not detected in this browser.");
    setStatus("Connecting MetaMask...");
    const accounts = await window.ethereum.request({{method: "eth_requestAccounts"}});
    const signer = (accounts && accounts[0]) || "";
    if (!signer) throw new Error("No MetaMask account unlocked.");
    if (EXPECTED_SIGNER && signer.toLowerCase() !== EXPECTED_SIGNER) {{
      throw new Error("Connected account " + signer + " does not match configured signer " + EXPECTED_SIGNER + ". Switch account in MetaMask and retry.");
    }}
    await ensurePolygon(window.ethereum);

    setStatus("Checking CLOB API session...");
    const authPrep = await postJson("/prepare-auth", {{signer}});
    if (authPrep.need_auth) {{
      setStatus("Step 1/2: Approve L1 CLOB Auth signature in MetaMask...");
      const authSig = await window.ethereum.request({{
        method: "eth_signTypedData_v4",
        params: [signer, JSON.stringify(authPrep.typed_data)]
      }});
      await postJson("/submit-auth", {{
        signer,
        signature: authSig,
        timestamp: authPrep.timestamp,
        nonce: authPrep.nonce
      }});
    }}

    setStatus("Step 2/2: Approve exact Order signature in MetaMask...");
    const orderPrep = await postJson("/prepare-order", {{signer}});
    const orderSig = await window.ethereum.request({{
      method: "eth_signTypedData_v4",
      params: [signer, JSON.stringify(orderPrep.typed_data)]
    }});

    setStatus("Submitting signed order to Polymarket CLOB...");
    const res = await postJson("/submit-order", {{
      signer,
      signature: orderSig,
      order_dict: orderPrep.order_dict
    }});
    if (res.ok) {{
      setStatus("Order submitted: " + (res.status || "matched") + " (ID: " + (res.order_id || "N/A") + "). You may close this tab.", "ok");
    }} else {{
      setStatus("Order rejected: " + (res.error || "unknown error"), "err");
      btn.disabled = false;
    }}
  }} catch (err) {{
    setStatus("Error: " + (err && err.message ? err.message : String(err)), "err");
    btn.disabled = false;
  }}
}}

async function cancelFlow() {{
  try {{ await postJson("/cancel", {{}}); }} catch (_) {{}}
  setStatus("Cancelled. You may close this tab.");
}}
</script>
</body>
</html>"""


def create_metamask_bridge_server(preview, conf, summary=None):
    """Creates an ephemeral 127.0.0.1 HTTP server coordinating MetaMask signing & CLOB submission."""
    csrf_token = secrets.token_urlsafe(16)
    done_event = threading.Event()
    state = {"result": None, "pending_order": None}
    html_doc = _render_signer_html(preview, conf, summary, csrf_token).encode("utf-8")
    host = str(conf.get("host") or "https://clob.polymarket.com").rstrip("/")
    chain_id = int(conf.get("chain_id") or 137)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send_json(self, code, payload):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self):
            ln = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(ln) if ln > 0 else b"{}"
            return json.loads(body.decode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/" and (qs.get("token") or [""])[0] == csrf_token:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_doc)))
                self.end_headers()
                self.wfile.write(html_doc)
                return
            self.send_error(404)

        def do_POST(self):
            try:
                data = self._read_json()
                if data.get("token") != csrf_token:
                    self._send_json(403, {"error": "Invalid session token"})
                    return

                if self.path == "/prepare-auth":
                    signer = to_checksum_address(data["signer"])
                    creds = resolve_initial_l2_creds(conf, signer)
                    if creds is not None:
                        self._send_json(200, {"need_auth": False})
                        return
                    typed_data, ts, nonce = build_auth_typed_data(signer, chain_id=chain_id)
                    self._send_json(200, {"need_auth": True, "typed_data": typed_data, "timestamp": ts, "nonce": nonce})
                    return

                if self.path == "/submit-auth":
                    derive_l2_creds_with_signature(
                        host,
                        data["signer"],
                        data["signature"],
                        data["timestamp"],
                        nonce=data.get("nonce", 0),
                        chain_id=chain_id,
                    )
                    self._send_json(200, {"ok": True})
                    return

                if self.path == "/prepare-order":
                    signer = to_checksum_address(data["signer"])
                    typed_data, order_dict = build_order_typed_data(preview, conf, signer)
                    state["pending_order"] = {"signer": signer, "order_dict": order_dict}
                    self._send_json(200, {"typed_data": typed_data, "order_dict": order_dict})
                    return

                if self.path == "/submit-order":
                    signer = to_checksum_address(data["signer"])
                    creds = resolve_initial_l2_creds(conf, signer)
                    if creds is None:
                        self._send_json(400, {"error": "CLOB API credentials not derived yet."})
                        return
                    order_dict = (state.get("pending_order") or {}).get("order_dict") or data["order_dict"]
                    res = submit_metamask_signed_order(
                        host,
                        chain_id,
                        signer,
                        creds,
                        order_dict,
                        data["signature"],
                        order_type=preview.get("order_type", "FAK"),
                    )
                    state["result"] = res
                    if res.get("ok"):
                        done_event.set()
                    self._send_json(200, res)
                    return

                if self.path == "/cancel":
                    state["result"] = {"ok": False, "order_id": "", "status": "cancelled", "error": "Cancelled in MetaMask UI", "raw": None}
                    done_event.set()
                    self._send_json(200, {"ok": True})
                    return

                self._send_json(404, {"error": "Unknown route"})
            except Exception as e:
                self._send_json(400, {"error": str(e)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/?token={csrf_token}"
    return server, url, done_event, state


def sign_and_submit_with_metamask(preview, conf, summary=None, timeout=180):
    """Launches localhost MetaMask signer page, waits for signature, and submits CLOB order."""
    if not isinstance(preview, dict) or not preview.get("ok"):
        return {"ok": False, "error": (preview or {}).get("reason", "Invalid order preview")}

    from trade import open_polymarket_url

    server, url, done_event, state = create_metamask_bridge_server(preview, conf, summary=summary)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        print(f"[*] Opening MetaMask signer: {url}")
        open_polymarket_url(url, browser_pref=conf.get("signing_browser"))
        print("[*] Waiting for MetaMask confirmation in browser (Ctrl+C to cancel)...")
        if not done_event.wait(timeout=timeout):
            return {"ok": False, "order_id": "", "status": "timeout", "error": "MetaMask signing timed out.", "raw": None}
        return state.get("result") or {"ok": False, "order_id": "", "status": "cancelled", "error": "No result", "raw": None}
    except KeyboardInterrupt:
        return {"ok": False, "order_id": "", "status": "cancelled", "error": "Cancelled by user.", "raw": None}
    finally:
        server.shutdown()
        server.server_close()
