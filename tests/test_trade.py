"""Unit tests for live CLOB order preview, signature detection, and execution."""
import base64
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import metamask_signer
import trade


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode("utf-8") if payload is not None else b""

    def json(self):
        return self._payload


class TestSignatureDetectionAndConfig(unittest.TestCase):
    def setUp(self):
        trade._SIG_TYPE_CACHE.clear()

    def test_detect_eoa_gnosis_and_poly_proxy(self):
        with patch.object(trade.api, "post", return_value=_FakeResponse(200, [{"id": 1, "result": "0x"}])):
            sig, _ = trade.detect_wallet_signature_type("0x1111111111111111111111111111111111111111")
            self.assertEqual(sig, 0)

        with patch.object(
            trade.api,
            "post",
            return_value=_FakeResponse(
                200,
                [
                    {"id": 1, "result": "0x60806040"},
                    {"id": 2, "result": "0x0000000000000000000000000000000000000000000000000000000000000020"},
                    {"id": 3, "error": {"code": 3}},
                ],
            ),
        ):
            sig, _ = trade.detect_wallet_signature_type("0x2222222222222222222222222222222222222222")
            self.assertEqual(sig, 2)

        with patch.object(
            trade.api,
            "post",
            return_value=_FakeResponse(
                200,
                [
                    {"id": 1, "result": "0x363d3d"},
                    {"id": 2, "error": {"code": 3}},
                    {"id": 3, "result": "0x000000000000000000000000d502cf92620d4822601cf5d36103b1312fc19733"},
                ],
            ),
        ):
            sig, _ = trade.detect_wallet_signature_type("0x3333333333333333333333333333333333333333")
            self.assertEqual(sig, 1)

    def test_get_clob_config_funder_handling(self):
        with patch.object(cfg, "CLOB_PRIVATE_KEY", "abc123"), \
             patch.object(cfg, "CLOB_SIGNATURE_TYPE", 1), \
             patch.object(cfg, "CLOB_FUNDER", ""):
            c = trade.get_clob_config(profile_id="0x4444444444444444444444444444444444444444", detect_remote=False)
            self.assertEqual(c["private_key"], "0xabc123")
            self.assertEqual(c["signature_type"], 1)
            self.assertEqual(c["funder"], "0x4444444444444444444444444444444444444444")

        with patch.object(cfg, "CLOB_PRIVATE_KEY", "0xabc123"), \
             patch.object(cfg, "CLOB_SIGNATURE_TYPE", 0), \
             patch.object(cfg, "CLOB_FUNDER", ""):
            c_eoa = trade.get_clob_config(profile_id="0x4444444444444444444444444444444444444444", detect_remote=False)
            self.assertEqual(c_eoa["signature_type"], 0)
            self.assertIsNone(c_eoa["funder"])


class TestOrderPreviewAndExecution(unittest.TestCase):
    SAMPLE_BOOK = {
        "asks": [{"price": "0.45", "size": "100"}, {"price": "0.52", "size": "100"}],
        "bids": [{"price": "0.43", "size": "80"}, {"price": "0.38", "size": "100"}],
        "tick_size": "0.01",
        "min_order_size": "5",
        "neg_risk": False,
    }

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._orig_dir = cfg.HISTORY_DIR
        self._orig_active = cfg.ACTIVE_PROFILE_FILE
        self._orig_wallet = cfg.POLYGON_WALLET
        cfg.HISTORY_DIR = self.tmp_dir.name
        cfg.ACTIVE_PROFILE_FILE = os.path.join(self.tmp_dir.name, "active_profile.txt")
        cfg.POLYGON_WALLET = ""

    def tearDown(self):
        cfg.HISTORY_DIR = self._orig_dir
        cfg.ACTIVE_PROFILE_FILE = self._orig_active
        cfg.POLYGON_WALLET = self._orig_wallet
        self.tmp_dir.cleanup()

    def test_round_price_to_tick(self):
        self.assertAlmostEqual(trade.round_price_to_tick(0.454, "0.01", mode="up"), 0.46)
        self.assertAlmostEqual(trade.round_price_to_tick(0.459, "0.01", mode="down"), 0.45)
        self.assertAlmostEqual(trade.round_price_to_tick(0.0, "0.001", mode="nearest"), 0.001)
        self.assertAlmostEqual(trade.round_price_to_tick(1.0, "0.001", mode="nearest"), 0.999)

    def test_preview_rejects_paper_profile(self):
        with patch.object(trade.api, "get", return_value=_FakeResponse(200, self.SAMPLE_BOOK)):
            prev_b = trade.preview_buy_order("tok1", 25.0, profile_id="local")
            self.assertFalse(prev_b["ok"])
            self.assertIn("Paper Sandbox", prev_b["reason"])

            prev_s = trade.preview_sell_order("tok1", 20.0, profile_id="sandbox")
            self.assertFalse(prev_s["ok"])
            self.assertIn("Paper Sandbox", prev_s["reason"])

    def test_preview_buy_fak_and_gtc(self):
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        with patch.object(trade.api, "get", return_value=_FakeResponse(200, self.SAMPLE_BOOK)):
            # Belief hurdle 0.50 with 0.02 min_edge -> takes 0.45 level ($45 max), skips 0.52
            prev = trade.preview_buy_order("tok_yes", 60.0, p_norm=0.50, fee_rate=0.0, min_edge=0.02, order_type="FAK", profile_id=wallet)
            self.assertTrue(prev["ok"])
            self.assertEqual(prev["amount_usdc"], 45.0)
            self.assertEqual(prev["price_cap"], 0.45)
            self.assertAlmostEqual(prev["est_shares"], 100.0)

            # $1 Market buy (FAK) succeeds even when est_shares < min_order_size (5 sh)
            high_book = {"asks": [{"price": "0.71", "size": "100"}], "bids": [{"price": "0.70", "size": "100"}], "tick_size": "0.01", "min_order_size": "5"}
            with patch.object(trade.api, "get", return_value=_FakeResponse(200, high_book)):
                prev_one = trade.preview_buy_order("tok_yes", 1.0, order_type="FAK", profile_id=wallet)
                self.assertTrue(prev_one["ok"])
                self.assertAlmostEqual(prev_one["est_shares"], 1.41)

                prev_sub_dollar = trade.preview_buy_order("tok_yes", 0.50, order_type="FAK", profile_id=wallet)
                self.assertFalse(prev_sub_dollar["ok"])
                self.assertIn("$1.00 minimum", prev_sub_dollar["reason"])

            # GTC too small (< 5 shares) fails
            prev_small = trade.preview_buy_order("tok_yes", 1.0, order_type="GTC", limit_price=0.50, profile_id=wallet)
            self.assertFalse(prev_small["ok"])
            self.assertIn("below market limit minimum", prev_small["reason"])

    def test_preview_sell_fak_and_gtc(self):
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        with patch.object(trade.api, "get", return_value=_FakeResponse(200, self.SAMPLE_BOOK)):
            prev = trade.preview_sell_order("tok_yes", 100.0, order_type="FAK", profile_id=wallet)
            self.assertTrue(prev["ok"])
            self.assertEqual(prev["shares"], 100.0)
            self.assertEqual(prev["price_cap"], 0.38)
            self.assertGreater(prev["est_proceeds"], 40.0)

            prev_gtc = trade.preview_sell_order("tok_yes", 50.0, order_type="GTC", limit_price=0.716, profile_id=wallet)
            self.assertTrue(prev_gtc["ok"])
            self.assertEqual(prev_gtc["price_cap"], 0.72)
            self.assertEqual(prev_gtc["est_proceeds"], 36.0)

            # Sell shares floor down so we never exceed held balance (10.129 -> 10.12)
            prev_floor = trade.preview_sell_order("tok_yes", 10.129, order_type="FAK", profile_id=wallet)
            self.assertTrue(prev_floor["ok"])
            self.assertEqual(prev_floor["shares"], 10.12)

            # Small market sell (FAK < 5 sh) succeeds; small resting limit sell (GTC < 5 sh) fails
            prev_small_fak_sell = trade.preview_sell_order("tok_yes", 2.0, order_type="FAK", profile_id=wallet)
            self.assertTrue(prev_small_fak_sell["ok"])
            prev_small_gtc_sell = trade.preview_sell_order("tok_yes", 2.0, order_type="GTC", limit_price=0.50, profile_id=wallet)
            self.assertFalse(prev_small_gtc_sell["ok"])
            self.assertIn("below market limit minimum", prev_small_gtc_sell["reason"])

            # Max single-order cap blocks oversized buys when signing key is configured
            with patch.object(cfg, "CLOB_PRIVATE_KEY", "0xdeadbeef"), \
                 patch.object(cfg, "CLOB_MAX_ORDER_USD", 50.0):
                prev_cap = trade.preview_buy_order("tok_yes", 75.0, order_type="FAK", profile_id=wallet)
                self.assertFalse(prev_cap["ok"])
                self.assertIn("CLOB_MAX_ORDER_USD", prev_cap["reason"])

            # Zero-key browser pre-flight is not blocked by CLOB_MAX_ORDER_USD
            with patch.object(cfg, "CLOB_PRIVATE_KEY", ""), \
                 patch.object(cfg, "CLOB_MAX_ORDER_USD", 50.0):
                prev_zk = trade.preview_buy_order("tok_yes", 75.0, order_type="FAK", profile_id=wallet)
                self.assertTrue(prev_zk["ok"])

    def test_execute_order_fak_and_gtc_calls_clob_client(self):
        fake_client = MagicMock()
        fake_client.create_market_order.return_value = "SIGNED_MKT"
        fake_client.create_order.return_value = "SIGNED_LIM"
        fake_client.post_order.return_value = {"success": True, "orderID": "ord_123", "status": "matched"}

        fak_prev = {
            "ok": True,
            "side": "BUY",
            "order_type": "FAK",
            "token_id": "tok_yes",
            "amount_usdc": 25.0,
            "price_cap": 0.46,
            "tick_size": "0.01",
            "neg_risk": False,
        }
        res_fak = trade.execute_order(fak_prev, client=fake_client)
        self.assertTrue(res_fak["ok"])
        self.assertEqual(res_fak["order_id"], "ord_123")
        fake_client.create_market_order.assert_called_once()
        mkt_args = fake_client.create_market_order.call_args[0][0]
        self.assertEqual(mkt_args.amount, 25.0)
        self.assertEqual(mkt_args.price, 0.46)
        self.assertEqual(str(mkt_args.order_type), "FAK")

        gtc_prev = {
            "ok": True,
            "side": "SELL",
            "order_type": "GTC",
            "token_id": "tok_no",
            "shares": 40.0,
            "price_cap": 0.72,
            "tick_size": "0.01",
            "neg_risk": True,
        }
        res_gtc = trade.execute_order(gtc_prev, client=fake_client)
        self.assertTrue(res_gtc["ok"])
        fake_client.create_order.assert_called_once()
        lim_args = fake_client.create_order.call_args[0][0]
        self.assertEqual(lim_args.size, 40.0)
        self.assertEqual(lim_args.price, 0.72)
        lim_opts = fake_client.create_order.call_args[1]["options"]
        self.assertTrue(lim_opts.neg_risk)

    def test_session_live_buy_candidates_selects_yes_no_token(self):
        import session
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        history = {"profile_id": wallet, "predicted": {}, "exited": {}, "resolved": {}, "skipped": {}}
        candidates = [{
            "market_id": "m1",
            "question": "Will event X happen?",
            "action": "NO",
            "side": "No",
            "alloc": 20.0,
            "target": 0.65,
            "fee_rate": 0.0,
            "clobTokenIds": '["tok_yes", "tok_no"]',
            "payload": {},
        }]
        with patch.object(cfg, "CLOB_PRIVATE_KEY", "0xdeadbeef"), \
             patch.object(cfg, "CLOB_SIGNATURE_TYPE", 1), \
             patch.object(trade.api, "get", return_value=_FakeResponse(200, self.SAMPLE_BOOK)), \
             patch("trade.execute_order", return_value={"ok": True, "order_id": "ord_9", "status": "matched"}) as mock_exec, \
             patch("wallet.import_polymarket_positions", return_value=history), \
             patch("storage.save_history") as mock_save, \
             patch("builtins.input", return_value="CONFIRM"):
            ok = session._execute_live_buy_candidates(candidates, history)
            self.assertTrue(ok)
            mock_exec.assert_called_once()
            self.assertEqual(mock_exec.call_args[0][0]["token_id"], "tok_no")
            self.assertEqual(history["profile_id"], wallet)
            mock_save.assert_called_once()

    def test_views_live_sell_and_buy_row(self):
        import views
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        history = {"profile_id": wallet, "wallet_summary": {"cash": 100.0, "bankroll": 200.0}, "predicted": {}, "exited": {}}
        row = {
            "Mode": "active",
            "Side": "Yes",
            "Market ID": "m1",
            "Question": "Will Y happen?",
            "Shares": 50.0,
            "HeldFrac": 0.1,
            "Kelly": 0.2,
            "Target": 0.70,
            "fee_rate": 0.0,
            "clobTokenIds": '["tok_yes", "tok_no"]',
        }
        with patch.object(cfg, "CLOB_PRIVATE_KEY", "0xdeadbeef"), \
             patch.object(cfg, "CLOB_SIGNATURE_TYPE", 1), \
             patch.object(trade.api, "get", return_value=_FakeResponse(200, self.SAMPLE_BOOK)), \
             patch("trade.execute_order", return_value={"ok": True, "order_id": "ord_sell", "status": "matched"}) as mock_exec, \
             patch("wallet.import_polymarket_positions", return_value=history), \
             patch("storage.save_history") as mock_save, \
             patch("builtins.input", side_effect=["50%", "1", "CONFIRM", ""]):
            ok = views._handle_live_trade_row(history, row, side="SELL")
            self.assertTrue(ok)
            prev = mock_exec.call_args[0][0]
            self.assertEqual(prev["side"], "SELL")
            self.assertEqual(prev["token_id"], "tok_yes")
            self.assertEqual(prev["shares"], 25.0)
            self.assertEqual(history["profile_id"], wallet)
            mock_save.assert_called_once()

    def test_zero_key_browser_preflight_session_and_views(self):
        import session
        import views
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        history = {"profile_id": wallet, "wallet_summary": {"cash": 100.0, "bankroll": 200.0}, "predicted": {}, "exited": {}, "resolved": {}, "skipped": {}}
        candidates = [{
            "market_id": "m1",
            "question": "Will event Z happen?",
            "slug": "event-z",
            "action": "YES",
            "side": "Yes",
            "alloc": 22.5,
            "target": 0.65,
            "fee_rate": 0.0,
            "clobTokenIds": '["tok_yes", "tok_no"]',
            "payload": {},
        }]
        history_sess = copy.deepcopy(history)
        with patch.object(cfg, "CLOB_PRIVATE_KEY", ""), \
             patch.object(trade.api, "get", return_value=_FakeResponse(200, self.SAMPLE_BOOK)), \
             patch("trade.open_polymarket_url", return_value=True) as mock_open, \
             patch("trade.execute_order") as mock_exec, \
             patch("wallet.import_polymarket_positions", return_value=history_sess) as mock_sync, \
             patch("storage.save_history"), \
             patch("builtins.input", side_effect=["o", ""]):
            ok = session._execute_live_buy_candidates(candidates, history_sess)
            self.assertTrue(ok)
            mock_exec.assert_not_called()
            mock_open.assert_called_once_with("https://polymarket.com/event/event-z")
            mock_sync.assert_called_once()
            self.assertEqual(history_sess["profile_id"], wallet)

        row = {
            "Mode": "active",
            "Side": "Yes",
            "Market ID": "m1",
            "Question": "Will event Z happen?",
            "Web Link": "https://polymarket.com/event/event-z",
            "Shares": 30.0,
            "HeldFrac": 0.1,
            "Kelly": 0.0,
            "Target": 0.72,
            "fee_rate": 0.0,
            "clobTokenIds": '["tok_yes", "tok_no"]',
        }
        history_view = copy.deepcopy(history)
        with patch.object(cfg, "CLOB_PRIVATE_KEY", ""), \
             patch.object(trade.api, "get", return_value=_FakeResponse(200, self.SAMPLE_BOOK)), \
             patch("trade.open_polymarket_url", return_value=True) as mock_open, \
             patch("trade.execute_order") as mock_exec, \
             patch("wallet.import_polymarket_positions", return_value=history_view) as mock_sync, \
             patch("storage.save_history") as mock_save, \
             patch("builtins.input", side_effect=["", "2", "72", "o", ""]):
            traded = views._handle_live_trade_row(history_view, row, side="SELL")
            self.assertTrue(traded)
            mock_exec.assert_not_called()
            mock_open.assert_called_once_with("https://polymarket.com/event/event-z")
            mock_sync.assert_called_once()
            self.assertEqual(history_view["profile_id"], wallet)
            mock_save.assert_called_once_with(history_view, profile_id=wallet)

    def test_sync_wallet_after_trade_preserves_in_place_dict_and_isolates_dir(self):
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        history = {"profile_id": wallet, "predicted": {"m1": [{"pu": 0.7}]}}
        with patch("wallet.import_polymarket_positions", return_value=history):
            ok = trade.sync_wallet_after_trade(wallet, history)
        self.assertTrue(ok)
        self.assertEqual(history["profile_id"], wallet)
        self.assertIn("m1", history["predicted"])
        expected_path = os.path.join(self.tmp_dir.name, f"history_{wallet}.json")
        self.assertTrue(os.path.exists(expected_path))

    def test_format_browser_ticket(self):
        t_buy = trade.format_browser_ticket(
            {"side": "BUY", "order_type": "FAK", "amount_usdc": 20.0, "est_shares": 44.44, "est_vwap": 0.45, "price_cap": 0.45},
            question="Q1",
            outcome="Yes",
            event_url="https://polymarket.com/event/q1",
        )
        self.assertIn("BUY YES  (Market Order)", t_buy)
        self.assertIn("$20.00 USDC", t_buy)
        self.assertIn("https://polymarket.com/event/q1", t_buy)

        t_sell = trade.format_browser_ticket(
            {"side": "SELL", "order_type": "GTC", "shares": 30.0, "est_proceeds": 21.6, "est_vwap": 0.72, "price_cap": 0.72},
            question="Q1",
            outcome="No",
        )
        self.assertIn("SELL NO  (Limit Order)", t_sell)
        self.assertIn("30.00 shares", t_sell)
        self.assertIn("72.0c", t_sell)


class TestMetaMaskBridgeAndBrowser(unittest.TestCase):
    TEST_PRIV_KEY = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"

    def setUp(self):
        metamask_signer._SESSION_L2_CREDS.clear()

    def _sample_preview_and_conf(self, signer_addr):
        preview = {
            "ok": True,
            "side": "BUY",
            "token_id": "10987654321098765432109876543210987654321098765432109876543210987654321098765",
            "amount_usdc": 25.0,
            "est_shares": 50.0,
            "est_vwap": 0.50,
            "price_cap": 0.50,
            "tick_size": "0.01",
            "neg_risk": False,
            "order_type": "FAK",
        }
        conf = {
            "host": "https://clob.polymarket.com",
            "chain_id": 137,
            "funder": signer_addr,
            "signature_type": 0,
            "has_private_key": False,
            "execution_mode": "metamask",
            "metamask_signer": signer_addr,
            "signing_browser": "default",
        }
        return preview, conf

    def _http_post_json(self, base_url, path, payload):
        req = Request(
            f"{base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_metamask_eip712_order_matches_py_order_utils(self):
        from eth_account import Account
        from eth_account.messages import encode_typed_data
        from py_order_utils.builders import OrderBuilder
        from py_order_utils.model import BUY as UTILS_BUY, OrderData
        from py_order_utils.signer import Signer as UtilsSigner
        from py_clob_client.config import get_contract_config

        acct = Account.from_key(self.TEST_PRIV_KEY)
        preview, conf = self._sample_preview_and_conf(acct.address)

        with patch.object(metamask_signer.api, "get", return_value=_FakeResponse(200, {"base_fee": 0})):
            typed_data, order_dict = metamask_signer.build_order_typed_data(preview, conf, acct.address)

        # Signer must be verifiable against the exact MetaMask eth_signTypedData_v4 JSON payload
        signed_mm = Account.sign_typed_data(self.TEST_PRIV_KEY, full_message=typed_data)
        signable = encode_typed_data(full_message=typed_data)
        self.assertEqual(Account.recover_message(signable, signature=signed_mm.signature), acct.address)

        # Sign the exact same OrderData fields using py_order_utils.OrderBuilder and verify identical signature
        contracts = get_contract_config(137, neg_risk=False)
        builder = OrderBuilder(
            contracts.exchange,
            137,
            UtilsSigner(key=self.TEST_PRIV_KEY),
            salt_generator=lambda: int(order_dict["salt"]),
        )
        data = OrderData(
            maker=order_dict["maker"],
            taker=order_dict["taker"],
            tokenId=str(order_dict["tokenId"]),
            makerAmount=str(order_dict["makerAmount"]),
            takerAmount=str(order_dict["takerAmount"]),
            side=UTILS_BUY,
            feeRateBps=str(order_dict["feeRateBps"]),
            nonce=str(order_dict["nonce"]),
            signer=order_dict["signer"],
            expiration=str(order_dict["expiration"]),
            signatureType=order_dict["signatureType"],
        )
        signed_utils = builder.build_signed_order(data)

        sig_mm_hex = signed_mm.signature.hex()
        if not sig_mm_hex.startswith("0x"):
            sig_mm_hex = "0x" + sig_mm_hex
        self.assertEqual(sig_mm_hex, signed_utils.signature)

    def test_metamask_bridge_end_to_end_l1_auth_and_l2_order(self):
        from eth_account import Account

        acct = Account.from_key(self.TEST_PRIV_KEY)
        preview, conf = self._sample_preview_and_conf(acct.address)
        server, url, done_event, state = metamask_signer.create_metamask_bridge_server(
            preview, conf, summary={"question": "Test market?", "outcome": "Yes"}
        )
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        try:
            parsed = urlparse(url)
            base_url = f"http://127.0.0.1:{parsed.port}"
            token = parse_qs(parsed.query)["token"][0]

            with self.assertRaises(HTTPError) as ctx_404:
                urlopen(f"{base_url}/", timeout=5)
            self.assertEqual(ctx_404.exception.code, 404)
            ctx_404.exception.close()

            with urlopen(url, timeout=5) as r_html:
                self.assertEqual(r_html.status, 200)
                self.assertIn(b"Polymarket MetaMask Signer", r_html.read())

            with self.assertRaises(HTTPError) as ctx_403:
                self._http_post_json(base_url, "/prepare-auth", {"token": "wrong", "signer": acct.address})
            self.assertEqual(ctx_403.exception.code, 403)
            ctx_403.exception.close()

            api_secret = base64.b64encode(b"s" * 32).decode("ascii")
            captured_post = {}

            def _fake_get(req_url, **kwargs):
                if "/fee-rate" in req_url:
                    return _FakeResponse(200, {"base_fee": 0})
                if "/auth/derive-api-key" in req_url:
                    return _FakeResponse(200, {"apiKey": "k-test", "secret": api_secret, "passphrase": "p-test"})
                return _FakeResponse(404, {})

            def _fake_post(req_url, headers=None, data=None, **kwargs):
                captured_post["url"] = req_url
                captured_post["headers"] = dict(headers or {})
                captured_post["body"] = json.loads(data)
                return _FakeResponse(200, {"success": True, "orderID": "mm-ord-1", "status": "matched"})

            def _fake_request(method, req_url, **kwargs):
                return _fake_get(req_url, **kwargs)

            with patch.object(metamask_signer.api, "get", side_effect=_fake_get), \
                 patch.object(metamask_signer.api, "request", side_effect=_fake_request), \
                 patch.object(metamask_signer.api, "post", side_effect=_fake_post):
                st, auth_prep = self._http_post_json(base_url, "/prepare-auth", {"token": token, "signer": acct.address})
                self.assertEqual(st, 200)
                self.assertTrue(auth_prep["need_auth"])

                auth_sig = Account.sign_typed_data(self.TEST_PRIV_KEY, full_message=auth_prep["typed_data"]).signature.hex()
                if not auth_sig.startswith("0x"):
                    auth_sig = "0x" + auth_sig
                st, auth_sub = self._http_post_json(
                    base_url,
                    "/submit-auth",
                    {
                        "token": token,
                        "signer": acct.address,
                        "signature": auth_sig,
                        "timestamp": auth_prep["timestamp"],
                        "nonce": auth_prep["nonce"],
                    },
                )
                self.assertEqual(st, 200)
                self.assertTrue(auth_sub["ok"])

                st, ord_prep = self._http_post_json(base_url, "/prepare-order", {"token": token, "signer": acct.address})
                self.assertEqual(st, 200)
                ord_sig = Account.sign_typed_data(self.TEST_PRIV_KEY, full_message=ord_prep["typed_data"]).signature.hex()
                if not ord_sig.startswith("0x"):
                    ord_sig = "0x" + ord_sig

                st, ord_sub = self._http_post_json(
                    base_url,
                    "/submit-order",
                    {"token": token, "signer": acct.address, "signature": ord_sig, "order_dict": ord_prep["order_dict"]},
                )
                self.assertEqual(st, 200)
                self.assertTrue(ord_sub["ok"])
                self.assertEqual(ord_sub["order_id"], "mm-ord-1")

            self.assertTrue(done_event.is_set())
            self.assertTrue(state["result"]["ok"])
            self.assertIn("POLY_API_KEY", captured_post["headers"])
            self.assertIn("POLY_SIGNATURE", captured_post["headers"])
            self.assertEqual(captured_post["body"]["order"]["signature"], ord_sig)
        finally:
            server.shutdown()
            server.server_close()

        # Verify /cancel closes flow cleanly too
        server2, url2, done2, state2 = metamask_signer.create_metamask_bridge_server(preview, conf)
        t2 = threading.Thread(target=server2.serve_forever, daemon=True)
        t2.start()
        try:
            p2 = urlparse(url2)
            base2 = f"http://127.0.0.1:{p2.port}"
            tok2 = parse_qs(p2.query)["token"][0]
            st, res_c = self._http_post_json(base2, "/cancel", {"token": tok2})
            self.assertEqual(st, 200)
            self.assertTrue(res_c["ok"])
            self.assertTrue(done2.is_set())
            self.assertEqual(state2["result"]["status"], "cancelled")
        finally:
            server2.shutdown()
            server2.server_close()

    def test_execute_order_routes_to_metamask_mode(self):
        preview, conf = self._sample_preview_and_conf("0x1111111111111111111111111111111111111111")
        with patch("trade.get_clob_config", return_value=conf), \
             patch("metamask_signer.sign_and_submit_with_metamask", return_value={"ok": True, "order_id": "mm-99", "status": "live"}) as mock_mm:
            res = trade.execute_order(preview, profile_id=conf["funder"], summary={"question": "Q", "outcome": "Yes"})
        self.assertTrue(res["ok"])
        self.assertEqual(res["order_id"], "mm-99")
        mock_mm.assert_called_once()

    def test_open_polymarket_url_resolves_signing_browser_and_local_bridge(self):
        self.assertFalse(trade.open_polymarket_url("http://example.com/phish"))
        with patch("trade.os.path.exists", side_effect=lambda p: p.endswith("zen.exe")), \
             patch("trade.subprocess.Popen") as mock_popen:
            ok = trade.open_polymarket_url("http://127.0.0.1:54321/?token=abc", browser_pref="zen")
        self.assertTrue(ok)
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertTrue(args[0].endswith("zen.exe"))
        self.assertEqual(args[1], "http://127.0.0.1:54321/?token=abc")

    def test_configure_trade_execution_saves_metamask_settings(self):
        import manage
        import storage

        wallet = "0x1111111111111111111111111111111111111111"
        signer = "0x2222222222222222222222222222222222222222"
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cfg, "HISTORY_DIR", tmp), \
                 patch.object(cfg, "ACTIVE_PROFILE_FILE", os.path.join(tmp, "active.txt")), \
                 patch.object(cfg, "EXECUTION_MODE", "browser"), \
                 patch.object(cfg, "METAMASK_SIGNER", ""), \
                 patch.object(cfg, "SIGNING_BROWSER", "default"), \
                 patch.object(cfg, "CLOB_SIGNATURE_TYPE", None), \
                 patch.dict(os.environ, {}, clear=False):
                for k in ("EXECUTION_MODE", "METAMASK_SIGNER", "SIGNING_BROWSER", "CLOB_SIGNATURE_TYPE"):
                    os.environ.pop(k, None)
                history = {"profile_id": wallet, "settings": {}}
                storage.save_history(history, profile_id=wallet)
                inputs = ["1", "metamask", "2", signer, "3", "zen", "4", "2", ""]
                with patch("trade.detect_wallet_signature_type", return_value=(2, "detected")), \
                     patch("builtins.input", side_effect=inputs):
                    manage._configure_trade_execution(history, wallet)
                self.assertEqual(history["settings"]["EXECUTION_MODE"], "metamask")
                self.assertEqual(history["settings"]["METAMASK_SIGNER"], signer)
                self.assertEqual(history["settings"]["SIGNING_BROWSER"], "zen")
                self.assertEqual(history["settings"]["CLOB_SIGNATURE_TYPE"], 2)

                loaded = storage.load_history(wallet)
                self.assertEqual(cfg.EXECUTION_MODE, "metamask")
                self.assertEqual(cfg.METAMASK_SIGNER, signer)
                self.assertEqual(cfg.SIGNING_BROWSER, "zen")
                self.assertEqual(cfg.CLOB_SIGNATURE_TYPE, 2)
                self.assertEqual(loaded["settings"]["SIGNING_BROWSER"], "zen")


if __name__ == "__main__":
    unittest.main()
