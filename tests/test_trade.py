"""Unit tests for live CLOB order preview, signature detection, and execution."""
import copy
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import trade


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

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

            # GTC too small (< 5 shares) fails
            prev_small = trade.preview_buy_order("tok_yes", 1.0, order_type="GTC", limit_price=0.50, profile_id=wallet)
            self.assertFalse(prev_small["ok"])
            self.assertIn("below market minimum", prev_small["reason"])

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


if __name__ == "__main__":
    unittest.main()
