"""Unit and integration tests for wallet.py reconciliation, trade delta replay, and resolution."""
import unittest
from unittest.mock import MagicMock, patch

import config as cfg
import wallet


class TestWalletPureHelpers(unittest.TestCase):
    def test_normalize_closed_positions_dedups_and_adjusts_loss_basis(self):
        open_pos = [{"conditionId": "0xAAA", "outcome": "Yes", "size": 10.0}]
        closed_pos = [
            {"conditionId": "0xaaa", "outcome": "Yes", "totalBought": 10.0, "avgPrice": 0.5, "curPrice": 0.0, "realizedPnl": -5.0},
            {"conditionId": "0xBBB", "outcome": "No", "totalBought": 20.0, "avgPrice": 0.4, "curPrice": 0.0, "realizedPnl": -6.5},
            {"conditionId": "0xCCC", "outcome": "Yes", "totalBought": 20.0, "avgPrice": 0.4, "curPrice": 0.99, "realizedPnl": 12.0},
        ]
        merged = wallet.normalize_closed_positions(open_pos, closed_pos)
        self.assertEqual(len(merged), 3)  # 0xAAA closed deduped
        self.assertEqual(merged[1]["conditionId"], "0xBBB")
        self.assertEqual(merged[1]["initialValue"], 6.5)  # abs(realizedPnl) on settled loss
        self.assertEqual(merged[1]["size"], 20.0)
        self.assertEqual(merged[2]["initialValue"], 8.0)  # gross_val = 20 * 0.4

    def test_reconcile_resolved_match_fills_missing_pnl_only(self):
        entries = [
            ("m1", {"outcome": 1.0, "shares": 100.0, "cost_basis": 40.0, "realized_pnl": None}),
            ("m1", {"outcome": 0.0, "shares": 50.0, "cost_basis": 25.0, "realized_pnl": -99.0}),  # already has PnL
            ("m1", {"outcome": 1.0, "is_watch": True, "shares": 10.0, "cost_basis": 5.0, "realized_pnl": None}),
        ]
        count = wallet.reconcile_resolved_match(entries, pos_realized_pnl=55.0, bankroll=100.0)
        self.assertEqual(count, 3)
        self.assertEqual(entries[0][1]["realized_pnl"], 60.0)
        self.assertEqual(entries[1][1]["realized_pnl"], -99.0)
        self.assertIsNone(entries[2][1]["realized_pnl"])

    def test_reconcile_exited_match_updates_exit_fields(self):
        entry = {"pm": 0.5, "snapshots": [{"pm": 0.5, "entry_price": 0.5}]}
        pos = {"totalBought": 100.0, "last_event_at": 1700000000}
        count = wallet.reconcile_exited_match(
            [("m1", entry)], pos, avg_price=0.4, cur_price=0.65, size=100.0, realized_pnl=20.0, total_bankroll=500.0
        )
        self.assertEqual(count, 1)
        self.assertEqual(entry["entry_cost"], 40.0)
        self.assertEqual(entry["exit_price"], 0.6)
        self.assertEqual(entry["realized_pnl"], 20.0)
        self.assertEqual(entry["pm"], 0.4)
        self.assertEqual(entry["snapshots"][0]["entry_price"], 0.4)

    def test_compute_tranche_delta_with_trade_replay_and_sells(self):
        wallet_preds = [{"cost_basis": 40.0, "tranche_pm": 0.4, "last_synced_ts": 100}]
        trades = [
            {"ts": 100, "side": "BUY", "usdc": 40.0, "shares": 100.0, "price": 0.4},
            {"ts": 200, "side": "BUY", "usdc": 30.0, "shares": 50.0, "price": 0.6},
            # Partial sell of 75 shares out of (100 + 50) = 150 -> keep_ratio = 0.5 -> new_buys scaled by 0.5
            {"ts": 300, "side": "SELL", "usdc": 45.0, "shares": 75.0, "price": 0.6},
        ]
        delta_usdc, delta_pm = wallet.compute_tranche_delta(
            wallet_preds, trades, initial_val=35.0, size=75.0, avg_price=0.4667, prev_bankroll=100.0, latest_trade_ts=300
        )
        self.assertAlmostEqual(delta_usdc, 15.0)
        self.assertAlmostEqual(delta_pm, 0.6)

    def test_compute_tranche_delta_without_trades_fallback(self):
        wallet_preds = [{"cost_basis": 40.0, "tranche_pm": 0.4, "last_synced_ts": 0}]
        # prev_shares = 40 / 0.4 = 100; new size = 150, initial_val = 70 -> delta = $30 for 50 shares -> 0.60
        delta_usdc, delta_pm = wallet.compute_tranche_delta(
            wallet_preds, [], initial_val=70.0, size=150.0, avg_price=0.5, prev_bankroll=100.0, latest_trade_ts=0
        )
        self.assertAlmostEqual(delta_usdc, 30.0)
        self.assertAlmostEqual(delta_pm, 0.6)


class TestResolveMarketEntries(unittest.TestCase):
    def test_resolve_predicted_and_exited_entries(self):
        history = {
            "profile_id": "local",
            "wallet_summary": {"bankroll": 200.0},
            "predicted": {
                "101": [
                    {"question": "Q1", "position": "Yes", "shares": 50.0, "cost_basis": 20.0},
                    {"question": "Q1", "position": "No", "entry_kelly": 0.1, "pm": 0.4},
                    {"question": "Q1", "position": "Yes", "is_watch": True, "shares": 5.0, "cost_basis": 2.0},
                ]
            },
            "exited": {
                "101": [
                    {"question": "Q1", "position": "Yes", "exited": True, "realized_pnl": 7.5, "cost_basis": 0.0, "entry_kelly": 0.2},
                ]
            },
            "resolved": {},
        }
        market_data = {"closed": True, "outcomePrices": '["1", "0"]'}
        did_resolve = wallet.resolve_market_entries(history, "101", market_data, bankroll=200.0)
        self.assertTrue(did_resolve)
        self.assertNotIn("101", history["predicted"])
        self.assertNotIn("101", history["exited"])

        res = history["resolved"]["101"]
        self.assertEqual(len(res), 4)
        # Active YES win: 50 * 1.0 - 20.0 = 30.0
        self.assertEqual(res[0]["outcome"], 1.0)
        self.assertEqual(res[0]["realized_pnl"], 30.0)
        # Active NO loss via kelly fallback: cb = 0.1 * 200 = 20 -> -20.0
        self.assertEqual(res[1]["outcome"], 0.0)
        self.assertEqual(res[1]["realized_pnl"], -20.0)
        # Watch-only -> None
        self.assertIsNone(res[2]["realized_pnl"])
        # Exited -> preserves 7.5
        self.assertEqual(res[3]["realized_pnl"], 7.5)

    def test_resolve_wallet_profile_unexecuted_local_gets_none(self):
        history = {
            "profile_id": "0x1234",
            "wallet_address": "0x1234",
            "predicted": {
                "202": [{"position": "Yes", "source": "local", "kelly": 0.1, "cost_basis": 0.0, "shares": 0.0}]
            },
            "exited": {},
            "resolved": {},
        }
        did_resolve = wallet.resolve_market_entries(history, "202", {"closed": True, "outcomePrices": '["1", "0"]'}, bankroll=100.0)
        self.assertTrue(did_resolve)
        self.assertIsNone(history["resolved"]["202"][0]["realized_pnl"])

    def test_open_market_not_resolved(self):
        history = {"predicted": {"303": [{"position": "Yes"}]}, "exited": {}, "resolved": {}}
        self.assertFalse(wallet.resolve_market_entries(history, "303", {"closed": False, "umaResolutionStatus": "pending"}))
        self.assertIn("303", history["predicted"])


class TestImportPolymarketPositionsEndToEnd(unittest.TestCase):
    def setUp(self):
        self._orig_bankroll = cfg.BANKROLL

    def tearDown(self):
        cfg.BANKROLL = self._orig_bankroll

    @patch("builtins.input")
    @patch("wallet.resolve_position_market_metadata")
    @patch("wallet.get_wallet_portfolio_value")
    @patch("wallet.get_wallet_usdc_balance")
    @patch("wallet.fetch_wallet_trades")
    @patch("wallet.fetch_v2_positions")
    def test_sync_open_revive_archive_and_import_with_rollback(
        self,
        mock_fetch_pos,
        mock_fetch_trades,
        mock_cash,
        mock_port_val,
        mock_meta,
        mock_input,
    ):
        open_positions = [
            # 1. Existing active position already in predicted (absorbs tranche)
            {
                "conditionId": "0xOPEN1",
                "outcome": "Yes",
                "outcomeIndex": 0,
                "avgPrice": 0.50,
                "curPrice": 0.60,
                "size": 100.0,
                "totalBought": 100.0,
                "initialValue": 50.0,
                "currentValue": 60.0,
                "realizedPnl": 0.0,
                "redeemable": False,
                "title": "Existing Market",
                "slug": "existing-market",
                "endDate": "2030-01-01T00:00:00Z",
                "timestamp": 200,
                "_from_closed": False,
            },
            # 2. Previously exited position reopened on-chain -> should revive
            {
                "conditionId": "0xREVIVE",
                "outcome": "Yes",
                "outcomeIndex": 0,
                "avgPrice": 0.40,
                "curPrice": 0.45,
                "size": 50.0,
                "totalBought": 50.0,
                "initialValue": 20.0,
                "currentValue": 22.5,
                "realizedPnl": 0.0,
                "redeemable": False,
                "title": "Revive Market",
                "slug": "revive-market",
                "endDate": "2030-01-01T00:00:00Z",
                "timestamp": 250,
                "_from_closed": False,
            },
            # 3. Brand new position 1 + 4. Brand new position 2 (test 'b' rollback + skip)
            {
                "conditionId": "0xNEW1",
                "outcome": "Yes",
                "outcomeIndex": 0,
                "avgPrice": 0.50,
                "curPrice": 0.55,
                "size": 40.0,
                "totalBought": 40.0,
                "initialValue": 20.0,
                "currentValue": 22.0,
                "realizedPnl": 0.0,
                "redeemable": False,
                "title": "New Market 1",
                "slug": "new-market-1",
                "endDate": "2030-01-01T00:00:00Z",
                "timestamp": 300,
                "_from_closed": False,
            },
        ]
        closed_positions = [
            # Closed position missing from open_positions that archives '0xVANISHED'
            {
                "conditionId": "0xVANISHED",
                "outcome": "Yes",
                "outcomeIndex": 0,
                "avgPrice": 0.40,
                "curPrice": 0.60,
                "size": 100.0,
                "totalBought": 100.0,
                "initialValue": 40.0,
                "currentValue": 0.0,
                "realizedPnl": 20.0,
                "redeemable": False,
                "title": "Vanished Market",
                "slug": "vanished-market",
                "timestamp": 400,
                "last_event_at": 400,
                "_from_closed": True,
            }
        ]
        mock_fetch_pos.side_effect = lambda addr, status: open_positions if status == "OPEN" else closed_positions
        mock_fetch_trades.return_value = {
            "0xopen1": [{"ts": 200, "side": "BUY", "usdc": 20.0, "shares": 40.0, "price": 0.50}],
            "0xvanished": [{"ts": 400, "side": "SELL", "usdc": 60.0, "shares": 100.0, "price": 0.60}],
        }
        mock_cash.return_value = 100.0
        mock_port_val.return_value = 104.5
        mock_meta.side_effect = lambda pos, cache: (
            {"id": pos["conditionId"], "closed": False, "endDate": "2030-01-01T00:00:00Z", "spread": 0.01},
            pos["conditionId"],
            pos["title"],
            pos["slug"],
            "2030-01-01T00:00:00Z",
        )
        # User enters 70 for 0xNEW1
        mock_input.side_effect = ["70"]

        history = {
            "wallet_summary": {"bankroll": 150.0, "cash": 50.0, "portfolio": 100.0},
            "predicted": {
                "m_open": [{
                    "question": "Existing Market",
                    "slug": "existing-market",
                    "conditionId": "0xOPEN1",
                    "source": "wallet",
                    "pu": 0.70,
                    "pm": 0.50,
                    "cost_basis": 30.0,
                    "kelly": 0.1,
                    "last_synced_ts": 100,
                    "position": "Yes",
                }],
                "m_vanished": [{
                    "question": "Vanished Market",
                    "slug": "vanished-market",
                    "conditionId": "0xVANISHED",
                    "source": "wallet",
                    "pu": 0.60,
                    "pm": 0.40,
                    "cost_basis": 40.0,
                    "position": "Yes",
                }],
            },
            "exited": {
                "m_revive": [{
                    "question": "Revive Market",
                    "slug": "revive-market",
                    "conditionId": "0xREVIVE",
                    "source": "wallet",
                    "exited": True,
                    "cost_basis": 0.0,
                    "position": "Yes",
                }],
            },
            "resolved": {},
            "skipped": {},
        }

        out = wallet.import_polymarket_positions("0x1234567890", history, auto_sync=False)

        # Bankroll updated
        self.assertAlmostEqual(out["wallet_summary"]["bankroll"], 204.5)
        # Existing active absorbed tranche (+$20 -> $50 cost basis)
        self.assertEqual(out["predicted"]["m_open"][0]["cost_basis"], 50.0)
        self.assertEqual(out["predicted"]["m_open"][0]["shares"], 100.0)
        # Revived position moved back to predicted
        self.assertNotIn("m_revive", out["exited"])
        self.assertIn("m_revive", out["predicted"])
        self.assertFalse(out["predicted"]["m_revive"][0]["exited"])
        # Vanished position moved to exited with closed position exit price 0.60
        self.assertNotIn("m_vanished", out["predicted"])
        self.assertIn("m_vanished", out["exited"])
        self.assertEqual(out["exited"]["m_vanished"][0]["exit_price"], 0.6)
        self.assertEqual(out["exited"]["m_vanished"][0]["realized_pnl"], 20.0)
        # Newly imported position stored with pu=0.70
        self.assertIn("0xNEW1", out["predicted"])
        self.assertEqual(out["predicted"]["0xNEW1"][0]["pu"], 0.7)

    @patch("builtins.input")
    @patch("wallet.resolve_position_market_metadata")
    @patch("wallet.get_wallet_portfolio_value", return_value=40.0)
    @patch("wallet.get_wallet_usdc_balance", return_value=100.0)
    @patch("wallet.fetch_wallet_trades", return_value={})
    @patch("wallet.fetch_v2_positions")
    def test_import_rollback_with_b_command(
        self, mock_fetch_pos, mock_trades, mock_cash, mock_pv, mock_meta, mock_input
    ):
        pos1 = {
            "conditionId": "0xP1", "outcome": "Yes", "outcomeIndex": 0, "avgPrice": 0.5, "curPrice": 0.5,
            "size": 20.0, "totalBought": 20.0, "initialValue": 10.0, "currentValue": 10.0,
            "realizedPnl": 0.0, "redeemable": False, "title": "P1", "slug": "p1", "_from_closed": False,
        }
        pos2 = {
            "conditionId": "0xP2", "outcome": "Yes", "outcomeIndex": 0, "avgPrice": 0.5, "curPrice": 0.5,
            "size": 20.0, "totalBought": 20.0, "initialValue": 10.0, "currentValue": 10.0,
            "realizedPnl": 0.0, "redeemable": False, "title": "P2", "slug": "p2", "_from_closed": False,
        }
        mock_fetch_pos.side_effect = lambda addr, status: [pos1, pos2] if status == "OPEN" else []
        mock_meta.side_effect = lambda pos, cache: (
            {"id": pos["conditionId"], "closed": False, "spread": 0.01},
            pos["conditionId"], pos["title"], pos["slug"], "2030-01-01T00:00:00Z"
        )
        # Enter 60 for P1, then 'b' at P2 to roll back P1, then 75 for P1, then 'a' to skip remaining
        mock_input.side_effect = ["60", "b", "75", "a"]

        history = {"predicted": {}, "exited": {}, "resolved": {}, "skipped": {}}
        out = wallet.import_polymarket_positions("0x1234", history, auto_sync=False)
        self.assertEqual(len(out["predicted"]["0xP1"]), 1)
        self.assertEqual(out["predicted"]["0xP1"][0]["pu"], 0.75)
        self.assertFalse(out["predicted"]["0xP1"][0].get("unforecasted", False))
        self.assertEqual(out["predicted"]["0xP2"][0]["pu"], 0.5)
        self.assertTrue(out["predicted"]["0xP2"][0].get("unforecasted"))
        self.assertTrue(out["predicted"]["0xP2"][0].get("exclude_from_ego"))

    @patch.object(wallet, "api")
    def test_fetch_v2_endpoints_handle_null_pagination(self, mock_api):
        pos_resp = MagicMock()
        pos_resp.status_code = 200
        pos_resp.json.return_value = {
            "data": [{"conditionId": "0xC1", "outcome": "Yes", "size": 5, "avgPrice": 0.5}],
            "pagination": None,
        }
        trade_resp = MagicMock()
        trade_resp.status_code = 200
        trade_resp.json.return_value = {
            "data": [{"conditionId": "0xC1", "side": "BUY", "size": 5, "price": 0.5, "timestamp": 1000}],
            "pagination": None,
        }
        mock_api.get.side_effect = [pos_resp, trade_resp]

        positions = wallet.fetch_v2_positions("0x1234", "OPEN")
        trades = wallet.fetch_wallet_trades("0x1234")
        self.assertEqual(len(positions), 1)
        self.assertIn("0xc1", trades)


if __name__ == "__main__":
    unittest.main()

