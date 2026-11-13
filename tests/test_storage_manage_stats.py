"""Tests for storage.py profile recovery/isolation, manage.py manual_resolve, and views.py stats."""
import glob
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import config as cfg
import manage
import storage
import views


class TestStorageProfileAndCorruptionRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._orig_dir = cfg.HISTORY_DIR
        self._orig_active = cfg.ACTIVE_PROFILE_FILE
        self._orig_wallet = cfg.POLYGON_WALLET
        self._orig_bankroll = cfg.BANKROLL
        cfg.HISTORY_DIR = self.tmp_dir.name
        cfg.ACTIVE_PROFILE_FILE = os.path.join(self.tmp_dir.name, "active_profile.txt")
        cfg.POLYGON_WALLET = ""
        cfg.BANKROLL = 100.0

    def tearDown(self):
        cfg.HISTORY_DIR = self._orig_dir
        cfg.ACTIVE_PROFILE_FILE = self._orig_active
        cfg.POLYGON_WALLET = self._orig_wallet
        cfg.BANKROLL = self._orig_bankroll
        self.tmp_dir.cleanup()

    def test_save_load_and_list_profiles(self):
        storage.set_active_profile("paper_test")
        data = {
            "predicted": {"m1": [{"pu": 0.7, "pm": 0.5, "kelly": 0.1}]},
            "exited": {},
            "resolved": {},
            "skipped": {},
            "wallet_summary": {"cash": 250.0, "portfolio": 0.0, "bankroll": 250.0},
        }
        storage.save_history(data, profile_id="paper_test")
        loaded = storage.load_history("paper_test")
        self.assertEqual(cfg.BANKROLL, 250.0)
        self.assertEqual(loaded["predicted"]["m1"][0]["pu"], 0.7)

        profiles = storage.list_profiles()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["profile_id"], "paper_test")
        self.assertTrue(profiles[0]["is_active"])
        self.assertEqual(profiles[0]["active_count"], 1)

    def test_corrupt_json_creates_backup_before_recovery(self):
        storage.set_active_profile("broken_prof")
        fpath = storage.get_profile_path("broken_prof")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("{broken json!!!")

        recovered = storage.load_history("broken_prof")
        self.assertEqual(recovered["predicted"], {})
        backups = glob.glob(fpath + ".corrupt.*")
        self.assertEqual(len(backups), 1)
        with open(backups[0], "r", encoding="utf-8") as bf:
            self.assertEqual(bf.read(), "{broken json!!!")


class TestManageManualResolve(unittest.TestCase):
    @patch("manage.save_history")
    @patch("builtins.input", side_effect=["2", "m_active", "1"])
    def test_manual_resolve_active_and_kelly_fallback(self, mock_input, mock_save):
        history = {
            "wallet_summary": {"bankroll": 200.0},
            "predicted": {
                "m_active": [
                    {"question": "Q", "position": "Yes", "entry_kelly": 0.1, "pm": 0.5}
                ]
            },
            "exited": {},
            "resolved": {},
        }
        out = manage.manual_resolve(history)
        self.assertNotIn("m_active", out["predicted"])
        self.assertIn("m_active", out["resolved"])
        # cb = 0.1 * 200 = 20, shares = 20 / 0.5 = 40, win = 40 - 20 = 20.0
        self.assertEqual(out["resolved"]["m_active"][0]["realized_pnl"], 20.0)
        mock_save.assert_called_once()

    @patch("manage.save_history")
    @patch("builtins.input", side_effect=["2", "m_exit", "2"])
    def test_manual_resolve_preserves_existing_exited_pnl(self, mock_input, mock_save):
        history = {
            "wallet_summary": {"bankroll": 200.0},
            "predicted": {},
            "exited": {
                "m_exit": [
                    {
                        "question": "Exited Q",
                        "position": "Yes",
                        "exited": True,
                        "realized_pnl": 14.5,
                        "cost_basis": 0.0,
                        "entry_kelly": 0.1,
                        "pm": 0.5,
                    }
                ]
            },
            "resolved": {},
        }
        out = manage.manual_resolve(history)
        self.assertNotIn("m_exit", out["exited"])
        self.assertEqual(out["resolved"]["m_exit"][0]["outcome"], 0.0)
        self.assertEqual(out["resolved"]["m_exit"][0]["realized_pnl"], 14.5)


class TestViewsPredictionStats(unittest.TestCase):
    def test_calculate_prediction_stats_held_buy_watch_counts(self):
        history = {
            "profile_id": "local",
            "wallet_summary": {"bankroll": 1000.0, "cash": 800.0, "portfolio": 200.0},
            "predicted": {
                # Held position (shares > 0, cost_basis > 0)
                "m1": [{
                    "question": "Held Q", "pu": 0.70, "pm": 0.50, "cur_pm": 0.50,
                    "shares": 400.0, "cost_basis": 200.0, "kelly": 0.1, "source": "local",
                }],
                # Buy signal (unheld, strong edge)
                "m2": [{
                    "question": "Buy Q", "pu": 0.80, "pm": 0.50, "cur_pm": 0.50,
                    "shares": 0.0, "cost_basis": 0.0, "kelly": 0.1, "source": "local",
                }],
                # Watch signal (no edge)
                "m3": [{
                    "question": "Watch Q", "pu": 0.50, "pm": 0.50, "cur_pm": 0.50,
                    "shares": 0.0, "cost_basis": 0.0, "kelly": 0.0, "source": "local",
                }],
            },
            "exited": {},
            "resolved": {},
        }
        stats = views.calculate_prediction_stats(history, sub_mode="predicted", base_ego=0.5, refresh_meta=False)
        self.assertEqual(stats["total_count"], 3)
        self.assertEqual(stats["held_count"], 1)
        self.assertEqual(stats["buy_count"], 1)
        self.assertEqual(stats["watch_count"], 1)
        self.assertAlmostEqual(stats["total_held_val"], 200.0)
        self.assertAlmostEqual(stats["effective_n_held"], 1.0)
        self.assertAlmostEqual(stats["held_bal_pct"], 100.0)

    def test_merge_market_tranches_does_not_mutate_source_records(self):
        t1 = {"source": "wallet", "shares": 10.0, "cost_basis": 5.0, "pu": 0.6, "date": "2026-11-01"}
        t2 = {"source": "wallet", "shares": 20.0, "cost_basis": 10.0, "pu": 0.7, "date": "2026-11-02"}
        merged = views.merge_market_tranches([t1, t2])
        self.assertEqual(merged["shares"], 30.0)
        self.assertEqual(merged["cost_basis"], 15.0)
        self.assertEqual(merged["pu"], 0.7)
        # Original tranche t1 must remain untouched
        self.assertEqual(t1["shares"], 10.0)
        self.assertEqual(t1["cost_basis"], 5.0)

    def test_classify_active_signal_and_dashboard_stats_parity(self):
        # 0% edge with MIN_EDGE=0.0 must be WATCH, not BUY
        self.assertEqual(views.classify_active_signal(False, {}, 0.0, 0.0, min_edge_th=0.0), "WATCH")
        # Sub-dust kelly (<=0.5%) must be WATCH
        self.assertEqual(views.classify_active_signal(False, {}, 0.004, 0.02, min_edge_th=0.0), "WATCH")
        # Depth-blocked BUY (opt_buy_usd=0.0) must be WATCH
        self.assertEqual(views.classify_active_signal(False, {}, 0.05, 0.05, min_edge_th=0.0, opt_buy_usd=0.0), "WATCH")
        # Held ADD that is depth-blocked must downgrade to HOLD
        self.assertEqual(views.classify_active_signal(True, {"signal_code": "ADD"}, 0.1, 0.05, opt_buy_usd=0.0), "HOLD")
        # Valid BUY
        self.assertEqual(views.classify_active_signal(False, {}, 0.05, 0.05, min_edge_th=0.0, opt_buy_usd=50.0, vwap_edge=0.04), "BUY")

        # Full parity check: duplicate predicted+exited mid + multi-tranche + 0%-edge watch
        history = {
            "profile_id": "local",
            "wallet_summary": {"bankroll": 1000.0, "cash": 800.0, "portfolio": 200.0},
            "predicted": {
                "m1": [
                    {"question": "Multi Held", "pu": 0.70, "pm": 0.50, "cur_pm": 0.50, "shares": 200.0, "cost_basis": 100.0, "kelly": 0.05, "source": "wallet", "date": "2026-11-01"},
                    {"question": "Multi Held", "pu": 0.72, "pm": 0.50, "cur_pm": 0.50, "shares": 200.0, "cost_basis": 100.0, "kelly": 0.05, "source": "wallet", "date": "2026-11-02"},
                ],
                "m2": [{"question": "Zero Edge Watch", "pu": 0.50, "pm": 0.50, "cur_pm": 0.50, "shares": 0.0, "cost_basis": 0.0, "kelly": 0.0, "source": "local"}],
            },
            "exited": {
                # Same market_id m1 already in predicted -> must be deduplicated in dashboard & stats
                "m1": [{"question": "Dup Exited", "pu": 0.70, "pm": 0.50, "cur_pm": 0.50, "exit_price": 0.70, "exited": True, "realized_pnl": 10.0}],
                # Distinct exited market with actionable re-entry edge
                "m3": [{"question": "Rebuy Candidate", "pu": 0.80, "pm": 0.50, "cur_pm": 0.50, "exit_price": 0.80, "exited": True, "realized_pnl": 5.0}],
            },
            "resolved": {},
        }
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            views.print_main_menu_dashboard(history)
        dash_out = buf.getvalue()
        stats = views.calculate_prediction_stats(history, sub_mode="predicted", refresh_meta=False)
        self.assertEqual(stats["total_count"], 3)
        self.assertEqual(stats["held_count"], 1)
        self.assertEqual(stats["buy_count"], 1)
        self.assertEqual(stats["watch_count"], 1)
        self.assertIn("Signals: ADD:1 BUY:1 WATCH:1", dash_out)

        # Verify review table renders WATCH (No Edge) instead of EXEC BUY $0 for 0%-edge item
        tbl_buf = io.StringIO()
        with patch("builtins.input", return_value=""), redirect_stdout(tbl_buf):
            views.print_review_table(history, sub_mode="predicted", refresh_meta=False)
        tbl_out = tbl_buf.getvalue()
        self.assertIn("WATCH (No Edge:", tbl_out)
        self.assertNotIn("EXEC BUY $0", tbl_out)


if __name__ == "__main__":
    unittest.main()

