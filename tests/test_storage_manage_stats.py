"""Tests for storage.py profile recovery/isolation, manage.py manual_resolve, and views.py stats."""
import glob
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import config as cfg
import analyse
import manage
import market
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
        # Sub-1% positive edge (+0.6%) respects MIN_EDGE=0.0 without a hardcoded 1% floor
        self.assertEqual(views.classify_active_signal(False, {}, 0.015, 0.006, min_edge_th=0.0, opt_buy_usd=15.0, vwap_edge=0.006), "BUY")

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
        # Also verify toggling expand ('x') and sort ('s edge') re-renders without rebuilding rows
        tbl_buf = io.StringIO()
        with patch("views._build_review_rows", wraps=views._build_review_rows) as spy_build, \
             patch("builtins.input", side_effect=["99", "x", "s edge", ""]), \
             redirect_stdout(tbl_buf):
            ret_enter = views.print_review_table(history, sub_mode="predicted", refresh_meta=False)
        self.assertIsNone(ret_enter)
        with patch("builtins.input", return_value="p"), redirect_stdout(io.StringIO()):
            ret_prev = views.print_review_table(history, sub_mode="predicted", refresh_meta=False)
        self.assertEqual(ret_prev, "p")
        tbl_out = tbl_buf.getvalue()
        self.assertEqual(spy_build.call_count, 1)
        self.assertIn("s: sort | x: expand | l: live | d <#>: details | u <#>: undo", tbl_out)
        self.assertIn("Row number out of range (1-3).", tbl_out)
        self.assertIn("WATCH (No Edge:", tbl_out)
        self.assertNotIn("EXEC BUY $0", tbl_out)
        self.assertIn("Full: Zero Edge Watch", tbl_out)
        self.assertIn("sorted by EDGE", tbl_out)

    def test_resolved_calibration_bias_and_luck_z_score(self):
        import io
        from contextlib import redirect_stdout

        # 6 balanced 50/50 resolved bets all winning -> extremely lucky positive Z-score and positive bias
        resolved_entries = {
            f"r{i}": [{
                "question": f"Resolved Q{i}",
                "position": "Yes",
                "pu": 0.60,
                "pm": 0.50,
                "cur_pm": 1.0,
                "outcome": 1.0,
                "kelly": 0.10,
                "realized_pnl": 10.0,
                "source": "local",
            }]
            for i in range(6)
        }
        history = {
            "profile_id": "local",
            "wallet_summary": {"bankroll": 100.0},
            "predicted": {},
            "exited": {},
            "resolved": resolved_entries,
        }
        stats = views.calculate_prediction_stats(history, sub_mode="resolved", base_ego=0.5, refresh_meta=False)
        self.assertEqual(stats["res_count"], 6)
        self.assertAlmostEqual(stats["res_actual_win_rate"], 1.0)
        self.assertGreater(stats["res_calib_bias_u"], 0.35)
        self.assertGreater(stats["res_luck_z"], 2.0)
        self.assertEqual(views.classify_luck_z(stats["res_luck_z"]), "Extremely Lucky")
        self.assertEqual(views.classify_luck_z(-2.1), "Extremely Unlucky")
        self.assertEqual(views.classify_luck_z(0.2), "Normal Variance")

        buf = io.StringIO()
        with redirect_stdout(buf):
            views.print_prediction_stats(history, sub_mode="resolved", base_ego=0.5, refresh_meta=False)
            views.print_main_menu_dashboard(history)
        out = buf.getvalue()
        self.assertIn("Calibration Bias :", out)
        self.assertIn("Luck Z-Score     :", out)
        self.assertIn("Extremely Lucky", out)
        self.assertIn("Reflection     :", out)

    def test_epistemic_pulse_and_book_churn_metrics(self):
        m_high_pulse = {
            "outcomePrices": '["0.50", "0.50"]',
            "bestBid": 0.49,
            "bestAsk": 0.51,
            "endDate": "2030-01-01T00:00:00Z",
            "volumeNum": 100000.0,
            "volume24hr": 40000.0,
            "liquidity": 10000.0,
        }
        m_low_liq = {
            "outcomePrices": '["0.90", "0.10"]',
            "bestBid": 0.89,
            "bestAsk": 0.91,
            "endDate": "2030-01-01T00:00:00Z",
            "volumeNum": 50000.0,
            "volume24hr": 20000.0,
            "liquidity": 0.0,
        }
        odds, _, _, vol, v24, vel, liq, eff, eii, cey, pulse, churn = analyse.extract_market_metrics(m_high_pulse)
        self.assertAlmostEqual(eff, 1.0)
        self.assertAlmostEqual(pulse, 40000.0)
        self.assertAlmostEqual(pulse, eii * vel)
        self.assertAlmostEqual(churn, 4.0)

        _, _, _, _, _, _, _, eff2, _, _, pulse2, churn2 = analyse.extract_market_metrics(m_low_liq)
        self.assertAlmostEqual(eff2, 4.0 * 0.9 * 0.1)
        self.assertAlmostEqual(pulse2, eff2 * 20000.0)
        self.assertEqual(churn2, 0.0)

        market.enrich_market_scale_metrics(m_high_pulse)
        self.assertAlmostEqual(m_high_pulse["pulse"], 40000.0)
        self.assertAlmostEqual(m_high_pulse["churn"], 4.0)

        rows = [
            {"id": "a", "pulse": 100.0, "churn": 5.0},
            {"id": "b", "pulse": 5000.0, "churn": 1.2},
        ]
        self.assertEqual([r["id"] for r in analyse.sort_markets(rows, "14", False)], ["b", "a"])
        self.assertEqual([r["id"] for r in analyse.sort_markets(rows, "15", False)], ["a", "b"])


class TestStorageBackupAndRestore(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._orig = {
            "HISTORY_DIR": cfg.HISTORY_DIR,
            "HISTORY_FILE": cfg.HISTORY_FILE,
            "ACTIVE_PROFILE_FILE": cfg.ACTIVE_PROFILE_FILE,
            "POLYGON_WALLET": cfg.POLYGON_WALLET,
            "BANKROLL": cfg.BANKROLL,
            "MIN_EDGE": cfg.MIN_EDGE,
            "MAX_DAYS": cfg.MAX_DAYS,
            "MIN_DAYS": cfg.MIN_DAYS,
            "EXTREME_ODDS": cfg.EXTREME_ODDS,
            "KELLY_GAMMA": cfg.KELLY_GAMMA,
        }
        cfg.HISTORY_DIR = os.path.join(self.tmp_dir.name, "data")
        cfg.HISTORY_FILE = os.path.join(self.tmp_dir.name, "prediction_history.json")
        cfg.ACTIVE_PROFILE_FILE = os.path.join(self.tmp_dir.name, "active_profile.txt")
        cfg.POLYGON_WALLET = ""
        cfg.BANKROLL = 100.0

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(cfg, k, v)
        self.tmp_dir.cleanup()

    def test_backup_and_restore_roundtrip_profiles_and_settings(self):
        import zipfile
        import predict

        storage.set_active_profile("local")
        local_data = {
            "alias": "Paper",
            "predicted": {"m1": [{"pu": 0.72, "pm": 0.50, "kelly": 0.12}]},
            "exited": {},
            "resolved": {},
            "skipped": {},
            "wallet_summary": {"cash": 500.0, "portfolio": 0.0, "bankroll": 500.0},
            "settings": {"MIN_EDGE": 0.02, "MAX_DAYS": 180},
        }
        storage.save_history(local_data, profile_id="local")

        wallet_pid = "0x1234567890abcdef1234567890abcdef12345678"
        wallet_data = {
            "alias": "Main",
            "predicted": {"m2": [{"pu": 0.65, "pm": 0.40, "kelly": 0.08}]},
            "exited": {},
            "resolved": {},
            "skipped": {},
            "wallet_summary": {"cash": 1200.0, "portfolio": 300.0, "bankroll": 1500.0},
            "settings": {"MIN_EDGE": 0.015, "EXTREME_ODDS": 0.03},
        }
        storage.save_history(wallet_data, profile_id=wallet_pid)
        storage.set_active_profile(wallet_pid)
        cfg.KELLY_GAMMA = 0.75
        cfg.MIN_EDGE = 0.015

        b_path = storage.create_backup()
        self.assertTrue(os.path.isfile(b_path))

        with zipfile.ZipFile(b_path, "r") as zf:
            names = set(zf.namelist())
            self.assertEqual(
                names,
                {"active_profile.txt", "data/history_local.json", f"data/history_{wallet_pid}.json"},
            )

        # Create a non-zip file in backups dir to verify list_backups ignores it
        backups_dir = os.path.dirname(b_path)
        with open(os.path.join(backups_dir, "stray.json"), "w", encoding="utf-8") as sf:
            sf.write("{}")
        backups = storage.list_backups()
        self.assertEqual(len(backups), 1)
        self.assertEqual(set(backups[0]["profiles"]), {"local", wallet_pid})

        # Wipe/mutate state before restoring
        os.remove(storage.get_profile_path("local"))
        storage.save_history({"predicted": {}, "wallet_summary": {"bankroll": 50.0}}, profile_id=wallet_pid)
        storage.set_active_profile("local")
        cfg.MIN_EDGE = 0.0

        res = storage.restore_backup(b_path, create_safety_backup=True)
        self.assertIsNotNone(res["safety_backup"])
        self.assertTrue(os.path.isfile(res["safety_backup"]))
        self.assertEqual(res["active_profile"], wallet_pid)
        self.assertEqual(storage.get_active_profile(), wallet_pid)
        self.assertEqual(cfg.MIN_EDGE, 0.015)

        restored_local = storage.load_history("local")
        self.assertEqual(restored_local["predicted"]["m1"][0]["pu"], 0.72)
        restored_wallet = storage.load_history(wallet_pid)
        self.assertEqual(restored_wallet["predicted"]["m2"][0]["pu"], 0.65)
        self.assertEqual(cfg.BANKROLL, 1500.0)

        # CLI flags smoke test
        custom_zip = os.path.join(self.tmp_dir.name, "manual_backup.zip")
        self.assertTrue(predict.handle_cli_args(["--backup", custom_zip]))
        self.assertTrue(os.path.isfile(custom_zip))
        self.assertTrue(predict.handle_cli_args(["--list-backups"]))
        self.assertTrue(predict.handle_cli_args(["--restore", custom_zip]))

    def test_auto_backup_throttling_and_retention_pruning(self):
        import time

        # No history yet -> auto_backup does nothing
        self.assertIsNone(storage.auto_backup())

        storage.set_active_profile("local")
        storage.save_history({"predicted": {"m1": [{"pu": 0.6}]}}, profile_id="local")

        # First call creates a backup; immediate second call is throttled
        b1 = storage.auto_backup(max_age_hours=24, keep=2)
        self.assertIsNotNone(b1)
        self.assertIsNone(storage.auto_backup(max_age_hours=24, keep=2))

        # Backdate b1 mtime so auto_backup triggers again, and test keep=2 pruning
        old_ts = time.time() - (25 * 3600)
        os.utime(b1, (old_ts, old_ts))
        b2 = storage.auto_backup(max_age_hours=24, keep=2)
        self.assertIsNotNone(b2)
        b3 = storage.create_backup(keep=2)
        self.assertTrue(os.path.isfile(b3))

        backups = storage.list_backups()
        self.assertEqual(len(backups), 2)
        self.assertFalse(os.path.exists(b1))

    def test_restore_rejects_unsafe_or_corrupt_zip_without_modifying_disk(self):
        import zipfile

        storage.set_active_profile("local")
        storage.save_history({"predicted": {"keep": [{"pu": 0.8}]}}, profile_id="local")

        traversal_zip = os.path.join(self.tmp_dir.name, "evil.zip")
        with zipfile.ZipFile(traversal_zip, "w") as zf:
            zf.writestr("../escape.txt", "bad")
        with self.assertRaises(ValueError):
            storage.restore_backup(traversal_zip, create_safety_backup=False)

        corrupt_zip = os.path.join(self.tmp_dir.name, "corrupt_inner.zip")
        with zipfile.ZipFile(corrupt_zip, "w") as zf:
            zf.writestr("data/history_local.json", "{not json")
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            storage.restore_backup(corrupt_zip, create_safety_backup=False)

        # Existing data must remain intact
        loaded = storage.load_history("local")
        self.assertIn("keep", loaded["predicted"])

    def test_load_history_migrates_exited_when_exited_bucket_has_legacy_dict(self):
        storage.set_active_profile("local")
        path = storage.get_profile_path("local")
        raw = {
            "predicted": {
                "m1": [
                    {"question": "Active", "pu": 0.6, "exited": False},
                    {"question": "Newly exited", "pu": 0.7, "exited": True},
                ]
            },
            "exited": {
                "m1": {"question": "Legacy exited dict", "pu": 0.5, "exited": True}
            },
            "resolved": {},
            "skipped": {},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        loaded = storage.load_history("local")
        self.assertEqual(len(loaded["predicted"]["m1"]), 1)
        self.assertEqual(len(loaded["exited"]["m1"]), 2)

    @patch("builtins.input")
    def test_delete_specific_markets_multiple_tranches_and_snapshots(self, mock_input):
        storage.set_active_profile("local")
        history = {
            "predicted": {
                "m_tr": [
                    {"question": "Multi tranche", "cost_basis": 10.0, "shares": 20.0, "position": "Yes"},
                    {"question": "Multi tranche", "cost_basis": 20.0, "shares": 40.0, "position": "Yes", "is_tranche": True},
                    {"question": "Multi tranche", "cost_basis": 30.0, "shares": 60.0, "position": "Yes", "is_tranche": True},
                ],
                "m_snap": [
                    {
                        "question": "Multi snap",
                        "pu": 0.8,
                        "snapshots": [
                            {"pu": 0.6, "pm": 0.5, "date": "2026-01-01"},
                            {"pu": 0.7, "pm": 0.5, "date": "2026-01-02"},
                            {"pu": 0.8, "pm": 0.5, "date": "2026-01-03"},
                        ],
                    }
                ],
            },
            "exited": {},
            "resolved": {},
            "skipped": {},
        }

        # Delete tranches 1.1 and 1.2 (leaves tranche 3 = $30)
        mock_input.side_effect = ["2", "1.1, 1.2", "y"]
        manage.delete_specific_markets(history)
        self.assertEqual(len(history["predicted"]["m_tr"]), 1)
        self.assertEqual(history["predicted"]["m_tr"][0]["cost_basis"], 30.0)

        # Undo snapshots 2.2 and 2.3 on m_snap (now index #2, wait m_tr is #1, m_snap is #2 -> leaves only initial snap)
        mock_input.side_effect = ["2", "2.2, 2.3", "y"]
        manage.delete_specific_markets(history)
        self.assertEqual(len(history["predicted"]["m_snap"][0]["snapshots"]), 1)
        self.assertEqual(history["predicted"]["m_snap"][0]["snapshots"][0]["pu"], 0.6)

    @patch("storage.save_history")
    def test_ensure_predicted_metadata_clears_stale_opt_buy_when_no_edge(self, _mock_save):
        def fake_api_get(url, timeout=5):
            if "clob.polymarket.com/book" in url:
                return unittest.mock.MagicMock(
                    status_code=200,
                    json=lambda: {
                        "bids": [{"price": "0.79", "size": "100"}],
                        "asks": [{"price": "0.80", "size": "100"}],
                    },
                )
            return unittest.mock.MagicMock(
                status_code=200,
                json=lambda: [{
                    "id": "m_stale",
                    "closed": False,
                    "active": True,
                    "outcomePrices": '["0.80", "0.20"]',
                    "bestBid": 0.79,
                    "bestAsk": 0.80,
                    "clobTokenIds": '["tok_yes", "tok_no"]',
                    "endDate": "2030-01-01T00:00:00Z",
                }],
            )

        with patch.object(cfg.api, "get", side_effect=fake_api_get):
            history = {
                "predicted": {
                    "m_stale": [{
                        "question": "Stale opt buy",
                        "slug": "stale-opt-buy",
                        "endDate": "2030-01-01T00:00:00Z",
                        "clobTokenIds": '["tok_yes", "tok_no"]',
                        "pu": 0.55,
                        "pm": 0.80,
                        "cur_pm": 0.80,
                        "kelly": 0.0,
                        "position": "Yes",
                        "cost_basis": 0.0,
                        "shares": 0.0,
                        "date": "2026-01-01 00:00",
                        "opt_buy_usd": 25.0,
                        "opt_buy_vwap": 0.51,
                        "opt_buy_edge": 0.04,
                    }]
                },
                "exited": {},
                "resolved": {},
                "skipped": {},
            }
            views.ensure_predicted_market_metadata(history, fetch_clob=True)
            p = history["predicted"]["m_stale"][0]
            self.assertNotIn("opt_buy_usd", p)
            self.assertNotIn("opt_buy_vwap", p)
            self.assertNotIn("opt_buy_edge", p)

    def test_toggle_ego_market_inclusion_uniform_and_unforecasted(self):
        import io
        from contextlib import redirect_stdout

        orig_ego_w = cfg.EGO_WEIGHTING
        cfg.EGO_WEIGHTING = "uniform"
        try:
            history = {
                "profile_id": "local",
                "wallet_summary": {"bankroll": 100.0},
                "predicted": {},
                "exited": {
                    "m_uncal": [{
                        "question": "Wallet Unforecasted",
                        "pu": 0.60, "pm": 0.60, "exit_price": 0.50,
                        "realized_pnl": -5.0, "source": "wallet", "date": "2026-01-02",
                    }],
                },
                "resolved": {
                    "m_real": [{
                        "question": "Genuine Forecast",
                        "pu": 0.80, "pm": 0.50, "outcome": 1.0,
                        "kelly": 0.05, "source": "local", "date": "2026-01-01",
                    }],
                },
                "skipped": {},
            }
            buf = io.StringIO()
            with patch.object(manage, "save_history"), patch("builtins.input", side_effect=["2", "reset", ""]):
                with redirect_stdout(buf):
                    manage.toggle_ego_market_inclusion(history)
            out = buf.getvalue()
            self.assertIn("Weight", out)
            self.assertIn("1.00x", out)
            self.assertIn("Date", out)
            self.assertIn("2026-01-01", out)
            self.assertIn("2026-01-02", out)
            self.assertIn("adj <#>: kelly | s: sort | sort <col> | reset", out)
            self.assertIn("[NO-PU]", out)
            self.assertIn("Active: 1 / 2 markets", out)
            self.assertIn("stays excluded from Base Ego", out)
            self.assertNotIn("exclude_from_ego", history["exited"]["m_uncal"][0])
            self.assertFalse(history["resolved"]["m_real"][0]["exclude_from_ego"])
        finally:
            cfg.EGO_WEIGHTING = orig_ego_w

    def test_settings_menu_ego_weighting_persists_and_restores(self):
        orig_dir = cfg.HISTORY_DIR
        orig_active = cfg.ACTIVE_PROFILE_FILE
        orig_ego_w = cfg.EGO_WEIGHTING
        with tempfile.TemporaryDirectory() as tmp:
            cfg.HISTORY_DIR = tmp
            cfg.ACTIVE_PROFILE_FILE = os.path.join(tmp, "active_profile.txt")
            try:
                storage.set_active_profile("paper_ego")
                storage.save_history({"predicted": {}, "exited": {}, "resolved": {}, "skipped": {}}, profile_id="paper_ego")
                cfg.EGO_WEIGHTING = "uniform"

                with patch("builtins.input", side_effect=["6", "kelly", "6", "invalid", "p"]):
                    manage.settings_menu()
                self.assertEqual(cfg.EGO_WEIGHTING, "kelly")

                cfg.EGO_WEIGHTING = "uniform"
                reloaded = storage.load_history("paper_ego")
                self.assertEqual(reloaded["settings"]["EGO_WEIGHTING"], "kelly")
                self.assertEqual(cfg.EGO_WEIGHTING, "kelly")
            finally:
                cfg.HISTORY_DIR = orig_dir
                cfg.ACTIVE_PROFILE_FILE = orig_active
                cfg.EGO_WEIGHTING = orig_ego_w

    def test_save_history_handles_nan_and_inf_fallback(self):
        orig_dir = cfg.HISTORY_DIR
        orig_active = cfg.ACTIVE_PROFILE_FILE
        with tempfile.TemporaryDirectory() as tmp:
            cfg.HISTORY_DIR = tmp
            cfg.ACTIVE_PROFILE_FILE = os.path.join(tmp, "active_profile.txt")
            try:
                storage.set_active_profile("nan_prof")
                storage.save_history(
                    {"predicted": {"m1": [{"apy": float("inf"), "edge": float("nan")}]}, "exited": {}, "resolved": {}, "skipped": {}},
                    profile_id="nan_prof",
                )
                loaded = storage.load_history("nan_prof")
                self.assertEqual(loaded["predicted"]["m1"][0]["apy"], "Infinity")
                self.assertEqual(loaded["predicted"]["m1"][0]["edge"], "NaN")
            finally:
                cfg.HISTORY_DIR = orig_dir
                cfg.ACTIVE_PROFILE_FILE = orig_active

    def test_auto_prune_persists_timestamp_across_reload(self):
        orig_dir = cfg.HISTORY_DIR
        orig_active = cfg.ACTIVE_PROFILE_FILE
        with tempfile.TemporaryDirectory() as tmp:
            cfg.HISTORY_DIR = tmp
            cfg.ACTIVE_PROFILE_FILE = os.path.join(tmp, "active_profile.txt")
            try:
                storage.set_active_profile("prune_prof")
                storage.save_history(
                    {"predicted": {}, "exited": {}, "resolved": {}, "skipped": {"m_skip": {"question": "Q", "endDate": "2030-01-01T00:00:00Z"}}},
                    profile_id="prune_prof",
                )
                with patch("manage.prune_closed_skipped_markets", return_value=0) as mock_prune:
                    h1 = storage.load_history("prune_prof")
                    views.auto_prune_closed_markets(h1)
                    self.assertEqual(mock_prune.call_count, 1)

                    h2 = storage.load_history("prune_prof")
                    views.auto_prune_closed_markets(h2)
                    self.assertEqual(mock_prune.call_count, 1)
            finally:
                cfg.HISTORY_DIR = orig_dir
                cfg.ACTIVE_PROFILE_FILE = orig_active

    @patch("views.save_history")
    def test_ensure_predicted_metadata_uses_prefetched_and_skips_exited_clob(self, _mock_save):
        called_urls = []

        def fake_get(url, timeout=5):
            called_urls.append(url)
            if "clob.polymarket.com/book" in url:
                return unittest.mock.MagicMock(
                    status_code=200,
                    json=lambda: {
                        "bids": [{"price": "0.49", "size": "100"}],
                        "asks": [{"price": "0.51", "size": "100"}],
                    },
                )
            raise AssertionError(f"Unexpected Gamma HTTP call: {url}")

        prefetched = {
            "m_pred": {
                "id": "m_pred", "closed": False, "active": True,
                "outcomePrices": '["0.50", "0.50"]', "bestBid": 0.49, "bestAsk": 0.51,
                "clobTokenIds": '["tok_pred_yes", "tok_pred_no"]', "endDate": "2030-01-01T00:00:00Z",
            },
            "m_exit": {
                "id": "m_exit", "closed": False, "active": True,
                "outcomePrices": '["0.60", "0.40"]', "bestBid": 0.59, "bestAsk": 0.61,
                "clobTokenIds": '["tok_exit_yes", "tok_exit_no"]', "endDate": "2030-01-01T00:00:00Z",
            },
        }
        history = {
            "predicted": {"m_pred": [{"pu": 0.7, "pm": 0.5, "position": "Yes", "clobTokenIds": '["tok_pred_yes", "tok_pred_no"]'}]},
            "exited": {"m_exit": [{"pu": 0.7, "pm": 0.5, "exited": True, "position": "Yes", "clobTokenIds": '["tok_exit_yes", "tok_exit_no"]'}]},
            "resolved": {},
            "skipped": {},
        }
        with patch.object(cfg.api, "get", side_effect=fake_get):
            views.ensure_predicted_market_metadata(history, fetch_clob=True, prefetched_markets=prefetched)

        self.assertEqual(len(called_urls), 1)
        self.assertIn("tok_pred_yes", called_urls[0])
        self.assertNotIn("tok_exit_yes", called_urls[0])
        self.assertEqual(history["exited"]["m_exit"][0]["cur_pm"], 0.6)

    def test_streamlined_data_management_clear_and_reset(self):
        orig_dir = cfg.HISTORY_DIR
        orig_active = cfg.ACTIVE_PROFILE_FILE
        with tempfile.TemporaryDirectory() as tmp:
            cfg.HISTORY_DIR = tmp
            cfg.ACTIVE_PROFILE_FILE = os.path.join(tmp, "active_profile.txt")
            try:
                storage.set_active_profile("dm_prof")
                storage.save_history(
                    {
                        "predicted": {"m1": [{"pu": 0.6, "pm": 0.5}]},
                        "exited": {},
                        "resolved": {},
                        "skipped": {"m_sk": {"question": "Skip"}},
                    },
                    profile_id="dm_prof",
                )
                # Option 3 -> 4 (Clear Skipped) -> CONFIRM -> Option 3 -> 5 (Reset Entire Profile) -> CONFIRM -> p (exit)
                with patch("builtins.input", side_effect=["3", "4", "CONFIRM", "3", "5", "CONFIRM", "p"]):
                    manage.handle_delete_data()
                after = storage.load_history("dm_prof")
                self.assertEqual(after["predicted"], {})
                self.assertEqual(after["skipped"], {})
            finally:
                cfg.HISTORY_DIR = orig_dir
                cfg.ACTIVE_PROFILE_FILE = orig_active


if __name__ == "__main__":
    unittest.main()

