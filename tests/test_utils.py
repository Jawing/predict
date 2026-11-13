"""Unit tests for utils.py pure helpers, valuation, PnL, and exit-field normalization."""
import math
import unittest
from datetime import datetime, timedelta, timezone

from utils import (
    ensure_list,
    parse_outcome_prices,
    parse_market_date,
    format_time_remaining,
    stringify_overflow,
    ansi_link,
    strip_ansi,
    get_history_counts,
    calculate_position_holding,
    get_effective_bankroll,
    calculate_implied_exit_price,
    calculate_resolved_pnl,
    apply_closed_exit_fields,
)


class TestUtilsFormattingAndParsing(unittest.TestCase):
    def test_ensure_list(self):
        self.assertEqual(ensure_list(None), [])
        self.assertEqual(ensure_list([1, 2]), [1, 2])
        self.assertEqual(ensure_list({"a": 1}), [{"a": 1}])

    def test_parse_outcome_prices(self):
        self.assertEqual(parse_outcome_prices({"outcomePrices": '["0.65", "0.35"]'}), [0.65, 0.35])
        self.assertEqual(parse_outcome_prices({"outcomePrices": [0.8, 0.2]}), [0.8, 0.2])
        self.assertEqual(parse_outcome_prices({"outcomePrices": "bad-json"}), [])
        self.assertEqual(parse_outcome_prices(None), [])

    def test_parse_market_date_and_format_remaining(self):
        future = (datetime.now(timezone.utc) + timedelta(days=2, hours=3)).isoformat()
        dt, secs, days = parse_market_date(future)
        self.assertIsNotNone(dt)
        self.assertGreater(secs, 2 * 86400)
        self.assertGreater(days, 2.0)

        self.assertEqual(parse_market_date(None), (None, None, None))
        self.assertEqual(parse_market_date("not-a-date"), (None, None, None))

        self.assertEqual(format_time_remaining(None), "unknown")
        self.assertEqual(format_time_remaining(-5), "Expired")
        self.assertEqual(format_time_remaining(3661), "01:01:01")
        self.assertEqual(format_time_remaining(2 * 86400 + 4 * 3600), "2d 04h")

    def test_stringify_overflow(self):
        payload = {"a": float("inf"), "b": [float("-inf"), float("nan"), 1.5]}
        cleaned = stringify_overflow(payload)
        self.assertEqual(cleaned, {"a": "Infinity", "b": ["-Infinity", "NaN", 1.5]})

    def test_ansi_link_and_strip(self):
        linked = ansi_link("https://example.com", "Label")
        self.assertEqual(strip_ansi(linked), "Label")
        self.assertEqual(ansi_link("unknown", "Label"), "Label")
        self.assertEqual(strip_ansi("\033[32mGreen\033[0m"), "Green")


class TestHistoryCountsAndHoldings(unittest.TestCase):
    def test_get_history_counts(self):
        history = {
            "predicted": {"m1": [{"a": 1}, {"a": 2}], "m2": {"a": 3}},
            "exited": {"m3": [{"a": 4}]},
            "resolved": {"m4": [{"a": 5}, {"a": 6}]},
            "skipped": {"m5": {}},
        }
        counts = get_history_counts(history)
        self.assertEqual(counts["active"], 3)
        self.assertEqual(counts["exited"], 1)
        self.assertEqual(counts["resolved"], 2)
        self.assertEqual(counts["skipped"], 1)
        self.assertEqual(counts["active_markets"], 2)
        self.assertEqual(counts["total_predictions"], 6)
        self.assertEqual(counts["total_markets"], 5)
        self.assertEqual(get_history_counts(None)["total_predictions"], 0)

    def test_calculate_position_holding_variants(self):
        # Exited returns zero
        self.assertEqual(calculate_position_holding({"exited": True, "shares": 100}, 0.8, 1000.0), (0.0, 0.0, 0.0))
        # Unexecuted local with no shares/cost_basis returns zero
        self.assertEqual(calculate_position_holding({"source": "local", "kelly": 0.1}, 0.5, 1000.0), (0.0, 0.0, 0.0))
        # Shares-based valuation
        lv, hk, sh = calculate_position_holding({"shares": 50.0, "cost_basis": 20.0}, cur_pm=0.6, bankroll=100.0)
        self.assertAlmostEqual(lv, 30.0)
        self.assertAlmostEqual(hk, 0.30)
        self.assertAlmostEqual(sh, 50.0)
        # Cost-basis valuation when shares=0
        lv2, hk2, sh2 = calculate_position_holding({"cost_basis": 25.0, "pm": 0.5}, cur_pm=0.75, bankroll=100.0)
        self.assertAlmostEqual(lv2, 37.5)
        self.assertAlmostEqual(hk2, 0.375)
        self.assertAlmostEqual(sh2, 50.0)
        # Wallet with 0 shares/cost_basis falls back to kelly * bankroll
        lv3, hk3, sh3 = calculate_position_holding({"source": "wallet", "kelly": 0.05}, cur_pm=0.5, bankroll=200.0)
        self.assertAlmostEqual(lv3, 10.0)
        self.assertAlmostEqual(hk3, 0.05)
        self.assertEqual(sh3, 0.0)


class TestFinancialHelpers(unittest.TestCase):
    def test_get_effective_bankroll(self):
        self.assertEqual(get_effective_bankroll({"wallet_summary": {"bankroll": 250.5}}, 100.0), 250.5)
        self.assertEqual(get_effective_bankroll({"wallet_summary": {"bankroll": 0}}, 123.0), 123.0)
        self.assertEqual(get_effective_bankroll(None, 80.0), 80.0)

    def test_calculate_implied_exit_price_clamping(self):
        self.assertEqual(calculate_implied_exit_price(0.40, 20.0, 100.0), 0.6)
        self.assertEqual(calculate_implied_exit_price(0.95, 50.0, 100.0), 0.9999)
        self.assertEqual(calculate_implied_exit_price(0.05, -50.0, 100.0), 0.0001)
        self.assertEqual(calculate_implied_exit_price(0.50, 10.0, 0.0, fallback_price=0.42), 0.42)

    def test_calculate_resolved_pnl_all_cases(self):
        # Standard shares + cost_basis across 1.0 / 0.0 / 0.5
        entry = {"shares": 100.0, "cost_basis": 40.0}
        self.assertEqual(calculate_resolved_pnl(entry, 1.0), 60.0)
        self.assertEqual(calculate_resolved_pnl(entry, 0.0), -40.0)
        self.assertEqual(calculate_resolved_pnl(entry, 0.5), 10.0)

        # Watch-only returns None
        self.assertIsNone(calculate_resolved_pnl({"is_watch": True, "shares": 10.0, "cost_basis": 5.0}, 1.0))
        self.assertIsNone(calculate_resolved_pnl({"action": "WATCH", "kelly": 0.1}, 1.0))

        # Exited preservation
        exited = {"exited": True, "realized_pnl": 12.34, "cost_basis": 0.0, "entry_kelly": 0.1, "pm": 0.5}
        self.assertEqual(calculate_resolved_pnl(exited, 0.0, bankroll=100.0, preserve_exited=True), 12.34)

        # Kelly fallback when cb=0, sh=0
        local_unsized = {"entry_kelly": 0.1, "pm": 0.5}
        # cb = 0.1 * 200 = 20, sh = 20 / 0.5 = 40 -> win = 40 - 20 = 20
        self.assertEqual(calculate_resolved_pnl(local_unsized, 1.0, bankroll=200.0, allow_kelly_fallback=True), 20.0)
        self.assertEqual(calculate_resolved_pnl(local_unsized, 0.0, bankroll=200.0, allow_kelly_fallback=True), -20.0)
        self.assertIsNone(calculate_resolved_pnl(local_unsized, 1.0, bankroll=200.0, allow_kelly_fallback=False))

        # entry_cost fallback when enabled
        entry_cost_rec = {"entry_cost": 30.0, "shares": 60.0}
        self.assertEqual(
            calculate_resolved_pnl(entry_cost_rec, 1.0, use_entry_cost_fallback=True, allow_kelly_fallback=False),
            30.0,
        )

        # Wallet API fallback when shares/cost_basis are zero
        wallet_fallback = {"source": "wallet"}
        self.assertEqual(
            calculate_resolved_pnl(wallet_fallback, 1.0, allow_kelly_fallback=False, fallback_api_pnl=18.5),
            18.5,
        )

    def test_apply_closed_exit_fields(self):
        rec = {
            "pm": 0.5,
            "cost_basis": 50.0,
            "snapshots": [{"pm": 0.5, "entry_price": 0.5}],
        }
        apply_closed_exit_fields(
            rec,
            avg_price=0.45,
            total_size=100.0,
            realized_pnl=15.0,
            cur_price=0.60,
            exit_price=0.60,
            exit_date="2026-11-01 12:00",
            bankroll=500.0,
            mark_exited=True,
        )
        self.assertTrue(rec["exited"])
        self.assertEqual(rec["cost_basis"], 0.0)
        self.assertEqual(rec["entry_cost"], 45.0)
        self.assertEqual(rec["total_size"], 100.0)
        self.assertEqual(rec["avg_price"], 0.45)
        self.assertEqual(rec["entry_price"], 0.45)
        self.assertEqual(rec["pm"], 0.45)
        self.assertEqual(rec["snapshots"][0]["pm"], 0.45)
        self.assertEqual(rec["snapshots"][0]["entry_price"], 0.45)
        self.assertEqual(rec["exit_price"], 0.60)
        self.assertEqual(rec["date"], "2026-11-01 12:00")
        self.assertEqual(rec["entry_bankroll"], 500.0)


if __name__ == "__main__":
    unittest.main()

