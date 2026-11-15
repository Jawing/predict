"""Tests for analyse.py metrics extraction, API query mapping, sorting, and navigation."""
import unittest
from unittest.mock import MagicMock, patch

import analyse


class TestAnalyse(unittest.TestCase):
    def test_extract_market_metrics_handles_null_numeric_fields(self):
        m = {
            "outcomePrices": '["0.60", "0.40"]',
            "bestBid": None,
            "bestAsk": None,
            "endDate": "2030-01-01T00:00:00Z",
            "volumeNum": None,
            "volume": "1250.5",
            "volume24hr": None,
            "liquidity": None,
            "liquidityNum": 300.0,
        }
        odds, spread, seconds_left, m_vol, m_24h, velocity, m_liq, eff, eii, cey, pulse, churn = (
            analyse.extract_market_metrics(m)
        )
        self.assertAlmostEqual(odds, 0.60)
        self.assertAlmostEqual(m_vol, 1250.5)
        self.assertEqual(m_24h, 0.0)
        self.assertAlmostEqual(m_liq, 300.0)
        self.assertEqual(spread, 0.0)

    def test_sort_markets_handles_mixed_empty_and_iso_start_dates(self):
        markets = [
            {"id": "1", "start_date": "2026-05-01T00:00:00Z"},
            {"id": "2", "start_date": ""},
            {"id": "3", "start_date": None},
            {"id": "4", "start_date": "2026-06-15T12:00:00Z"},
        ]
        sorted_desc = analyse.sort_markets(markets, sort_mode="10", reverse_sort=False)
        self.assertEqual([m["id"] for m in sorted_desc[:2]], ["4", "1"])
        sorted_asc = analyse.sort_markets(markets, sort_mode="10", reverse_sort=True)
        self.assertEqual([m["id"] for m in sorted_asc[-2:]], ["1", "4"])

    @patch("storage.load_history", return_value={"predicted": {}})
    @patch.object(analyse, "api")
    def test_fetch_and_compile_markets_sort_mode_query_mapping(self, mock_api, _mock_hist):
        sample_events = [{
            "id": "ev1",
            "title": "Test Event",
            "slug": "test-event",
            "competitive": None,
            "openInterest": None,
            "tags": [{"label": "Politics"}],
            "markets": [{
                "id": "m1",
                "question": "Will test pass?",
                "active": True,
                "closed": False,
                "umaResolutionStatus": None,
                "outcomePrices": '["0.55", "0.45"]',
                "bestBid": 0.54,
                "bestAsk": 0.56,
                "endDate": "2030-01-01T00:00:00Z",
                "volumeNum": 5000,
                "volume24hr": 400,
                "liquidity": 1000,
            }],
        }]

        expected_queries = [
            ("3", "order=volume", "ascending=false", False),
            ("4", "order=volume24hr", "ascending=false", False),
            ("5", "order=liquidity", "ascending=false", False),
            ("7", "order=volume", "ascending=false", False),
            ("8", "order=endDate", "ascending=true", True),
            ("9", "order=competitive", "ascending=false", False),
            ("10", "order=startDate", "ascending=false", False),
        ]

        for sort_mode, exp_order, exp_asc, expect_end_date_min in expected_queries:
            captured_urls = []

            def fake_get(url, timeout=10):
                captured_urls.append(url)
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = sample_events if "offset=0" in url else []
                return resp

            mock_api.get.side_effect = fake_get
            markets, total_oi = analyse.fetch_and_compile_markets(
                fetch_depth=50, sort_mode=sort_mode, exclude_expired=True
            )
            self.assertEqual(total_oi, 0.0)
            self.assertEqual(len(markets), 1)
            self.assertIn(exp_order, captured_urls[0], f"sort_mode={sort_mode}: {captured_urls[0]}")
            self.assertIn(exp_asc, captured_urls[0], f"sort_mode={sort_mode}: {captured_urls[0]}")
            if expect_end_date_min:
                self.assertIn("end_date_min=", captured_urls[0])
            else:
                self.assertNotIn("end_date_min=", captured_urls[0])

    @patch("sys.stdin")
    @patch("builtins.input")
    def test_get_navigation_key_non_tty_validation(self, mock_input, mock_stdin):
        mock_stdin.isatty.return_value = False

        mock_input.return_value = "#12"
        self.assertEqual(analyse.get_navigation_key("> "), "predict:12")

        mock_input.return_value = "predict 4"
        self.assertEqual(analyse.get_navigation_key("> "), "predict:4")

        mock_input.return_value = "#"
        self.assertEqual(analyse.get_navigation_key("> "), "invalid")

        mock_input.return_value = "#abc"
        self.assertEqual(analyse.get_navigation_key("> "), "invalid")

        mock_input.return_value = "#12"
        self.assertEqual(analyse.get_navigation_key("> ", allow_predict=False), "next")


if __name__ == "__main__":
    unittest.main()
