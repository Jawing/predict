"""Unit tests for model.py mathematical, Kelly sizing, Brier, and orderbook functions."""
import unittest

import config as cfg
from model import (
    binary_log_loss,
    compute_dynamic_ego,
    calculate_efficiency_scale,
    calculate_logistic_metrics,
    calculate_dynamic_kelly_action,
    compute_trajectory_brier,
    compute_trajectory_logloss,
    calculate_trade_weight,
    calculate_trade_ego_metrics,
    calculate_base_ego,
    is_unforecasted_wallet_trade,
    calculate_annualized_yield,
    parse_user_input,
    calculate_allocation,
    calculate_vwap,
    calculate_optimal_market_buy,
    restore_snapshot_to_prediction,
)


class TestEgoAndLogisticMetrics(unittest.TestCase):
    def setUp(self):
        self._orig_gamma = cfg.KELLY_GAMMA
        self._orig_ego_w = cfg.EGO_WEIGHTING
        cfg.KELLY_GAMMA = 0.5
        cfg.EGO_WEIGHTING = "kelly"

    def tearDown(self):
        cfg.KELLY_GAMMA = self._orig_gamma
        cfg.EGO_WEIGHTING = self._orig_ego_w

    def test_compute_dynamic_ego(self):
        self.assertAlmostEqual(compute_dynamic_ego(0.5, 0.0), 0.5)
        self.assertAlmostEqual(compute_dynamic_ego(None, 0.0), 0.5)
        self.assertAlmostEqual(compute_dynamic_ego(0.0, 0.0), 0.01)
        self.assertLess(compute_dynamic_ego(0.5, 0.5), 0.5)
        self.assertAlmostEqual(compute_dynamic_ego(0.5, 1.0), 0.0)

    def test_calculate_efficiency_scale(self):
        self.assertEqual(calculate_efficiency_scale(0.64, gamma=0.0), 1.0)
        self.assertEqual(calculate_efficiency_scale(0.0, gamma=0.5), 0.0)
        self.assertEqual(calculate_efficiency_scale(0.64, gamma=1.0), 0.64)
        self.assertAlmostEqual(calculate_efficiency_scale(0.64, gamma=0.5), 0.8)

    def test_calculate_logistic_metrics_positive_and_zero_edge(self):
        m_zero = calculate_logistic_metrics(0.5, 0.5, ego=0.5)
        self.assertEqual(m_zero["active_edge"], 0.0)
        self.assertEqual(m_zero["model_kelly"], 0.0)
        self.assertEqual(m_zero["eff_model_kelly"], 0.0)

        m_pos = calculate_logistic_metrics(0.70, 0.50, ego=0.5, fee_rate=0.0)
        self.assertGreater(m_pos["pu_norm"], 0.50)
        self.assertLess(m_pos["pu_norm"], 0.70)
        self.assertGreater(m_pos["active_edge"], 0.0)
        self.assertGreater(m_pos["eff_model_kelly"], 0.0)
        self.assertAlmostEqual(m_pos["eff_m"], 1.0)

    def test_calculate_logistic_metrics_raw_yes_space_selects_no_side(self):
        m = calculate_logistic_metrics(0.30, 0.50, ego=0.5, raw_yes_space=True)
        self.assertFalse(m["is_yes"])
        self.assertGreater(m["active_edge"], 0.0)
        self.assertGreater(m["eff_model_kelly"], 0.0)

    def test_calculate_dynamic_kelly_action_states(self):
        # 1. Price at/above pooled target -> SELL
        act_sell = calculate_dynamic_kelly_action(0.55, 0.65, held_frac=0.10, bankroll=1000.0, entry_kelly=0.10)
        self.assertEqual(act_sell["action_type"], "SELL")
        self.assertEqual(act_sell["signal_code"], "SELL")

        # 2. Within 0.5c of target at bid -> TAKE_PROFIT / EXIT
        act_tp = calculate_dynamic_kelly_action(0.55, 0.548, held_frac=0.10, bankroll=1000.0, entry_kelly=0.20, best_bid=0.546)
        self.assertEqual(act_tp["action_type"], "TAKE_PROFIT")
        self.assertEqual(act_tp["signal_code"], "EXIT")

        # 3. Under-allocated relative to ask -> ADD
        act_add = calculate_dynamic_kelly_action(0.75, 0.50, held_frac=0.05, bankroll=1000.0, entry_kelly=0.05)
        self.assertEqual(act_add["action_type"], "ADD")
        self.assertGreater(act_add["delta_usd"], 0.0)

        # 4. Held inside no-trade zone -> HOLD
        act_hold = calculate_dynamic_kelly_action(0.60, 0.50, held_frac=0.20, bankroll=1000.0, entry_kelly=0.20)
        self.assertEqual(act_hold["action_type"], "HOLD")
        self.assertEqual(act_hold["delta_usd"], 0.0)

        # 5. Grown > 2 * f_star_bid -> TRIM
        act_trim = calculate_dynamic_kelly_action(0.60, 0.50, held_frac=0.55, bankroll=1000.0, entry_kelly=0.20)
        self.assertEqual(act_trim["action_type"], "TRIM")
        self.assertGreater(act_trim["delta_usd"], 0.0)

        # 6. Thesis downgrade: stale higher entry_kelly must not suppress TRIM off executable f_star_bid
        act_downgrade = calculate_dynamic_kelly_action(0.52, 0.50, held_frac=0.20, bankroll=1000.0, entry_kelly=0.20)
        self.assertEqual(act_downgrade["action_type"], "TRIM")
        self.assertAlmostEqual(act_downgrade["f_star_hold"], 0.04, places=4)
        self.assertAlmostEqual(act_downgrade["delta_usd"], (0.20 - act_downgrade["f_star_bid"]) * 1000.0, places=2)

    def test_wide_spread_exit_symmetry(self):
        # Midpoint above target (0.51 >= 0.502), but wide Bid=0.40 still offers +10.2% edge -> LIMIT_SELL, never market SELL into 0.40
        act_limit = calculate_dynamic_kelly_action(
            0.502, 0.51, held_frac=0.10, bankroll=100.0, half_spread=0.11, best_bid=0.40, best_ask=0.62
        )
        self.assertEqual(act_limit["action_type"], "LIMIT_SELL")
        self.assertEqual(act_limit["signal_code"], "LIMIT")
        self.assertIn("LIMIT SELL", act_limit["action_label"])

        # Holding 15% when midpoint=0.54 (f*_hold=2.17% -> 2x=4.35%) yet Bid=0.40 (f*_bid=24.5%) -> HOLD, no spread-crossing TRIM
        act_wide_hold = calculate_dynamic_kelly_action(
            0.55, 0.54, held_frac=0.15, bankroll=100.0, half_spread=0.14, best_bid=0.40, best_ask=0.68
        )
        self.assertEqual(act_wide_hold["action_type"], "HOLD")
        self.assertEqual(act_wide_hold["delta_usd"], 0.0)

        # If oversized > 2 * f_star_bid even at the low Bid=0.40 -> TRIM back to f_star_bid
        act_wide_trim = calculate_dynamic_kelly_action(
            0.502, 0.51, held_frac=0.50, bankroll=100.0, half_spread=0.11, best_bid=0.40, best_ask=0.62
        )
        self.assertEqual(act_wide_trim["action_type"], "TRIM")
        self.assertAlmostEqual(act_wide_trim["f_star_eff"], act_wide_trim["f_star_bid"], places=6)

    def test_calculate_logistic_metrics_tail_precision(self):
        m_high = calculate_logistic_metrics(0.999, 0.995, ego=0.5, fee_rate=0.0)
        self.assertGreater(m_high["pu_norm"], 0.995)
        self.assertLess(m_high["pu_norm"], 0.999)
        self.assertGreater(m_high["active_edge"], 0.0)

        m_low = calculate_logistic_metrics(0.001, 0.005, ego=0.5, fee_rate=0.0, raw_yes_space=True)
        self.assertFalse(m_low["is_yes"])
        self.assertGreater(m_low["active_edge"], 0.0)

        # Extreme boundary beyond eps never flips edge negative when pu > pm
        m_ext = calculate_logistic_metrics(1.0, 0.99995, ego=0.5, fee_rate=0.0)
        self.assertGreaterEqual(m_ext["pu_norm"], 0.99995)
        self.assertGreaterEqual(m_ext["active_edge"], 0.0)


class TestBrierAndEgoCalibration(unittest.TestCase):
    def setUp(self):
        self._orig_ego_w = cfg.EGO_WEIGHTING
        cfg.EGO_WEIGHTING = "kelly"

    def tearDown(self):
        cfg.EGO_WEIGHTING = self._orig_ego_w

    def test_compute_trajectory_brier(self):
        bu, bm = compute_trajectory_brier({"pu": 0.9, "pm": 0.5}, outcome=1.0)
        self.assertAlmostEqual(bu, 0.01)
        self.assertAlmostEqual(bm, 0.25)

        data = {
            "snapshots": [
                {"ts": 100, "pu": 0.8, "pm": 0.5},
                {"ts": 200, "pu": 0.9, "pm": 0.6},
            ]
        }
        bu_t, bm_t = compute_trajectory_brier(data, outcome=1.0)
        self.assertLess(bu_t, bm_t)

    def test_calculate_base_ego_fallback_and_exclusion(self):
        self.assertEqual(calculate_base_ego({"resolved": {}}), 0.5)

        history = {
            "wallet_summary": {"bankroll": 100.0},
            "resolved": {
                f"m{i}": [{
                    "pu": 0.85,
                    "pm": 0.50,
                    "outcome": 1.0,
                    "kelly": 0.1,
                    "cost_basis": 10.0,
                    "realized_pnl": 10.0,
                }]
                for i in range(4)
            },
        }
        ego = calculate_base_ego(history, verbose=False)
        self.assertGreater(ego, 0.5)

        # Marking all trades excluded from ego falls back to 0.5
        for preds in history["resolved"].values():
            preds[0]["exclude_from_ego"] = True
        self.assertEqual(calculate_base_ego(history, verbose=False), 0.5)

    def test_calculate_trade_weight_and_metrics(self):
        w = calculate_trade_weight({"cost_basis": 25.0, "entry_bankroll": 100.0}, bankroll=100.0, is_live_wallet=True)
        self.assertAlmostEqual(w, 0.25)
        met = calculate_trade_ego_metrics(
            {"pu": 0.8, "pm": 0.5, "outcome": 1.0, "cost_basis": 25.0, "realized_pnl": 25.0},
            cat="resolved",
            bankroll=100.0,
            is_live_wallet=True,
        )
        self.assertLess(met["bs_u"], met["bs_m"])
        self.assertGreater(met["net_adv"], 0.0)

    def test_exited_ego_uses_expected_logloss(self):
        met = calculate_trade_ego_metrics(
            {"pu": 0.70, "pm": 0.50, "exit_price": 0.65, "cost_basis": 20.0, "exited": True},
            cat="exited",
            bankroll=100.0,
        )
        exp_ll_u = binary_log_loss(0.70, 0.65)
        exp_ll_m = binary_log_loss(0.50, 0.65)
        self.assertAlmostEqual(met["bs_u"], exp_ll_u)
        self.assertAlmostEqual(met["bs_m"], exp_ll_m)
        self.assertAlmostEqual(met["net_adv"], 0.20 * (exp_ll_m - exp_ll_u))

        ego = calculate_base_ego({
            "wallet_summary": {"bankroll": 100.0},
            "resolved": {},
            "exited": {"m1": [{"pu": 0.70, "pm": 0.50, "exit_price": 0.65, "cost_basis": 20.0, "exited": True}]},
        }, verbose=False)
        self.assertAlmostEqual(ego, exp_ll_m / (exp_ll_u + exp_ll_m))
        self.assertLess(ego, 0.60)

    def test_uniform_weighting_includes_watch_forecasts(self):
        watch_trade = {"pu": 0.80, "pm": 0.50, "outcome": 1.0, "is_watch": True, "action": "WATCH"}
        self.assertEqual(calculate_trade_weight(watch_trade, weighting="kelly"), 0.0)
        self.assertEqual(calculate_trade_weight(watch_trade, weighting="uniform"), 1.0)

        cfg.EGO_WEIGHTING = "uniform"
        ego = calculate_base_ego({
            "wallet_summary": {"bankroll": 100.0},
            "resolved": {"m1": [watch_trade]},
            "exited": {},
        }, verbose=False)
        self.assertGreater(ego, 0.5)

    def test_compute_trajectory_logloss(self):
        lu, lm = compute_trajectory_logloss({"pu": 0.9, "pm": 0.5}, outcome=1.0)
        self.assertAlmostEqual(lu, binary_log_loss(0.9, 1.0))
        self.assertAlmostEqual(lm, binary_log_loss(0.5, 1.0))
        self.assertLess(lu, lm)

    def test_unforecasted_wallet_sync_excluded_and_symmetric_ties(self):
        cfg.EGO_WEIGHTING = "uniform"
        real_forecast = {"pu": 0.75, "pm": 0.50, "outcome": 1.0, "source": "local"}
        wallet_win = {"pu": 0.60, "pm": 0.60, "exit_price": 0.70, "realized_pnl": 5.0, "source": "wallet"}
        wallet_loss = {"pu": 0.60, "pm": 0.60, "exit_price": 0.50, "realized_pnl": -5.0, "source": "wallet"}

        self.assertTrue(is_unforecasted_wallet_trade(wallet_win))
        self.assertEqual(calculate_trade_weight(wallet_win, weighting="uniform"), 0.0)
        self.assertEqual(calculate_trade_weight(wallet_loss, weighting="kelly"), 0.0)

        # Even if stats passes exclude_from_ego=False, unforecasted wallet trades carry 0 ego drag
        m_wloss = calculate_trade_ego_metrics({**wallet_loss, "exclude_from_ego": False}, cat="exited")
        self.assertEqual(m_wloss["weight"], 0.0)
        self.assertEqual(m_wloss["ego_drag"], 0.0)

        # Unforecasted wallet trades do not dilute genuine blind-forecast ego
        ego_real = calculate_base_ego({"resolved": {"m1": [real_forecast]}, "exited": {}}, verbose=False)
        ego_with_wallet = calculate_base_ego(
            {"resolved": {"m1": [real_forecast]}, "exited": {"m2": [wallet_win], "m3": [wallet_loss]}},
            verbose=False,
        )
        self.assertAlmostEqual(ego_with_wallet, ego_real)

        # Genuine local blind prediction matching pm (pu == pm) is scored symmetrically for wins and losses
        local_tie_loss = {"pu": 0.60, "pm": 0.60, "exit_price": 0.50, "realized_pnl": -5.0, "source": "local"}
        m_tie_loss = calculate_trade_ego_metrics(local_tie_loss, cat="exited", weighting="uniform")
        self.assertEqual(m_tie_loss["weight"], 1.0)
        self.assertAlmostEqual(m_tie_loss["bs_u"], m_tie_loss["bs_m"])
        self.assertAlmostEqual(m_tie_loss["ego_drag"], 0.0)

        # Uniform weighting grades multi-snapshot updates by held duration dt_i (invariant to micro-updates & kelly=0)
        traj_data = {
            "snapshots": [
                {"ts": 100, "pu": 0.40, "pm": 0.50, "kelly": 0.0},
                {"ts": 200, "pu": 0.80, "pm": 0.50, "kelly": 0.25},
            ]
        }
        lu_u, lm_u = compute_trajectory_logloss(traj_data, outcome=1.0, weighting="uniform")
        lu_k, _ = compute_trajectory_logloss(traj_data, outcome=1.0, weighting="kelly")
        self.assertAlmostEqual(lu_u, 0.5 * (binary_log_loss(0.40, 1.0) + binary_log_loss(0.80, 1.0)))
        self.assertAlmostEqual(lm_u, binary_log_loss(0.50, 1.0))
        self.assertLess(lu_k, lu_u)

        # Late micro-tweaks right before resolution cannot dilute a long-held early forecast
        micro_data = {
            "resolved_ts": 1110,
            "snapshots": [
                {"ts": 100, "pu": 0.90, "pm": 0.50, "kelly": 0.0},
                {"ts": 1100, "pu": 0.10, "pm": 0.15, "kelly": 0.0},
                {"ts": 1101, "pu": 0.11, "pm": 0.15, "kelly": 0.0},
                {"ts": 1102, "pu": 0.12, "pm": 0.15, "kelly": 0.0},
            ],
        }
        lu_m, _ = compute_trajectory_logloss(micro_data, outcome=0.0, weighting="uniform")
        expected_micro = (
            1000.0 * binary_log_loss(0.90, 0.0)
            + 1.0 * binary_log_loss(0.10, 0.0)
            + 1.0 * binary_log_loss(0.11, 0.0)
            + 8.0 * binary_log_loss(0.12, 0.0)
        ) / 1010.0
        self.assertAlmostEqual(lu_m, expected_micro, places=6)


class TestInputsAllocationAndOrderbook(unittest.TestCase):
    def test_parse_user_input(self):
        self.assertEqual(parse_user_input("60"), (0.60, 0.60))
        self.assertEqual(parse_user_input("55-65"), (0.55, 0.65))
        self.assertEqual(parse_user_input("70% - 60%"), (0.60, 0.70))
        with self.assertRaises(ValueError):
            parse_user_input("105")
        with self.assertRaises(ValueError):
            parse_user_input("abc")

    def test_calculate_annualized_yield(self):
        roi, apy = calculate_annualized_yield(0.10, 0.50, 36.5)
        self.assertAlmostEqual(roi, 0.20)
        self.assertGreater(apy, 1.0)
        self.assertEqual(calculate_annualized_yield(0.0, 0.5, 10.0), (0.0, 0.0))

    def test_calculate_allocation(self):
        action, true_px, raw_k, eff_k, alloc_usd, dyn_ego, edge, lm = calculate_allocation(
            0.65, 0.75, pm_bid=0.49, pm_ask=0.50, fee_rate=0.0, bankroll=1000.0, base_ego=0.5
        )
        self.assertEqual(action, "YES")
        self.assertGreater(alloc_usd, 0.0)
        self.assertGreater(edge, 0.0)
        self.assertTrue(lm["is_yes"])

        # Below min_edge gives THIN_EDGE with 0 allocation
        action_t, _, _, _, alloc_t, _, _, _ = calculate_allocation(
            0.52, 0.54, pm_bid=0.49, pm_ask=0.50, fee_rate=0.0, bankroll=1000.0, base_ego=0.5, custom_min_edge=0.05
        )
        self.assertEqual(action_t, "THIN_EDGE")
        self.assertEqual(alloc_t, 0.0)

        # No edge gives NONE
        action_n, _, _, _, alloc_n, _, _, _ = calculate_allocation(
            0.49, 0.50, pm_bid=0.49, pm_ask=0.50, fee_rate=0.0, bankroll=1000.0, base_ego=0.5
        )
        self.assertEqual(action_n, "NONE")
        self.assertEqual(alloc_n, 0.0)

    def test_calculate_vwap_and_optimal_buy(self):
        asks = [
            {"price": "0.40", "size": "100"},  # $40 depth
            {"price": "0.50", "size": "100"},  # $50 depth
        ]
        vwap, spent = calculate_vwap(asks, 65.0)
        # $40 @ 0.40 = 100 sh; $25 @ 0.50 = 50 sh -> 150 sh total -> vwap = 65/150
        self.assertAlmostEqual(spent, 65.0)
        self.assertAlmostEqual(vwap, 65.0 / 150.0)

        fill_usd, mean_vwap, vwap_edge = calculate_optimal_market_buy(asks, target_usdc=65.0, p_norm=0.55, fee_rate=0.0, min_edge=0.02)
        self.assertAlmostEqual(fill_usd, 65.0)
        self.assertAlmostEqual(mean_vwap, round(65.0 / 150.0, 4))
        self.assertGreater(vwap_edge, 0.02)

        # If p_norm is below first level edge, nothing is filled
        fill_none, _, _ = calculate_optimal_market_buy(asks, target_usdc=65.0, p_norm=0.40, fee_rate=0.0, min_edge=0.02)
        self.assertEqual(fill_none, 0.0)

    def test_restore_snapshot_to_prediction(self):
        p = {"pu": 0.9, "pm": 0.7, "snapshots": []}
        snap = {"pu": 0.65, "pm": 0.50, "kelly": 0.08, "edge": 0.075, "dynamic_ego": 0.5, "action": "BUY"}
        restore_snapshot_to_prediction(p, snap, base_ego=0.5)
        self.assertEqual(p["pu"], 0.65)
        self.assertEqual(p["pm"], 0.50)
        self.assertEqual(p["kelly"], 0.08)
        self.assertEqual(p["action"], "BUY")


if __name__ == "__main__":
    unittest.main()
