import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline tests for scripts/backtest.py (Phase 2 evidence system): synthetic data only, never the network.

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "backtest.py")
_spec = importlib.util.spec_from_file_location("backtest", _PATH)
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def month_ends(n, start="2015-01-31"):
    return pd.date_range(start=start, periods=n, freq="ME")


class FxAndSeriesTests(unittest.TestCase):
    def test_usd_to_gbp_divides_by_usd_per_gbp(self):
        idx = month_ends(2)
        usd = pd.Series([100.0, 110.0], index=idx)
        gbpusd = pd.Series([1.25, 1.10], index=idx)
        gbp = bt.usd_to_gbp(usd, gbpusd)
        self.assertAlmostEqual(gbp.iloc[0], 80.0)
        self.assertAlmostEqual(gbp.iloc[1], 100.0)  # a weaker pound raises the GBP value of a USD asset

    def test_splice_uses_fallback_returns_then_primary_returns(self):
        idx = month_ends(5)
        fallback = pd.Series([100.0, 110.0, 121.0, 121.0, 121.0], index=idx)
        primary = pd.Series([np.nan, np.nan, 50.0, 55.0, 60.5], index=idx)
        s = bt.splice_series(primary, fallback)
        self.assertAlmostEqual(s.iloc[0], 1.0)
        self.assertAlmostEqual(s.iloc[2], 1.21)          # fallback growth up to the primary's first price
        self.assertAlmostEqual(s.iloc[4], 1.21 * 1.21)   # primary growth afterwards


class SignalTests(unittest.TestCase):
    def test_momentum_12_1_skips_the_latest_month(self):
        idx = month_ends(13)
        a = np.linspace(100, 120, 13)
        b = np.linspace(100, 110, 13)
        b[-1] = 500.0  # huge jump in the latest month must be ignored
        prices = pd.DataFrame({"A": a, "B": b}, index=idx)
        mom = bt.momentum_12_1(prices)
        self.assertAlmostEqual(mom["A"], a[-2] / a[0] - 1.0)
        self.assertAlmostEqual(mom["B"], b[-2] / b[0] - 1.0)
        self.assertEqual(bt.top_by_momentum(prices, n=1), ["A"])
        self.assertTrue(bt.momentum_12_1(prices.iloc[:12]).isna().all())  # fewer than 13 month-ends

    def test_momentum_uses_only_history_up_to_t(self):
        idx = month_ends(15)
        prices = pd.DataFrame({"A": np.linspace(100, 150, 15), "B": np.linspace(100, 101, 15)}, index=idx)
        prices.loc[idx[13]:, "B"] = 1000.0  # future data must not leak into a decision at idx[12]
        self.assertEqual(bt.top_by_momentum(prices, n=1, t=idx[12]), ["A"])

    def test_sma_signal(self):
        idx = month_ends(10)
        self.assertTrue(bt.sma_signal(pd.Series(np.linspace(100, 120, 10), index=idx)))
        self.assertFalse(bt.sma_signal(pd.Series(np.linspace(120, 100, 10), index=idx)))
        with self.assertRaises(ValueError):
            bt.sma_signal(pd.Series(np.linspace(100, 120, 9), index=month_ends(9)))


class MechanicsTests(unittest.TestCase):
    def test_band_rebalance_triggers_only_beyond_band(self):
        target = pd.Series({"core": 0.85, "gold": 0.075, "gilts": 0.075})
        inside = pd.Series({"core": 0.895, "gold": 0.05, "gilts": 0.055})
        w, hit = bt.band_rebalance(inside, target, 0.05)
        self.assertFalse(hit)
        pd.testing.assert_series_equal(w, inside)
        outside = pd.Series({"core": 0.91, "gold": 0.04, "gilts": 0.05})
        w, hit = bt.band_rebalance(outside, target, 0.05)
        self.assertTrue(hit)
        pd.testing.assert_series_equal(w, target)

    def test_costs_are_turnover_times_rate_with_fx_only_on_sectors(self):
        cost = bt.make_cost_fn(["ACWI", "GLD", "XLK"])
        # 10% moved out of ACWI into XLK: spread on both legs, FX fee on the XLK leg only
        c = cost(np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.0, 0.1]))
        self.assertAlmostEqual(c, 0.1 * bt.SPREAD + 0.1 * (bt.SPREAD + bt.FX_FEE))
        self.assertEqual(cost(np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.0])), 0.0)

    def test_simulate_has_no_look_ahead(self):
        # Asset A jumps 50% in month 3 only. A rule that buys A after seeing a big A return must miss the jump.
        idx = month_ends(6)
        rets = pd.DataFrame({"A": [np.nan, 0.0, 0.0, 0.5, 0.0, 0.0], "B": [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
        prices = (1.0 + rets.fillna(0.0)).cumprod()

        def chase(date_, hist, drifted):
            last = hist["A"].pct_change(fill_method=None).iloc[-1] if len(hist) > 1 else 0.0
            return {"A": 1.0} if last > 0.4 else {"B": 1.0}

        sim = bt.simulate(rets, chase, None, signal_data=prices)
        self.assertAlmostEqual(sim["equity"].iloc[-1], 1.0)  # it only ever holds A after the jump
        self.assertEqual(sim["weights"].loc[idx[3], "A"], 1.0)  # switched at the close that revealed the jump

    def test_weights_drift_between_rebalances_and_costs_apply_on_trades(self):
        idx = month_ends(3)
        rets = pd.DataFrame({"A": [np.nan, 1.0, 0.0], "B": [np.nan, 0.0, 0.0]}, index=idx)
        hold = bt.simulate(rets, lambda d, h, drifted: {"A": 0.5, "B": 0.5} if drifted is None else drifted, None)
        self.assertAlmostEqual(hold["weights"].iloc[1]["A"], 2.0 / 3.0)
        self.assertAlmostEqual(hold["equity"].iloc[-1], 1.5)
        self.assertEqual(hold["turnover"].sum(), 0.0)
        cost = bt.make_cost_fn(["A", "B"], spread=0.01, fx_fee=0.0, fx_assets=())
        reb = bt.simulate(rets, bt.StaticWeights({"A": 0.5, "B": 0.5}), cost)
        tv = abs(0.5 - 2.0 / 3.0) * 2
        self.assertAlmostEqual(reb["turnover"].iloc[1], tv)
        self.assertAlmostEqual(reb["equity"].iloc[-1], 1.5 * (1.0 - tv * 0.01))

    def test_core_satellite_bands_rotate_satellite_within_drifted_sleeve(self):
        fn = bt.CoreSatelliteBands({"core": 0.65, "satellite": 0.20, "gold": 0.075, "gilts": 0.075},
                                   "ACWI", "GLD", "GILTS", ["XLK", "XLF"], select=lambda hist: ["XLF"])
        first = fn(month_ends(1)[0], None, None)
        self.assertAlmostEqual(first["XLF"], 0.20)
        drifted = pd.Series({"ACWI": 0.66, "GLD": 0.07, "GILTS": 0.07, "XLK": 0.0, "XLF": 0.20})
        w = fn(month_ends(2)[1], None, drifted)
        self.assertAlmostEqual(w["ACWI"], 0.66)  # within bands: sleeves keep their drifted weights
        self.assertAlmostEqual(w["XLF"], 0.20)
        self.assertEqual(fn.band_events, [])


class MetricsTests(unittest.TestCase):
    def test_cagr_and_max_drawdown_on_a_known_curve(self):
        idx = month_ends(25)
        eq = pd.Series(np.linspace(1.0, 1.21, 25), index=idx)
        eq.iloc[12] = eq.iloc[11] * 0.8  # a 20% dip that recovers
        m = bt.compute_metrics(eq)
        self.assertAlmostEqual(m["cagr"], 1.21 ** 0.5 - 1.0, places=9)
        self.assertAlmostEqual(m["max_dd"], -0.2, places=9)
        self.assertEqual(m["months"], 24)

    def test_years_beating_benchmark(self):
        idx = pd.date_range("2019-12-31", periods=25, freq="ME")
        bench = pd.Series(np.linspace(1.0, 1.2, 25), index=idx)
        better = pd.Series(np.linspace(1.0, 1.5, 25), index=idx)
        m = bt.compute_metrics(better, bench)
        self.assertEqual(m["years_beating"], 2)
        self.assertEqual(m["years_total"], 2)


if __name__ == "__main__":
    unittest.main()
