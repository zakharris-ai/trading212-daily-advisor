import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline tests for config.py: defensive env parsing, review-fix constants and the optional settings
# (earmarked funds, outside holdings, reserve top-ups, account wrappers, leverage policy). Behaviour is
# exercised through injected inputs, so the values in a user's own .env never change the outcome.

import json  # noqa: E402
import unittest  # noqa: E402
from datetime import date  # noqa: E402
from unittest import mock  # noqa: E402

import config  # noqa: E402
from config import _safe_float, _safe_int  # noqa: E402

# Fixed account names (the public defaults) for _build_accounts(), independent of the user's .env
ACCOUNT_ENV = {"ACCOUNT_NAME": "Main", "SECOND_ACCOUNT_NAME": "Secondary"}


def build_accounts(**kwargs):
    with mock.patch.dict(os.environ, ACCOUNT_ENV):
        return config._build_accounts(**kwargs)


class SafeParserTests(unittest.TestCase):
    def test_safe_int(self):
        self.assertEqual(_safe_int("true", 7), 7)
        self.assertEqual(_safe_int("", 7), 7)
        self.assertEqual(_safe_int(None, 7), 7)
        self.assertEqual(_safe_int("12", 7), 12)
        self.assertEqual(_safe_int(" 12 ", 7), 12)
        self.assertEqual(_safe_int("false", 7), 7)
        self.assertEqual(_safe_int("abc", 7), 7)

    def test_safe_float(self):
        self.assertEqual(_safe_float("true", 1.5), 1.5)
        self.assertEqual(_safe_float("", 1.5), 1.5)
        self.assertEqual(_safe_float(None, 1.5), 1.5)
        self.assertEqual(_safe_float("12", 1.5), 12.0)
        self.assertEqual(_safe_float("null", 1.5), 1.5)
        self.assertEqual(_safe_float("1,500", 1.5), 1.5)


class ReviewConstantsTests(unittest.TestCase):
    # These tests gate the daily email in CI, so they check that settings are well-formed,
    # never their configured values (changing a setting must not block the email).
    def test_rebalance_config_fractions_are_valid(self):
        for key in ("max_cluster_pct", "max_leveraged_etf_pct", "max_position_pct", "target_crash_shield_pct"):
            with self.subTest(key=key):
                val = config.REBALANCE_CONFIG[key]
                self.assertIsInstance(val, float)
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)
        self.assertIsInstance(config.REBALANCE_CONFIG["cap_applies_to_broad_etfs"], bool)

    def test_model_version_is_set(self):
        self.assertIsInstance(config.MODEL_VERSION, str)
        self.assertTrue(config.MODEL_VERSION.strip())

    def test_london_date(self):
        self.assertIsInstance(config.london_date(), date)
        self.assertIsNotNone(config.LONDON_TZ)

    def test_shipped_defaults_are_generic(self):
        # Every optional setting is opt-in via the environment; nothing personal ships as a default.
        self.assertEqual(config._DEFAULT_EARMARKED_TICKERS, set())
        self.assertEqual(config._DEFAULT_OUTSIDE_HOLDINGS, {})
        self.assertEqual(config._DEFAULT_RESERVE_TOPUPS, {})
        self.assertEqual(config._DEFAULT_ACCOUNT_WRAPPERS, {})
        self.assertEqual(config._DEFAULT_MAX_LEVERAGED_ETF_PCT, 0.08)

    def test_earmarked_tickers_parsing(self):
        default = {"CSH2.L"}
        self.assertEqual(config._parse_ticker_set(None, default), {"CSH2.L"})
        self.assertEqual(config._parse_ticker_set("   ", default), {"CSH2.L"})
        self.assertEqual(config._parse_ticker_set("csh2.l, vwrp.l,,", default), {"CSH2.L", "VWRP.L"})
        self.assertEqual(config._parse_ticker_set("none", default), set())
        self.assertEqual(config._parse_ticker_set(None, set()), set())
        self.assertIsInstance(config.EARMARKED_TICKERS, set)
        self.assertTrue(all(t == t.upper() for t in config.EARMARKED_TICKERS))
        self.assertIsInstance(config.EXCLUDED_TICKERS, set)
        self.assertTrue(all(t == t.upper() for t in config.EXCLUDED_TICKERS))

    def test_leverage_fraction_parsing(self):
        parse = config._parse_fraction
        self.assertEqual(parse(None, 0.08, "MAX_LEVERAGED_ETF_PCT"), 0.08)
        self.assertEqual(parse("  ", 0.08, "MAX_LEVERAGED_ETF_PCT"), 0.08)
        self.assertEqual(parse("0", 0.08, "MAX_LEVERAGED_ETF_PCT"), 0.0)  # "exit 3x ETFs" policy
        self.assertEqual(parse("0.05", 0.08, "MAX_LEVERAGED_ETF_PCT"), 0.05)
        for bad in ("8", "-0.1", "abc", "true"):
            with self.subTest(raw=bad):
                with self.assertLogs("config", level="WARNING"):
                    self.assertEqual(parse(bad, 0.08, "MAX_LEVERAGED_ETF_PCT"), 0.08)


class AccountTests(unittest.TestCase):
    def test_default_account_name_is_main(self):
        blank = {"ACCOUNT_NAME": "", "SECOND_ACCOUNT_NAME": "", "T212_KEY_ID_2": "", "T212_SECRET_2": "",
                 "T212_API_KEY_2": ""}
        with mock.patch.dict(os.environ, blank):
            accounts = config._build_accounts(today=date(2026, 10, 20), topups={})
        self.assertEqual(list(accounts), ["Main"])

    def test_named_second_account(self):
        accounts = build_accounts(today=date(2026, 10, 20), topups={})
        self.assertEqual(list(accounts), ["Main", "Secondary"])

    def test_configured_accounts_are_well_formed(self):
        self.assertTrue(config.ACCOUNTS)
        for name, acc in config.ACCOUNTS.items():
            with self.subTest(account=name):
                self.assertIsInstance(name, str)
                self.assertGreaterEqual(acc["min_cash_reserve_gbp"], acc["base_min_cash_reserve_gbp"])
                self.assertGreaterEqual(acc["base_min_cash_reserve_gbp"], 0.0)

    def test_account_wrappers_from_env(self):
        names = ["Main", "Secondary", "My ISA"]
        env = {"ACCOUNT_WRAPPER": "", "ACCOUNT_WRAPPER_2": "", "ACCOUNT_WRAPPER_MAIN": "isa",
               "ACCOUNT_WRAPPER_SECONDARY": "", "ACCOUNT_WRAPPER_MY_ISA": "isa"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(config._build_account_wrappers(defaults={}, account_names=names),
                             {"Main": "ISA", "My ISA": "ISA"})
        env.update({"ACCOUNT_WRAPPER_MAIN": "", "ACCOUNT_WRAPPER_SECONDARY": "gia"})
        with mock.patch.dict(os.environ, env):
            wrappers = config._build_account_wrappers(defaults={"Main": "isa"}, account_names=names)
        self.assertEqual(wrappers["Main"], "ISA")  # blank env keeps the default
        self.assertEqual(wrappers["Secondary"], "GIA")
        # Positional aliases (first / second configured account), as mapped by the GitHub Actions workflow
        env.update({"ACCOUNT_WRAPPER_SECONDARY": "", "ACCOUNT_WRAPPER_MY_ISA": "", "ACCOUNT_WRAPPER": "isa",
                    "ACCOUNT_WRAPPER_2": "gia"})
        with mock.patch.dict(os.environ, env):
            self.assertEqual(config._build_account_wrappers(defaults={}, account_names=names),
                             {"Main": "ISA", "Secondary": "GIA"})

    def test_configured_wrappers_are_well_formed(self):
        self.assertIsInstance(config.ACCOUNT_WRAPPERS, dict)
        self.assertTrue(all(isinstance(v, str) and v == v.upper() for v in config.ACCOUNT_WRAPPERS.values()))


class ReserveTopupTests(unittest.TestCase):
    TOPUPS = {"Main": {"amount_gbp": 2000.0, "until": "2026-11-01"}}

    def test_topup_active_before_and_on_until_date(self):
        for today in (date(2026, 10, 18), date(2026, 11, 1)):
            with self.subTest(today=today):
                accounts = build_accounts(today=today, topups=self.TOPUPS)
                main = accounts["Main"]
                self.assertAlmostEqual(main["min_cash_reserve_gbp"], main["base_min_cash_reserve_gbp"] + 2000.0)
                self.assertEqual(main["reserve_topup_gbp"], 2000.0)
                self.assertEqual(main["reserve_topup_until"], "2026-11-01")
                secondary = accounts["Secondary"]
                self.assertEqual(secondary["min_cash_reserve_gbp"], secondary["base_min_cash_reserve_gbp"])

    def test_topup_inactive_after_until_date(self):
        accounts = build_accounts(today=date(2026, 11, 2), topups=self.TOPUPS)
        main = accounts["Main"]
        self.assertEqual(main["min_cash_reserve_gbp"], main["base_min_cash_reserve_gbp"])
        self.assertEqual(main["reserve_topup_gbp"], 0.0)
        self.assertIsNone(main["reserve_topup_until"])

    def test_topup_keys_match_account_names_case_insensitively(self):
        accounts = build_accounts(today=date(2026, 10, 18), topups={"main": {"amount_gbp": 500, "until": "2026-10-31"}})
        self.assertEqual(accounts["Main"]["reserve_topup_gbp"], 500.0)

    def test_no_topups_by_default(self):
        accounts = build_accounts(today=date(2026, 10, 20), topups={})
        for acc in accounts.values():
            self.assertEqual(acc["reserve_topup_gbp"], 0.0)
            self.assertEqual(acc["min_cash_reserve_gbp"], acc["base_min_cash_reserve_gbp"])

    def test_configured_topup_table_is_well_formed(self):
        self.assertIsInstance(config.RESERVE_TOPUPS, dict)
        accounts = build_accounts(today=date(2026, 10, 20))
        for acc in accounts.values():
            self.assertGreaterEqual(acc["min_cash_reserve_gbp"], acc["base_min_cash_reserve_gbp"])

    def test_malformed_topups_are_safe(self):
        today = date(2026, 10, 18)
        for topups in ({"Main": "garbage"}, {"Main": {"amount_gbp": "true", "until": "2026-11-01"}}, {"Main": {}}):
            with self.subTest(topups=topups):
                with mock.patch.object(config.logger, "warning"):
                    main = build_accounts(today=today, topups=topups)["Main"]
                self.assertEqual(main["min_cash_reserve_gbp"], main["base_min_cash_reserve_gbp"])

    def test_unreadable_until_keeps_cash_protected(self):
        with self.assertLogs("config", level="WARNING"):
            main = build_accounts(
                today=date(2027, 1, 1), topups={"Main": {"amount_gbp": "2000", "until": "not-a-date"}}
            )["Main"]
        self.assertAlmostEqual(main["min_cash_reserve_gbp"], main["base_min_cash_reserve_gbp"] + 2000.0)

    def test_reserve_topups_json_parsing(self):
        parse = config._parse_reserve_topups
        table = {"Main": {"amount_gbp": 2000, "until": "2026-12-31"}}
        for raw in (None, "", "   "):
            with self.subTest(raw=raw):
                parsed = parse(raw, table)
                self.assertEqual(parsed, table)
                self.assertIsNot(parsed, table)
        self.assertEqual(parse(json.dumps(table), {}), table)
        self.assertEqual(parse("none", table), {})
        for bad in ("{not valid json", "[1, 2]"):
            with self.subTest(raw=bad):
                with self.assertLogs("config", level="WARNING"):
                    self.assertEqual(parse(bad, {}), {})


class OutsideHoldingsTests(unittest.TestCase):
    DEFAULT = {"Main": [{"ticker": "ACME", "shares_vested": 10.0, "shares_unvested": 5.0, "currency": "USD"}]}

    def test_configured_value_is_well_formed(self):
        self.assertIsInstance(config.OUTSIDE_HOLDINGS, dict)
        for items in config.OUTSIDE_HOLDINGS.values():
            for holding in items:
                self.assertTrue(str(holding.get("ticker", "")).strip())
                self.assertGreaterEqual(float(holding.get("shares_vested", 0.0)), 0.0)
                self.assertGreaterEqual(float(holding.get("shares_unvested", 0.0)), 0.0)

    def test_unset_or_blank_uses_default_copy(self):
        for raw in (None, "", "   "):
            with self.subTest(raw=raw):
                parsed = config._parse_outside_holdings(raw, self.DEFAULT)
                self.assertEqual(parsed, self.DEFAULT)
                self.assertIsNot(parsed, self.DEFAULT)
        parsed["Main"][0]["shares_vested"] = -1.0  # deep copy: the default is untouched
        self.assertEqual(self.DEFAULT["Main"][0]["shares_vested"], 10.0)

    def test_valid_json_override(self):
        override = {"Main": [{"ticker": "ACME", "shares_vested": 250.0, "shares_unvested": 300.0}], "Secondary": []}
        self.assertEqual(config._parse_outside_holdings(json.dumps(override), {}), override)

    def test_invalid_json_keeps_default_and_warns(self):
        with self.assertLogs("config", level="WARNING"):
            self.assertEqual(config._parse_outside_holdings("{not valid json", self.DEFAULT), self.DEFAULT)

    def test_wrong_shape_keeps_default_and_warns(self):
        for raw in ("[1, 2]", '{"Main": {"ticker": "ACME"}}', '{"Main": [{"shares_vested": 1}]}'):
            with self.subTest(raw=raw):
                with self.assertLogs("config", level="WARNING"):
                    self.assertEqual(config._parse_outside_holdings(raw, self.DEFAULT), self.DEFAULT)


if __name__ == "__main__":
    unittest.main()
