import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Offline tests for t212_client: ticker normalisation, retry/backoff and fail-closed position fetching.

import unittest  # noqa: E402
from unittest import mock  # noqa: E402

import requests  # noqa: E402

import t212_client  # noqa: E402
from t212_client import (  # noqa: E402
    Trading212Client,
    normalize_t212_ticker_for_yfinance,
    resolve_canonical_ticker,
)


def _response(status=200, json_data=None, json_exc=None, headers=None):
    """Builds a fake requests.Response-like object."""
    resp = mock.Mock()
    resp.status_code = status
    resp.headers = headers or {}
    if json_exc is not None:
        resp.json.side_effect = json_exc
    else:
        resp.json.return_value = json_data
    return resp


class NormalizeTickerTests(unittest.TestCase):
    MUST_KEEP = {
        "RRl_EQ": "RR.L",
        "VUAAl_EQ": "VUAA.L",
        "SGLNl_EQ": "SGLN.L",
        "CSH2l_EQ": "CSH2.L",
        "ASMLa_EQ": "ASML.AS",
        "SAPd_EQ": "SAP.DE",
        "NESNs_EQ": "NESN.SW",
        "AAPL_US_EQ": "AAPL",
        "BRK_B_US_EQ": "BRK-B",
        "VUAA_L_EQ": "VUAA.L",
    }
    MUST_FIX = {
        "MCp_EQ": "MC.PA",
        "KERpa_EQ": "KER.PA",
        "ORfp_EQ": "OR.PA",
        "SHOP_CA_EQ": "SHOP.TO",
        "ENIm_EQ": "ENI.MI",
        "SANe_EQ": "SAN.MC",
    }

    def test_existing_mappings_preserved(self):
        for raw, expected in self.MUST_KEEP.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_t212_ticker_for_yfinance(raw), expected)

    def test_previously_broken_mappings_fixed(self):
        for raw, expected in self.MUST_FIX.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_t212_ticker_for_yfinance(raw), expected)

    def test_separator_and_case_folded_forms(self):
        # strategy_engine.tickers_match upper-cases tickers before normalising them.
        cases = {
            "RRL_EQ": "RR.L",
            "SGLNL_EQ": "SGLN.L",   # London keeps priority over a trailing 'NL'
            "ASMLA_EQ": "ASML.AS",
            "ORPA_EQ": "OR.PA",     # previously mis-mapped to ORP.AS
            "KERPA_EQ": "KER.PA",
            "ORFP_EQ": "OR.PA",
            "MCP_EQ": "MC.PA",
            "ENIM_EQ": "ENI.MI",
            "SANE_EQ": "SAN.MC",
            "NESNS_EQ": "NESN.SW",
            "SAPD_EQ": "SAP.DE",
            "ASML_AS_EQ": "ASML.AS",
            "XYZ_l_EQ": "XYZ.L",
            "BRK.B_US_EQ": "BRK-B",
            "RCI_B_CA_EQ": "RCI-B.TO",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_t212_ticker_for_yfinance(raw), expected)

    def test_yahoo_style_tickers_unchanged(self):
        for ticker in ("NVDA", "VUAA.L", "BRK-B", "MC.PA", "ASML.AS"):
            with self.subTest(ticker=ticker):
                self.assertEqual(normalize_t212_ticker_for_yfinance(ticker), ticker)

    def test_unknown_suffix_returns_stripped_base_and_warns(self):
        for raw, expected in {"FOO_ZZ_EQ": "FOO", "FOOzz_EQ": "FOO", "FOOX_EQ": "FOOX"}.items():
            with self.subTest(raw=raw):
                with self.assertLogs("t212_client", level="WARNING"):
                    self.assertEqual(normalize_t212_ticker_for_yfinance(raw), expected)

    def test_never_returns_empty_string(self):
        with mock.patch.object(t212_client.logger, "warning"):
            for raw in ("_EQ", "__EQ", "x_EQ", "l_EQ", "_l_EQ", "A_EQ", "Q_ZZ_EQ", "_US_EQ", "pa_EQ"):
                with self.subTest(raw=raw):
                    self.assertTrue(normalize_t212_ticker_for_yfinance(raw))

    def test_resolve_canonical_ticker_reuses_parser_and_keeps_special_cases(self):
        cases = {
            "ASMLa_EQ": "ASML.AS",
            "MCp_EQ": "MC.PA",
            "SHOP_CA_EQ": "SHOP.TO",
            "ENIm_EQ": "ENI.MI",
            "RR": "RR.L",
            "VUAA": "VUAA.L",
            "brk.b": "BRK-B",
            "CSH2": "CSH2.L",
            "SGLNL_EQ": "SGLN.L",
            # Idempotent on canonical Yahoo tickers
            "ASML.AS": "ASML.AS",
            "NVDA": "NVDA",
            "VUAA.L": "VUAA.L",
            "MC.PA": "MC.PA",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(resolve_canonical_ticker(raw), expected)


class GetOpenPositionsTests(unittest.TestCase):
    def setUp(self):
        self.client = Trading212Client(
            key_id="test-key", secret="test-secret", base_url="https://example.invalid/api/v0"
        )
        sleep_patcher = mock.patch.object(t212_client.time, "sleep")
        self.sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def _patch_get(self, **kwargs):
        patcher = mock.patch.object(self.client.session, "get", **kwargs)
        get = patcher.start()
        self.addCleanup(patcher.stop)
        return get

    def test_http_500_returns_none_after_retries(self):
        get = self._patch_get(return_value=_response(500))
        self.assertIsNone(self.client.get_open_positions())
        self.assertEqual(get.call_count, 4)
        self.assertEqual(self.sleep.call_count, 3)

    def test_timeout_returns_none_after_retries(self):
        get = self._patch_get(side_effect=requests.Timeout("read timed out"))
        self.assertIsNone(self.client.get_open_positions())
        self.assertEqual(get.call_count, 4)

    def test_bad_json_returns_none(self):
        bad_json = requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)
        self._patch_get(return_value=_response(200, json_exc=bad_json))
        self.assertIsNone(self.client.get_open_positions())

    def test_unexpected_payload_type_returns_none(self):
        self._patch_get(return_value=_response(200, json_data={"items": []}))
        self.assertIsNone(self.client.get_open_positions())

    def test_malformed_position_returns_none(self):
        self._patch_get(return_value=_response(200, json_data=[{"quantity": 3.0}]))
        self.assertIsNone(self.client.get_open_positions())

    def test_genuine_empty_list_returns_empty_list(self):
        self._patch_get(return_value=_response(200, json_data=[]))
        self.assertEqual(self.client.get_open_positions(), [])

    def test_positions_are_enriched_with_yf_ticker(self):
        payload = [{"ticker": "ASMLa_EQ", "quantity": 1.5}, {"ticker": "MCp_EQ", "quantity": 2.0}]
        self._patch_get(return_value=_response(200, json_data=payload))
        positions = self.client.get_open_positions()
        self.assertEqual([p["yf_ticker"] for p in positions], ["ASML.AS", "MC.PA"])

    def test_transient_503_then_success(self):
        get = self._patch_get(side_effect=[_response(503), _response(200, json_data=[])])
        self.assertEqual(self.client.get_open_positions(), [])
        self.assertEqual(get.call_count, 2)

    def test_rate_limit_429_is_retried(self):
        get = self._patch_get(side_effect=[_response(429), _response(408), _response(200, json_data=[])])
        self.assertEqual(self.client.get_open_positions(), [])
        self.assertEqual(get.call_count, 3)

    def test_client_error_is_not_retried(self):
        get = self._patch_get(return_value=_response(401))
        self.assertIsNone(self.client.get_open_positions())
        self.assertEqual(get.call_count, 1)

    def test_unconfigured_client_returns_none(self):
        client = Trading212Client(key_id="", secret="")
        with mock.patch.object(client.session, "get") as get:
            self.assertIsNone(client.get_open_positions())
            get.assert_not_called()


class GetAccountCashTests(unittest.TestCase):
    def setUp(self):
        self.client = Trading212Client(key_id="k", secret="s", base_url="https://example.invalid/api/v0")
        sleep_patcher = mock.patch.object(t212_client.time, "sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def test_unparseable_cash_json_returns_none(self):
        with mock.patch.object(self.client.session, "get", return_value=_response(200, json_exc=ValueError("bad"))):
            self.assertIsNone(self.client.get_account_cash())

    def test_transactions_failure_is_flagged(self):
        responses = {
            "equity/account/cash": _response(200, json_data={"free": 0.0, "total": 100.0, "invested": 100.0}),
            "equity/account/info": _response(200, json_data={"currencyCode": "GBP", "id": 1}),
        }

        def fake_get(url, timeout=None):
            for endpoint, resp in responses.items():
                if url.endswith(endpoint):
                    return resp
            return _response(500)  # transactions endpoint keeps failing

        with mock.patch.object(self.client.session, "get", side_effect=fake_get):
            cash = self.client.get_account_cash()
        self.assertEqual(cash["free"], 0.0)
        self.assertEqual(cash["recent_transactions"], [])
        self.assertFalse(cash["recent_transactions_ok"])


if __name__ == "__main__":
    unittest.main()
