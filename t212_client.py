"""Trading 212 API client module with Basic Authentication and Ticker Normalization."""

import logging
import re
import time
from typing import Any, Dict, List, Optional
import requests
from config import PENDING_SELLS_CONFIG, PENDING_BUYS_CONFIG

logger = logging.getLogger(__name__)

# --- Trading 212 -> Yahoo Finance exchange-code tables (GEMINI.md §3) ---
# T212 equity tickers end in "_EQ". European listings append a lowercase exchange code directly to the
# symbol ("RRl_EQ", "ASMLa_EQ", "KERpa_EQ"); US/Canadian listings use a separator ("AAPL_US_EQ").
_EQ_SUFFIX_RE = re.compile(r"^(?P<body>.+)_EQ$", re.IGNORECASE)
# Symbol (uppercase/digits/dots/dashes/underscores) followed by its maximal trailing lowercase run.
_ATTACHED_LOWER_RE = re.compile(r"^(?P<base>.*[^a-z])(?P<code>[a-z]+)$")
_BRK_B_RE = re.compile(r"^BRK_B(_US)?(_EQ)?$")

# Lowercase codes attached to the symbol (T212's canonical European form). Matched exactly against the
# whole trailing lowercase run, so multi-letter codes always win ('KERpa' is Paris, never 'KERp' + 'a').
_ATTACHED_LOWER_CODES: Dict[str, str] = {
    "l": ".L",                                 # London Stock Exchange
    "d": ".DE",                                # Deutsche Boerse Xetra
    "a": ".AS", "as": ".AS", "nl": ".AS",      # Euronext Amsterdam
    "p": ".PA", "pa": ".PA", "fp": ".PA",      # Euronext Paris
    "s": ".SW",                                # SIX Swiss Exchange
    "m": ".MI",                                # Borsa Italiana (Milan)
    "e": ".MC",                                # Bolsa de Madrid
}

# Codes written after an underscore separator ("VUAA_L_EQ", "SHOP_CA_EQ", "AAPL_US_EQ"); case-insensitive.
_SEPARATED_CODES: Dict[str, str] = {
    "US": "",
    "CA": ".TO", "TO": ".TO",
    "L": ".L",
    "D": ".DE", "DE": ".DE",
    "A": ".AS", "AS": ".AS", "NL": ".AS",
    "P": ".PA", "PA": ".PA", "FP": ".PA",
    "S": ".SW", "SW": ".SW",
    "M": ".MI", "MI": ".MI",
    "E": ".MC", "MC": ".MC",
}

# Case-folded attached codes (callers such as strategy_engine.tickers_match upper-case tickers before
# normalising, e.g. 'SGLNl_EQ' -> 'SGLNL_EQ'). London and Xetra keep their legacy priority (so 'SGLNL'
# stays SGLN.L rather than SGL + NL), then two-letter codes beat one-letter codes ('ORPA' -> OR.PA).
_ATTACHED_UPPER_PRIORITY = ("L", "D", "PA", "FP", "AS", "NL", "A", "S", "P", "M", "E")


def _yahoo_class_symbol(base: str) -> str:
    """Formats a US/Canadian symbol for Yahoo, which writes share classes with '-' (BRK_B / BRK.B -> BRK-B)."""
    return base.strip("._").replace("_", "-").replace(".", "-")


def normalize_t212_ticker_for_yfinance(t212_ticker: str) -> str:
    """Converts Trading 212 ticker notation into a Yahoo Finance ticker.

    Anchored parser for the trailing exchange code of "<SYMBOL><code>_EQ" / "<SYMBOL>_<CODE>_EQ" tickers
    (London .L, Xetra .DE, Amsterdam .AS, Paris .PA, Swiss .SW, Milan .MI, Madrid .MC, US, Toronto .TO).
    Inputs without an "_EQ" suffix (already Yahoo-style, e.g. "NVDA", "VUAA.L") are returned unchanged.
    Unknown exchange codes are stripped (with a warning) and the bare symbol is returned; the result is
    never an empty string for a non-empty input.

    Args:
        t212_ticker: Raw ticker symbol returned from Trading 212 API.

    Returns:
        Cleaned ticker compatible with Yahoo Finance (e.g. RR.L, VUAA.L).
    """
    raw = str(t212_ticker or "").strip()
    if not raw:
        return raw

    # Specific known tickers
    if _BRK_B_RE.match(raw.upper()):
        return "BRK-B"

    m = _EQ_SUFFIX_RE.match(raw)
    if not m:
        return raw  # Not Trading 212 notation (already a Yahoo-style or plain symbol).
    body = m.group("body")

    # 1. Canonical European form: symbol + lowercase exchange code ("RRl", "ASMLa", "KERpa", "BT_Al").
    lower = _ATTACHED_LOWER_RE.match(body)
    if lower:
        base = lower.group("base").rstrip("._")
        suffix = _ATTACHED_LOWER_CODES.get(lower.group("code"))
        if suffix is not None and base:
            return f"{base}{suffix}"
        return _unknown_suffix(raw, base or body)

    # 2. Separator form: "VUAA_L", "AAPL_US", "BRK_B_US", "SHOP_CA".
    if "_" in body:
        base, code = body.rsplit("_", 1)
        base = base.rstrip("._")
        suffix = _SEPARATED_CODES.get(code.upper())
        if suffix is not None and base:
            if suffix in ("", ".TO"):
                return f"{_yahoo_class_symbol(base)}{suffix}"
            return f"{base}{suffix}"
        return _unknown_suffix(raw, base)

    # 3. Case-folded attached code ("RRL", "ASMLA", "KERPA"): ambiguous by nature, resolved by priority.
    for code in _ATTACHED_UPPER_PRIORITY:
        if body.endswith(code) and len(body) > len(code):
            base = body[: -len(code)].rstrip("._")
            if base:
                return f"{base}{_ATTACHED_LOWER_CODES[code.lower()]}"

    return _unknown_suffix(raw, body)


def _unknown_suffix(raw: str, stripped_base: str) -> str:
    """Fallback for unrecognised exchange codes: stripped base if non-empty, else the raw ticker (never '')."""
    result = stripped_base.strip() if stripped_base and stripped_base.strip() else raw
    logger.warning(
        "Unrecognised Trading 212 exchange suffix in %r; using %r for market-data lookups.", raw, result
    )
    return result


def resolve_canonical_ticker(ticker_str: str) -> str:
    """Resolves any user-configured or broker ticker into a canonical Yahoo Finance symbol."""
    raw = str(ticker_str or "").strip()
    u = raw.upper()
    if u in ("RR", "RR.L", "RRL_EQ", "RR_L_EQ"):
        return "RR.L"
    if u in ("BRK_B", "BRK B", "BRKB", "BRK.B", "BRK-B", "BRK_B_US_EQ"):
        return "BRK-B"
    if u in ("AMZN", "AMZN_US_EQ"):
        return "AMZN"
    if u in ("QQQ3", "QQQ3.L", "QQQ3L_EQ", "QQQ3_L_EQ"):
        return "QQQ3.L"
    if u in ("VUAA", "VUAA.L", "VUAAL_EQ", "VUAA_L_EQ"):
        return "VUAA.L"
    if u in ("VWRP", "VWRP.L", "VWRPL_EQ", "VWRP_L_EQ"):
        return "VWRP.L"
    if u in ("BA.L", "BAL_EQ", "BA_L_EQ"):
        return "BA.L"
    if u in ("CNDX", "CNDX.L", "CNDXL_EQ", "CNDX_L_EQ"):
        return "CNDX.L"
    if u in ("SMGB", "SMGB.L", "SMGBL_EQ", "SMGB_L_EQ"):
        return "SMGB.L"
    if u in ("ASMLA_EQ", "ASML_A_EQ", "ASML_AS_EQ", "ASML.AS"):
        return "ASML.AS"
    if u in ("SGLN", "SGLN.L", "SGLNL_EQ", "SGLN_L_EQ"):
        return "SGLN.L"
    if u in ("CSH2", "CSH2.L", "CSH2L_EQ", "CSH2_L_EQ"):
        return "CSH2.L"
    if u in ("ABBV", "ABBV_US_EQ"):
        return "ABBV"
    if "_EQ" in u:
        return normalize_t212_ticker_for_yfinance(raw)
    return u


class Trading212Client:
    """Client for Trading 212 REST API v0 using Basic Authentication."""

    def __init__(
        self,
        key_id: str,
        secret: str = "",
        base_url: str = "https://live.trading212.com/api/v0",
    ):
        self.key_id = key_id.strip()
        self.secret = secret.strip()
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Trading212DailyAdvisor/1.0",
            "Accept": "application/json",
        })
        if self.key_id and self.secret:
            self.session.auth = (self.key_id, self.secret)
        elif self.key_id:
            # If only a single token was provided
            self.session.headers["Authorization"] = self.key_id

    def is_configured(self) -> bool:
        """Returns True if valid credentials have been provided."""
        return bool(self.key_id and self.secret) or bool(self.key_id)

    # HTTP statuses worth retrying: rate limiting (429), request timeout (408) and any 5xx server error.
    _RETRYABLE_STATUS_CODES = frozenset({408, 429})
    _MAX_RETRY_WAIT_SECONDS = 30.0

    @classmethod
    def _is_retryable_status(cls, status_code: int) -> bool:
        return status_code in cls._RETRYABLE_STATUS_CODES or 500 <= status_code <= 599

    @classmethod
    def _retry_wait_seconds(cls, attempt: int, resp: Optional[requests.Response] = None) -> float:
        """Backoff before the next attempt: 3s/6s/9s for 429, 2s/4s/8s otherwise; honours Retry-After (capped)."""
        status = getattr(resp, "status_code", None)
        wait = 3.0 * (attempt + 1) if status == 429 else 2.0 * (2 ** attempt)
        try:
            headers = getattr(resp, "headers", None) or {}
            retry_after = float(headers.get("Retry-After"))
            wait = max(wait, retry_after)
        except (TypeError, ValueError):
            pass
        return min(wait, cls._MAX_RETRY_WAIT_SECONDS)

    def _get_with_retry(self, endpoint: str, timeout: int = 15, max_retries: int = 4) -> Optional[requests.Response]:
        """Executes a GET request against Trading 212 API with automatic backoff.

        Retries 429 rate limits, 408/5xx server errors and timeouts/connection errors, up to
        `max_retries` attempts in total. Returns the final response (which may still carry an error
        status for the caller to inspect), or None if no response was received at all.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_resp: Optional[requests.Response] = None
        for attempt in range(max_retries):
            is_last_attempt = attempt == max_retries - 1
            try:
                resp = self.session.get(url, timeout=timeout)
            except requests.RequestException as e:
                if is_last_attempt:
                    logger.warning("Request to %s failed after %d attempts: %s", endpoint, max_retries, e)
                    return None
                wait_time = self._retry_wait_seconds(attempt)
                logger.warning(
                    "Request to %s failed (%s). Retrying in %.1fs (attempt %d/%d)...",
                    endpoint, e, wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
                continue

            if self._is_retryable_status(resp.status_code):
                last_resp = resp
                if is_last_attempt:
                    logger.warning(
                        "Trading 212 %s still returning HTTP %s after %d attempts; giving up.",
                        endpoint, resp.status_code, max_retries
                    )
                    return resp
                wait_time = self._retry_wait_seconds(attempt, resp)
                logger.warning(
                    "Trading 212 returned HTTP %s on %s. Backing off for %.1fs (attempt %d/%d)...",
                    resp.status_code, endpoint, wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
                continue
            return resp
        return last_resp

    def get_account_cash(self) -> Optional[Dict[str, Any]]:
        """Fetches account balance, cash, and total portfolio valuation."""
        if not self.is_configured():
            logger.warning("Trading 212 credentials not configured.")
            return None

        resp = self._get_with_retry("equity/account/cash", timeout=15)
        if resp is not None and resp.status_code == 200:
            try:
                cash_data = resp.json()
            except Exception as ex:
                logger.warning("Account cash endpoint returned unparseable JSON: %s", ex)
                return None
            if not isinstance(cash_data, dict):
                logger.warning("Account cash endpoint returned unexpected format: %s", type(cash_data).__name__)
                return None
            # Ensure blocked funds are explicitly recorded (e.g. pending limit orders)
            cash_data.setdefault("blocked", 0.0)
            # Fetch account info to resolve base currency dynamically
            try:
                info_resp = self._get_with_retry("equity/account/info", timeout=10)
                if info_resp is not None and info_resp.status_code == 200:
                    info_data = info_resp.json()
                    cash_data["currency"] = info_data.get("currencyCode", "GBP")
                    cash_data["account_id"] = info_data.get("id")
                else:
                    cash_data.setdefault("currency", "GBP")
            except Exception as ex:
                logger.debug("Could not fetch account info currency: %s", ex)
                cash_data.setdefault("currency", "GBP")
            # Fetch recent cash transactions (withdrawals, deposits, interest).
            # recent_transactions_ok tells deposit-adjusted performance whether the list is trustworthy.
            try:
                txs = self._fetch_cash_transactions(limit=50)
            except Exception as ex:
                logger.debug("Could not fetch recent transactions: %s", ex)
                txs = None
            cash_data["recent_transactions"] = txs if txs is not None else []
            cash_data["recent_transactions_ok"] = txs is not None
            return cash_data
        elif resp is not None:
            logger.warning("Account cash endpoint returned status %s", resp.status_code)
            return None
        else:
            logger.warning("Account cash endpoint failed to return a response.")
            return None

    def get_cash_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches recent cash transactions including deposits, withdrawals, interest, and fees."""
        if not self.is_configured():
            return []
        txs = self._fetch_cash_transactions(limit=limit)
        return txs if txs is not None else []

    def _fetch_cash_transactions(self, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """Like get_cash_transactions, but returns None (not []) when the endpoint fails or is unparseable."""
        resp = self._get_with_retry(f"equity/history/transactions?limit={limit}", timeout=15)
        if resp is None or resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception as ex:
            logger.debug("Transactions endpoint returned unparseable JSON: %s", ex)
            return None
        if isinstance(data, dict):
            items = data.get("items", [])
            return items if isinstance(items, list) else None
        elif isinstance(data, list):
            return data
        return None

    def get_pending_orders(self) -> Dict[str, Any]:
        """Fetches active pending orders from Trading 212 API.

        Gracefully detects when the 'Orders' API permission scope is missing (403),
        falling back to high-level blocked cash telemetry from the cash endpoint.

        Returns:
            Dictionary containing:
                - has_orders_scope: bool
                - orders: list of raw pending orders
                - pending_buys: dict mapping yf_ticker -> order details
                - pending_sells: dict mapping yf_ticker -> order details
                - total_pending_buy_value: float
                - total_pending_sell_value: float
                - scope_notice: status description
        """
        result = {
            "has_orders_scope": False,
            "orders": [],
            "pending_buys": {},
            "pending_sells": {},
            "total_pending_buy_value": 0.0,
            "total_pending_sell_value": 0.0,
            "scope_notice": "",
        }

        if not self.is_configured():
            return result

        try:
            resp = self._get_with_retry("equity/orders", timeout=15)
            if resp is not None and resp.status_code == 200:
                raw_orders = resp.json()
                result["has_orders_scope"] = True
                result["orders"] = raw_orders

                for o in raw_orders:
                    raw_ticker = o.get("ticker", "")
                    yf_ticker = normalize_t212_ticker_for_yfinance(raw_ticker)
                    status = str(o.get("status", "")).upper()
                    if status in ["FILLED", "CANCELLED", "REJECTED"]:
                        continue

                    side = str(o.get("side", "")).upper()
                    raw_qty = float(o.get("quantity", 0.0) or 0.0)
                    limit_price = float(o.get("limitPrice", 0.0) or 0.0)
                    stop_price = float(o.get("stopPrice", 0.0) or 0.0)
                    order_val = float(o.get("value", 0.0) or 0.0)

                    # Determine buy vs sell
                    is_buy = (side == "BUY") or (raw_qty > 0)
                    qty = abs(raw_qty)

                    est_val = order_val if order_val > 0 else (qty * limit_price)

                    order_item = {
                        "id": o.get("id"),
                        "raw_ticker": raw_ticker,
                        "yf_ticker": yf_ticker,
                        "name": o.get("instrument", {}).get("name", yf_ticker),
                        "side": "BUY" if is_buy else "SELL",
                        "quantity": qty,
                        "limit_price": limit_price,
                        "stop_price": stop_price,
                        "value": order_val,
                        "strategy": o.get("strategy", "QUANTITY"),
                        "order_type": o.get("type", "MARKET"),
                        "status": status,
                        "date_created": o.get("createdAt", o.get("dateCreated", "")),
                        "estimated_value": est_val,
                    }

                    if is_buy:
                        result["pending_buys"][yf_ticker] = order_item
                        result["total_pending_buy_value"] += est_val
                    else:
                        result["pending_sells"][yf_ticker] = order_item
                        result["total_pending_sell_value"] += est_val

                logger.info(
                    "Retrieved %d open orders (%d buys, %d sells) from Trading 212 API.",
                    len(result["orders"]),
                    len(result["pending_buys"]),
                    len(result["pending_sells"]),
                )
            elif resp is not None and resp.status_code == 403:
                result["has_orders_scope"] = False
                result["scope_notice"] = (
                    "Trading 212 API key does not have the 'Orders' scope enabled. "
                    "Blocked cash telemetry is utilized for pending capital accounting."
                )
                logger.info(
                    "Trading 212 Orders endpoint returned 403 (read-only key). Blocked cash telemetry will be used."
                )
            elif resp is not None:
                logger.warning("Trading 212 Orders endpoint returned unexpected status: %s", resp.status_code)
            else:
                logger.warning("Trading 212 Orders endpoint failed to return a response or timed out.")
        except Exception as e:
            logger.warning("Failed to fetch Trading 212 pending orders: %s", e)

        # Incorporate any configured pending orders (fallback when API key is read-only)
        for s in PENDING_SELLS_CONFIG:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_key = resolve_canonical_ticker(s_clean)
            if s_key not in result["pending_sells"]:
                result["pending_sells"][s_key] = {
                    "ticker": s_clean,
                    "yf_ticker": s_key,
                    "name": s_clean,
                    "side": "SELL",
                    "is_manual": True,
                    "quantity": 0.0,
                    "limit_price": 0.0,
                    "stop_price": 0.0,
                    "value": 0.0,
                    "estimated_value": 0.0,
                    "strategy": "QUANTITY",
                    "order_type": "PENDING_SELL",
                    "status": "QUEUED",
                }
                logger.info("Pending sell order recognized from configuration fallback: %s", s_key)

        for b in PENDING_BUYS_CONFIG:
            b_clean = b.strip()
            if not b_clean:
                continue
            b_key = resolve_canonical_ticker(b_clean)
            if b_key not in result["pending_buys"]:
                result["pending_buys"][b_key] = {
                    "ticker": b_clean,
                    "yf_ticker": b_key,
                    "name": b_clean,
                    "side": "BUY",
                    "is_manual": True,
                    "quantity": 0.0,
                    "limit_price": 0.0,
                    "stop_price": 0.0,
                    "value": 0.0,
                    "estimated_value": 0.0,
                    "strategy": "QUANTITY",
                    "order_type": "PENDING_BUY",
                    "status": "QUEUED",
                }
                logger.info("Pending buy order recognized from configuration fallback: %s", b_key)

        return result

    def get_open_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches all open positions for this account.

        Returns:
            The list of open positions (each enriched with 'yf_ticker'). An empty list is returned ONLY
            when the API genuinely reports no positions. None is returned on ANY failure (missing
            credentials, non-200 status, timeout/connection error, unparseable JSON, unexpected payload
            type or a position without a ticker) so callers can fail closed instead of mistaking an
            outage for an empty account.
        """
        if not self.is_configured():
            logger.warning("Trading 212 credentials not configured.")
            return None

        resp = self._get_with_retry("equity/portfolio", timeout=15)
        if resp is None:
            logger.error("Failed to fetch Trading 212 positions: no response (timeout or connection error).")
            return None
        if resp.status_code != 200:
            logger.error("Failed to fetch Trading 212 positions: endpoint returned HTTP %s.", resp.status_code)
            return None
        try:
            positions = resp.json()
        except Exception as e:
            logger.error("Error parsing Trading 212 positions response: %s", e)
            return None
        if not isinstance(positions, list):
            logger.error(
                "Trading 212 portfolio endpoint returned unexpected format: %s", type(positions).__name__
            )
            return None
        for pos in positions:
            raw_ticker = pos.get("ticker") if isinstance(pos, dict) else None
            if not isinstance(raw_ticker, str) or not raw_ticker.strip():
                logger.error("Trading 212 portfolio endpoint returned a malformed position entry: %.200r", pos)
                return None
            pos["yf_ticker"] = normalize_t212_ticker_for_yfinance(raw_ticker)
        return positions

