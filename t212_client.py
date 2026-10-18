"""Trading 212 API client module with Basic Authentication and Ticker Normalization."""

import logging
import re
from typing import Any, Dict, List, Optional
import requests
from config import PENDING_SELLS_CONFIG, PENDING_BUYS_CONFIG

logger = logging.getLogger(__name__)


def normalize_t212_ticker_for_yfinance(t212_ticker: str) -> str:
    """Converts Trading 212 ticker notation into a Yahoo Finance ticker.

    Args:
        t212_ticker: Raw ticker symbol returned from Trading 212 API.

    Returns:
        Cleaned ticker compatible with Yahoo Finance (e.g. RR.L, VUAA.L).
    """
    raw = t212_ticker.strip()
    
    # Specific known tickers
    if "BRK_B" in raw:
        return "BRK-B"

    # London Stock Exchange (ends with l_EQ, _l_EQ, L_EQ, or _L_EQ)
    if re.search(r"_?[lL]_EQ$", raw):
        base = re.sub(r"_?[lL]_EQ$", "", raw).rstrip(".")
        return f"{base}.L"
    
    # German XETRA (_d_EQ or d_EQ)
    if re.search(r"_?[dD]_EQ$", raw):
        base = re.sub(r"_?[dD]_EQ$", "", raw).rstrip(".")
        return f"{base}.DE"

    # Amsterdam (_as_EQ, as_EQ, _nl_EQ, a_EQ, _a_EQ)
    if re.search(r"_?(as|nl|a|AS|NL|A)_EQ$", raw):
        base = re.sub(r"_?(as|nl|a|AS|NL|A)_EQ$", "", raw).rstrip(".")
        return f"{base}.AS"

    # Paris (_fp_EQ, _pa_EQ, pa_EQ)
    if re.search(r"_?(fp|pa|FP|PA)_EQ$", raw):
        base = re.sub(r"_?(fp|pa|FP|PA)_EQ$", "", raw).rstrip(".")
        return f"{base}.PA"

    # Standard US equities (_US_EQ)
    if re.search(r"_[uU][sS]_EQ$", raw):
        base = re.sub(r"_[uU][sS]_EQ$", "", raw)
        return base.replace("_", "-")

    # Swiss Exchange (_s_EQ or s_EQ)
    if re.search(r"_?[sS]_EQ$", raw):
        base = re.sub(r"_?[sS]_EQ$", "", raw).rstrip(".")
        return f"{base}.SW"

    # Fallback: strip any remaining _XX_EQ or XX_EQ suffix
    return re.sub(r"_?[a-zA-Z0-9]+_EQ$", "", raw)


def resolve_canonical_ticker(ticker_str: str) -> str:
    """Resolves any user-configured or broker ticker into a canonical Yahoo Finance symbol."""
    raw = ticker_str.strip()
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

    def _get_with_retry(self, endpoint: str, timeout: int = 15, max_retries: int = 4) -> Optional[requests.Response]:
        """Executes a GET request against Trading 212 API with automatic backoff for 429 rate limits."""
        import time
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=timeout)
                if resp.status_code == 429:
                    wait_time = 3.0 * (attempt + 1)
                    logger.warning(
                        "Trading 212 rate limit (429) on %s. Backing off for %.1fs (attempt %d/%d)...",
                        endpoint, wait_time, attempt + 1, max_retries
                    )
                    time.sleep(wait_time)
                    continue
                return resp
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.warning("Request to %s failed after %d retries: %s", endpoint, max_retries, e)
                    return None
                time.sleep(2.0)
        return None

    def get_account_cash(self) -> Optional[Dict[str, Any]]:
        """Fetches account balance, cash, and total portfolio valuation."""
        if not self.is_configured():
            logger.warning("Trading 212 credentials not configured.")
            return None

        resp = self._get_with_retry("equity/account/cash", timeout=15)
        if resp is not None and resp.status_code == 200:
            cash_data = resp.json()
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
            # Fetch recent cash transactions (withdrawals, deposits, interest)
            try:
                txs = self.get_cash_transactions(limit=10)
                cash_data["recent_transactions"] = txs
            except Exception as ex:
                logger.debug("Could not fetch recent transactions: %s", ex)
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
        resp = self._get_with_retry(f"equity/history/transactions?limit={limit}", timeout=15)
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data.get("items", [])
            elif isinstance(data, list):
                return data
        return []

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

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetches all open positions for this account."""
        if not self.is_configured():
            logger.warning("Trading 212 credentials not configured.")
            return []

        resp = self._get_with_retry("equity/portfolio", timeout=15)
        if resp is not None and resp.status_code == 200:
            try:
                positions = resp.json()
                if isinstance(positions, list):
                    for pos in positions:
                        if isinstance(pos, dict):
                            raw_ticker = pos.get("ticker", "")
                            pos["yf_ticker"] = normalize_t212_ticker_for_yfinance(raw_ticker)
                    return positions
                logger.warning("Trading 212 portfolio endpoint returned unexpected format: %s", type(positions))
                return []
            except Exception as e:
                logger.error("Error parsing Trading 212 positions response: %s", e)
                return []
        else:
            logger.error("Failed to fetch Trading 212 positions: endpoint returned non-200 or timed out.")
            return []

