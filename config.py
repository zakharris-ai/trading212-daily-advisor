"""Configuration module for Trading 212 Daily Recommendation & Rebalancing Engine."""

import copy
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


def load_dotenv():
    """Lightweight loader for .env file if present."""
    env_path = Path(__file__).parent / ".env"
    if env_path.is_file():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v


load_dotenv()


def _safe_float(val: Any, default: float) -> float:
    """Safely converts an environment variable to float, avoiding crashes on 'true', empty, or malformed strings."""
    if val is None:
        return default
    try:
        s = str(val).strip()
        if not s or s.lower() in ("true", "false", "none", "null"):
            return default
        return float(s)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int) -> int:
    """Safely converts an environment variable to int, avoiding crashes on 'true', empty, or malformed strings."""
    if val is None:
        return default
    try:
        s = str(val).strip()
        if not s or s.lower() in ("true", "false", "none", "null"):
            return default
        return int(s)
    except (ValueError, TypeError):
        return default


def _get_london_tz():
    """Returns the Europe/London tzinfo (falls back to UTC only if no tz database is available)."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/London")
    except Exception as e:  # pragma: no cover - only hit on systems without tz data
        logger.warning("Europe/London timezone data unavailable (%s); falling back to UTC.", e)
        return timezone.utc


# Single source of truth for "today" across the pipeline (dedupe marker, snapshots, top-up expiry).
LONDON_TZ = _get_london_tz()


def london_date() -> date:
    """Returns today's calendar date in Europe/London."""
    return datetime.now(LONDON_TZ).date()


def resolve_base_url(env_suffix: str = "") -> str:
    """Resolves the Trading 212 API base URL from T212_BASE_URL or T212_ENV (live vs demo)."""
    explicit_url = os.getenv(f"T212_BASE_URL{env_suffix}") or os.getenv("T212_BASE_URL")
    if explicit_url and explicit_url.strip():
        return explicit_url.strip()
    t212_env = (os.getenv(f"T212_ENV{env_suffix}") or os.getenv("T212_ENV") or "live").strip().lower()
    if t212_env in ("demo", "practice", "paper"):
        return "https://demo.trading212.com/api/v0"
    return "https://live.trading212.com/api/v0"


def _parse_cred_pair(key_id_raw: str, secret_raw: str, combined_raw: str = "") -> Dict[str, str]:
    """Parses Trading 212 Key ID and Secret (supporting both split and 'key:secret' formats)."""
    key_id = (key_id_raw or "").strip()
    secret = (secret_raw or "").strip()

    if combined_raw and not (key_id and secret):
        if ":" in combined_raw:
            parts = combined_raw.split(":", 1)
            key_id, secret = parts[0].strip(), parts[1].strip()
        else:
            key_id = combined_raw.strip()
            secret = ""
    elif ":" in key_id and not secret:
        parts = key_id.split(":", 1)
        key_id, secret = parts[0].strip(), parts[1].strip()

    return {"key_id": key_id, "secret": secret}


def _account_env_tag(name: str) -> str:
    """Env-var suffix for an account name, e.g. 'Main' -> 'MAIN', 'My ISA' -> 'MY_ISA'."""
    return str(name or "").strip().upper().replace(" ", "_")


def _parse_ticker_set(raw: Optional[str], default: Set[str]) -> Set[str]:
    """Parses a comma-separated ticker list into an uppercased set.

    Unset or blank values return a copy of `default` (GitHub Actions renders missing secrets as
    empty strings); the literal 'none' returns an empty set to clear the list explicitly.
    """
    if raw is None or not str(raw).strip():
        return set(default)
    text = str(raw).strip()
    if text.lower() in ("none", "null"):
        return set()
    return {t.strip().upper() for t in text.split(",") if t.strip()}


def _parse_iso_date(val: Any) -> Optional[date]:
    """Parses 'YYYY-MM-DD' (or a date) safely; returns None when missing or malformed."""
    if isinstance(val, date):
        return val
    if val is None:
        return None
    try:
        return date.fromisoformat(str(val).strip()[:10])
    except (ValueError, TypeError):
        return None


def _lookup_account_entry(table: Any, account_name: str) -> Any:
    """Returns table[account_name] (exact match first, then case-insensitive); None when absent."""
    if not isinstance(table, dict):
        return None
    if account_name in table:
        return table[account_name]
    wanted = str(account_name or "").strip().lower()
    for key, value in table.items():
        if str(key).strip().lower() == wanted:
            return value
    return None


# --- OPTIONAL: TEMPORARY CASH-RESERVE TOP-UPS ---
# Raise an account's protected reserve for a limited time, e.g. while money you plan to withdraw sits in
# the account as free cash. While a top-up is active its cash is never used for buys. Set
#   RESERVE_TOPUPS_JSON='{"Main": {"amount_gbp": 2000, "until": "2026-12-31"}}'
# Keys are account names (ACCOUNT_NAME / SECOND_ACCOUNT_NAME). The reserve is raised by `amount_gbp` while the
# Europe/London date is <= `until` (inclusive); afterwards the top-up lapses automatically. Default: no top-ups.
_DEFAULT_RESERVE_TOPUPS: Dict[str, Dict[str, Any]] = {}


def _parse_reserve_topups(
    raw: Optional[str],
    default: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Parses RESERVE_TOPUPS_JSON; returns a deep copy of `default` when unset, blank or invalid.

    Malformed individual entries are ignored (with a warning) when the accounts are built, so one bad
    entry never disables the others. The literal 'none' clears all top-ups.
    """
    if raw is None or not str(raw).strip():
        return copy.deepcopy(default)
    text = str(raw).strip()
    if text.lower() in ("none", "null"):
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as e:
        logger.warning("RESERVE_TOPUPS_JSON is not valid JSON (%s); keeping the default reserve top-ups.", e)
        return copy.deepcopy(default)
    if not isinstance(parsed, dict) or not all(isinstance(k, str) for k in parsed):
        logger.warning(
            "RESERVE_TOPUPS_JSON must be an object mapping account name -> "
            "{\"amount_gbp\": <number>, \"until\": \"YYYY-MM-DD\"}; keeping the default reserve top-ups."
        )
        return copy.deepcopy(default)
    return parsed


RESERVE_TOPUPS: Dict[str, Dict[str, Any]] = _parse_reserve_topups(
    os.getenv("RESERVE_TOPUPS_JSON"), _DEFAULT_RESERVE_TOPUPS
)


def _apply_reserve_topups(accounts: Dict[str, Dict[str, Any]], today: date, topups: Any) -> None:
    """Applies active temporary reserve top-ups in place (records the base / top-up breakdown on every account)."""
    for name, cfg in accounts.items():
        base_reserve = cfg["min_cash_reserve_gbp"]
        cfg["base_min_cash_reserve_gbp"] = base_reserve
        cfg["reserve_topup_gbp"] = 0.0
        cfg["reserve_topup_until"] = None

        topup = _lookup_account_entry(topups, name)
        if not isinstance(topup, dict):
            if topup is not None:
                logger.warning("Ignoring malformed reserve top-up for %s: %r", name, topup)
            continue
        amount = _safe_float(topup.get("amount_gbp"), 0.0)
        if amount <= 0:
            continue
        until = _parse_iso_date(topup.get("until"))
        if until is None:
            # Fail safe: the topped-up cash must never be deployed, so an unreadable expiry keeps
            # the top-up active rather than silently releasing the cash to buys.
            logger.warning(
                "Reserve top-up for %s has an invalid 'until' (%r); applying it until the date is fixed.",
                name, topup.get("until"),
            )
            active = True
        else:
            active = today <= until
        if active:
            cfg["min_cash_reserve_gbp"] = base_reserve + amount
            cfg["reserve_topup_gbp"] = amount
            cfg["reserve_topup_until"] = until.isoformat() if until else None


def _build_accounts(
    today: Optional[date] = None,
    topups: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Builds account configuration from simple single-user or multi-account env vars.

    Args:
        today: Europe/London calendar date used to decide whether a reserve top-up is active (default: today).
        topups: Top-up table (default: RESERVE_TOPUPS), e.g. {"Main": {"amount_gbp": 2000.0, "until": "2026-12-31"}}.

    Returns:
        ACCOUNTS mapping. While a top-up is active, min_cash_reserve_gbp = base reserve + amount_gbp;
        base_min_cash_reserve_gbp / reserve_topup_gbp / reserve_topup_until record the breakdown.
    """
    if today is None:
        today = london_date()
    if topups is None:
        topups = RESERVE_TOPUPS

    accounts: Dict[str, Dict[str, Any]] = {}

    primary_name = (os.getenv("ACCOUNT_NAME") or "Main").strip() or "Main"
    upper_tag = _account_env_tag(primary_name)

    primary_creds = _parse_cred_pair(
        os.getenv("T212_KEY_ID") or os.getenv(f"T212_KEY_ID_{upper_tag}", ""),
        os.getenv("T212_SECRET") or os.getenv(f"T212_SECRET_{upper_tag}", ""),
        os.getenv("T212_API_KEY") or os.getenv(f"T212_API_KEY_{upper_tag}", ""),
    )
    default_reserve = _safe_float(os.getenv("MIN_CASH_RESERVE_GBP"), 1500.0)

    accounts[primary_name] = {
        **primary_creds,
        "base_url": resolve_base_url(),
        "manual_cash": _safe_float(os.getenv("MANUAL_CASH") or os.getenv(f"MANUAL_CASH_{upper_tag}"), 0.0),
        "cash_currency": (os.getenv("CASH_CURRENCY") or os.getenv(f"CASH_CURRENCY_{upper_tag}") or "GBP").strip(),
        "min_cash_reserve_gbp": _safe_float(
            os.getenv(f"MIN_CASH_RESERVE_{upper_tag}"),
            default_reserve,
        ),
    }

    # Optional second account support (for partners / ISA + Invest split)
    second_name = (os.getenv("SECOND_ACCOUNT_NAME") or "").strip()
    second_creds = _parse_cred_pair(
        os.getenv("T212_KEY_ID_2") or (os.getenv(f"T212_KEY_ID_{second_name.upper()}", "") if second_name else ""),
        os.getenv("T212_SECRET_2") or (os.getenv(f"T212_SECRET_{second_name.upper()}", "") if second_name else ""),
        os.getenv("T212_API_KEY_2", ""),
    )
    if second_name or (second_creds["key_id"] and second_creds["secret"]):
        sec_label = second_name or "Secondary"
        sec_tag = _account_env_tag(sec_label)
        accounts[sec_label] = {
            **second_creds,
            "base_url": resolve_base_url("_2"),
            "manual_cash": _safe_float(os.getenv("MANUAL_CASH_2") or os.getenv(f"MANUAL_CASH_{sec_tag}"), 0.0),
            "cash_currency": (os.getenv("CASH_CURRENCY_2") or os.getenv(f"CASH_CURRENCY_{sec_tag}") or "GBP").strip(),
            "min_cash_reserve_gbp": _safe_float(
                os.getenv("MIN_CASH_RESERVE_2") or os.getenv(f"MIN_CASH_RESERVE_{sec_tag}"),
                500.0,
            ),
        }

    _apply_reserve_topups(accounts, today, topups)
    return accounts


ACCOUNTS: Dict[str, Dict[str, Any]] = _build_accounts()

# --- OPTIONAL: ACCOUNT TAX WRAPPERS (informational only; no tax logic depends on this) ---
# Set ACCOUNT_WRAPPER_<NAME>=ISA (or GIA, SIPP, ...) per account, e.g. ACCOUNT_WRAPPER_MAIN=ISA. Following the
# single/second-account convention above, ACCOUNT_WRAPPER (first account) and ACCOUNT_WRAPPER_2 (second account)
# also work. When every account is an ISA, the dashboard footer notes that trades carry no CGT. Default: not set.
_DEFAULT_ACCOUNT_WRAPPERS: Dict[str, str] = {}
_POSITIONAL_WRAPPER_ENV = ("ACCOUNT_WRAPPER", "ACCOUNT_WRAPPER_2")


def _build_account_wrappers(
    defaults: Optional[Dict[str, str]] = None,
    account_names: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """Returns {account: wrapper} from env overrides, falling back to `defaults` (blank env = default).

    Lookup order per account: ACCOUNT_WRAPPER_<NAME>, then the positional ACCOUNT_WRAPPER / ACCOUNT_WRAPPER_2
    (first / second entry of `account_names`), then `defaults`.

    Args:
        defaults: Default wrappers per account name (default: _DEFAULT_ACCOUNT_WRAPPERS, empty).
        account_names: Configured account names in order (default: the ACCOUNTS keys).
    """
    if defaults is None:
        defaults = _DEFAULT_ACCOUNT_WRAPPERS
    ordered = [n for n in (list(ACCOUNTS.keys()) if account_names is None else list(account_names)) if n]
    names: List[str] = []
    for name in ordered + list(defaults.keys()):
        if name and name not in names:
            names.append(name)
    wrappers: Dict[str, str] = {}
    for name in names:
        candidates = [f"ACCOUNT_WRAPPER_{_account_env_tag(name)}", f"ACCOUNT_WRAPPER_{str(name).upper()}"]
        if name in ordered and ordered.index(name) < len(_POSITIONAL_WRAPPER_ENV):
            candidates.append(_POSITIONAL_WRAPPER_ENV[ordered.index(name)])
        override = next((os.getenv(k).strip() for k in candidates if (os.getenv(k) or "").strip()), "")
        if override:
            wrappers[name] = override.upper()
        elif defaults.get(name):
            wrappers[name] = str(defaults[name]).strip().upper()
    return wrappers


ACCOUNT_WRAPPERS: Dict[str, str] = _build_account_wrappers()

# --- PENDING ORDERS OVERRIDE (For Read-Only API Keys without 'Orders' scope) ---
# Comma-separated list of symbols with pending orders in Trading 212 (e.g. "RR, AMZN, BRK-B")
PENDING_SELLS_ENV = os.getenv("PENDING_SELLS", "")
PENDING_BUYS_ENV = os.getenv("PENDING_BUYS", "")

PENDING_SELLS_CONFIG = [s.strip() for s in PENDING_SELLS_ENV.split(",") if s.strip()]
PENDING_BUYS_CONFIG = [b.strip() for b in PENDING_BUYS_ENV.split(",") if b.strip()]

# --- EMAIL (SMTP) CREDENTIALS ---
SMTP_HOST = os.getenv("SMTP_HOST") or "smtp.gmail.com"
SMTP_PORT = _safe_int(os.getenv("SMTP_PORT"), 587)
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

RECIPIENTS_STR = os.getenv("ALERT_EMAILS", "") or SMTP_USER
RECIPIENT_EMAILS: List[str] = [
    email.strip() for email in RECIPIENTS_STR.split(",") if email.strip()
]

# --- STRATEGY PARAMETERS (5-YEAR ALL-WEATHER & CRASH-PROTECTED HORIZON) ---
STRATEGY_CONFIG = {
    "ema_short": 20,
    "ema_medium": 50,
    "ema_long": 200,
    "rsi_period": 14,
    "rsi_overbought": 82,
    "rsi_oversold": 35,
    "atr_period": 14,
    "atr_multiplier_stop": 3.0,
    "extension_trim_threshold": 0.25,
    "investment_horizon_years": 5,
}


def _parse_fraction(raw: Optional[str], default: float, env_name: str) -> float:
    """Parses a 0..1 fraction from the environment (e.g. '0.08' = 8%); blank/invalid/out-of-range -> default."""
    if raw is None or not str(raw).strip():
        return default
    val = _safe_float(raw, float("nan"))
    if not (0.0 <= val <= 1.0):  # also rejects NaN
        logger.warning("%s=%r is not a fraction between 0 and 1; using the default %.2f.", env_name, raw, default)
        return default
    return val


# --- LEVERAGED-ETF POLICY ---
# Maximum weight in 3x daily-leveraged ETFs (QQQ3, 3LUS, 3SEM, ...). Holdings above the cap are trimmed back to it
# and leveraged ETFs are never bought. Set MAX_LEVERAGED_ETF_PCT=0 for a "no 3x ETFs" policy: any 3x ETF you hold
# is then flagged as a full SELL (exit). Default: 0.08 (8%).
_DEFAULT_MAX_LEVERAGED_ETF_PCT = 0.08

# --- CAPITAL ALLOCATION & REBALANCING ---
REBALANCE_CONFIG = {
    "min_cash_reserve_gbp": _safe_float(os.getenv("MIN_CASH_RESERVE_GBP"), 1500.0),  # Protected cash withdrawal reserve
    "min_cash_buffer_pct": 0.05,    # 5% additional buffer above reserve
    "max_position_pct": 0.15,       # Maximum 15% of total portfolio in one asset
    "min_order_gbp": 300.0,         # Minimum tranche size in GBP
    "max_new_ideas_to_buy": 2,      # Limit concurrent new entries to keep portfolio focused
    # Target >=5% in real defensive assets only: cash, money-market funds, physical gold and short-dated gilts
    # (aligned with the 5% household defensive target; low-beta stocks and equity index funds do not count)
    "target_crash_shield_pct": max(0.0, min(1.0, _safe_float(os.getenv("TARGET_CRASH_SHIELD_PCT"), 0.05))),
    # Maximum cap on 3x daily leveraged instruments (0.0 = exit them entirely; see LEVERAGED-ETF POLICY above)
    "max_leveraged_etf_pct": _parse_fraction(
        os.getenv("MAX_LEVERAGED_ETF_PCT"), _DEFAULT_MAX_LEVERAGED_ETF_PCT, "MAX_LEVERAGED_ETF_PCT"
    ),
    "max_cluster_pct": 0.24,         # Maximum 24% of combined wealth in one correlated macro cluster
    "cap_applies_to_broad_etfs": False,  # max_position_pct is not enforced on broad index ETFs (e.g. VWRP, VUAA)
}


def _build_allocation_config() -> Dict[str, Any]:
    """Household strategic allocation, with safe env overrides.

    The household = all configured Trading 212 accounts + vested outside holdings, treated as one portfolio.
    New money is steered to the bucket furthest below target; profitable holdings are never sold to rebalance.
    Targets that do not sum to 1.0 are normalised (with a warning); an all-zero/invalid set falls back to 75/20/5.
    """
    defaults = {"core": 0.75, "satellite": 0.20, "defensive": 0.05}
    targets = {
        "core": _safe_float(os.getenv("TARGET_CORE_PCT"), defaults["core"]),
        "satellite": _safe_float(os.getenv("TARGET_SATELLITE_PCT"), defaults["satellite"]),
        "defensive": _safe_float(os.getenv("TARGET_DEFENSIVE_PCT"), defaults["defensive"]),
    }
    targets = {k: max(0.0, v) for k, v in targets.items()}
    total = sum(targets.values())
    if total <= 0:
        logger.warning("Household allocation targets are invalid (sum %.3f); using the 75/20/5 default.", total)
        targets = dict(defaults)
    elif abs(total - 1.0) > 1e-6:
        logger.warning("Household allocation targets sum to %.3f, not 1.0; normalising them.", total)
        targets = {k: v / total for k, v in targets.items()}

    cadence = (os.getenv("BUY_CADENCE") or "monthly").strip().lower()
    if cadence not in ("monthly", "daily"):
        logger.warning("BUY_CADENCE=%r is not 'monthly' or 'daily'; using 'monthly'.", cadence)
        cadence = "monthly"

    gold_share = min(1.0, max(0.0, _safe_float(os.getenv("DEFENSIVE_GOLD_SHARE"), 1.0)))

    return {
        "targets": targets,
        # Rebalancing band in percentage points: a bucket further than this from target is "out of band" and
        # gets new money first
        "band_pp": max(0.0, _safe_float(os.getenv("ALLOCATION_BAND_PP"), 5.0)),
        # Max household weight for any single satellite stock (new buys only; existing positions are not trimmed by it)
        "satellite_max_single_pct": max(0.0, _safe_float(os.getenv("SATELLITE_MAX_SINGLE_PCT"), 0.05)),
        # Defensive sleeve split: default 100% physical gold (SGLN.L, accumulating) / 0% new short gilts (env DEFENSIVE_GOLD_SHARE);
        # existing cash, money-market and short-gilt holdings still count towards the defensive bucket
        "defensive_split": {"gold": round(gold_share, 6), "short_gilts": round(1.0 - gold_share, 6)},
        # GBP-quoted lines for new money (no 0.15% FX fee)
        "core_buy_ticker": (os.getenv("CORE_BUY_TICKER") or "VWRP.L").strip().upper(),
        "gold_buy_ticker": (os.getenv("GOLD_BUY_TICKER") or "SGLN.L").strip().upper(),
        "short_gilt_buy_ticker": (os.getenv("SHORT_GILT_BUY_TICKER") or "IGLS.L").strip().upper(),
        # US big-tech cap as a fraction of household wealth (fund look-through + vested outside shares included)
        "big_tech_cap_pct": max(0.0, _safe_float(os.getenv("BIG_TECH_CAP_PCT"), 0.30)),
        # Buys only on the first London business day of the month, unless deployable cash >= off_cycle_min_deploy_gbp
        # (sells and risk alerts are always daily). 'daily' restores daily buying.
        "buy_cadence": cadence,
        "off_cycle_min_deploy_gbp": max(0.0, _safe_float(os.getenv("OFF_CYCLE_MIN_DEPLOY_GBP"), 2000.0)),
    }


ALLOCATION_CONFIG: Dict[str, Any] = _build_allocation_config()

# --- US BIG TECH (household concentration cap) ---
BIG_TECH_TICKERS: Set[str] = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AVGO", "TSLA"}
# Approximate share of each fund held in BIG_TECH_TICKERS (fund look-through), keyed by ticker root.
# APPROXIMATIONS as of Oct 2026 from published index weights: review once a year.
# QQQ3 is 3x daily-leveraged Nasdaq-100 exposure, hence 1.5.
BIG_TECH_LOOKTHROUGH: Dict[str, float] = {
    "VWRP": 0.22, "VWRA": 0.22, "VWRL": 0.22,                  # FTSE All-World
    "VUAA": 0.35, "VUAG": 0.35, "VUSA": 0.35, "CSPX": 0.35,    # S&P 500
    "CNDX": 0.50, "CNX1": 0.50, "EQQQ": 0.50,                  # Nasdaq-100
    "SMGB": 0.30,                                              # VanEck Semiconductor
    "QQQ3": 1.50,                                              # 3x Nasdaq-100
}

# --- MODEL VERSION (stamped on every recommendation-ledger record and snapshot) ---
MODEL_VERSION = "2026.10.27-max-compounding"

# --- OPTIONAL EXCLUDED TICKERS ---
# Set EXCLUDED_TICKERS="TSLA,COIN" in .env if you want the engine to never buy specific tickers.
EXCLUDED_TICKERS: Set[str] = _parse_ticker_set(os.getenv("EXCLUDED_TICKERS"), set())

# --- OPTIONAL: EARMARKED TICKERS (money with a fixed purpose; excluded from the investable book) ---
# e.g. EARMARKED_TICKERS="CSH2.L" for a money-market fund holding savings you will need soon. Earmarked holdings
# are never traded and are left out of position weights, cluster caps, risk metrics and projections.
# Blank/unset = no earmarks; "none" = no earmarks.
_DEFAULT_EARMARKED_TICKERS: Set[str] = set()
EARMARKED_TICKERS: Set[str] = _parse_ticker_set(os.getenv("EARMARKED_TICKERS"), _DEFAULT_EARMARKED_TICKERS)

# --- OPTIONAL: OUTSIDE HOLDINGS (held outside Trading 212; count towards concentration & cluster caps) ---
# e.g. shares held at another broker or through a workplace share plan. They are never traded; vested value counts
# towards single-name and cluster caps, unvested value is shown for information only. Set OUTSIDE_HOLDINGS_JSON to
#   {"Main": [{"ticker": "MSFT", "shares_vested": 10, "shares_unvested": 5, "currency": "USD"}]}
# Optional per-holding keys: "macro_cluster", "value_hint_usd_vested", "value_hint_usd_unvested" (used if the live
# price lookup fails) and "as_of". Keys must match your account names exactly. Invalid JSON keeps the default
# (none) and logs a warning.
_DEFAULT_OUTSIDE_HOLDINGS: Dict[str, List[Dict[str, Any]]] = {}


def _parse_outside_holdings(
    raw: Optional[str],
    default: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Parses OUTSIDE_HOLDINGS_JSON; returns a deep copy of `default` when unset, blank or invalid."""
    if raw is None or not str(raw).strip():
        return copy.deepcopy(default)
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as e:
        logger.warning("OUTSIDE_HOLDINGS_JSON is not valid JSON (%s); keeping the default outside holdings.", e)
        return copy.deepcopy(default)
    valid = isinstance(parsed, dict) and all(
        isinstance(acc, str)
        and isinstance(items, list)
        and all(isinstance(item, dict) and str(item.get("ticker") or "").strip() for item in items)
        for acc, items in parsed.items()
    )
    if not valid:
        logger.warning(
            "OUTSIDE_HOLDINGS_JSON must be an object mapping account -> list of holdings with a 'ticker'; "
            "keeping the default outside holdings."
        )
        return copy.deepcopy(default)
    return parsed


OUTSIDE_HOLDINGS: Dict[str, List[Dict[str, Any]]] = _parse_outside_holdings(
    os.getenv("OUTSIDE_HOLDINGS_JSON"), _DEFAULT_OUTSIDE_HOLDINGS
)

# --- CURATED 5-YEAR HIGH-SORTINO & CRASH-PROTECTED WATCHLIST ---
# Data-driven screening for highest 5-year Sortino ratio, lowest default risk (Altman Z),
# superior forensic earnings quality (Piotroski F-Score >= 8), and crisis-alpha crash protection.
WATCHLIST = [
    # Core All-Weather Foundations & Crisis Alpha Shields
    {"ticker": "SGLN.L", "name": "iShares Physical Gold ETC", "theme": "Physical Gold & Crisis Alpha Shield"},
    {"ticker": "IGLS.L", "name": "iShares UK Gilts 0-5yr ETF", "theme": "UK Short-Dated Gilts (Defensive Ballast)"},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway", "theme": "Fortress Cash Ballast & Underwriting"},
    {"ticker": "VWRP.L", "name": "Vanguard FTSE All-World ETF", "theme": "Global Equities Core Foundation"},
    # GBP-quoted lines (no 0.15% FX fee): VUAG.L = S&P 500 (sister of USD-line VUAA.L), CNX1.L = Nasdaq-100 (sister of CNDX.L)
    {"ticker": "VUAG.L", "name": "Vanguard S&P 500 ETF (GBP)", "theme": "US Large-Cap S&P 500 Secular Core"},
    {"ticker": "CNX1.L", "name": "iShares Nasdaq 100 (GBP)", "theme": "Nasdaq-100 High-Beta Tech Leaders"},
    # AI Compute, Foundry & Semiconductor Monopolies
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "theme": "AI Compute & Full-Stack Data Center Monopoly"},
    {"ticker": "TSM", "name": "Taiwan Semi", "theme": "AI Foundry Monopoly"},
    {"ticker": "ASML", "name": "ASML Holding", "theme": "EUV Lithography Monopoly"},
    {"ticker": "AVGO", "name": "Broadcom", "theme": "AI Networking & ASIC Silicon"},
    # Cloud Hyperscalers & Digital Consumer Ecosystems
    {"ticker": "MSFT", "name": "Microsoft", "theme": "Enterprise AI & Azure Compounder"},
    {"ticker": "AMZN", "name": "Amazon.com", "theme": "AWS Cloud Infrastructure & Logistics Moat"},
    {"ticker": "META", "name": "Meta Platforms", "theme": "AI Monetization & Free Cash Flow"},
    {"ticker": "AAPL", "name": "Apple", "theme": "Consumer Tech & Apple Intelligence"},
    {"ticker": "COST", "name": "Costco Wholesale", "theme": "Membership Retail & Pricing Power Fortress"},
    # Healthcare, Biopharma & MedTech Compounders
    {"ticker": "LLY", "name": "Eli Lilly", "theme": "Pharma / GLP-1 Mega-Cycle"},
    {"ticker": "ABBV", "name": "AbbVie Inc.", "theme": "Immunology & Dividend Cash Fortress"},
    {"ticker": "ISRG", "name": "Intuitive Surgical", "theme": "Robotic Surgery & Medical Innovation"},
    # Financial Tollbooths, Industrial Gases & Power Infrastructure
    {"ticker": "V", "name": "Visa Inc.", "theme": "Global Financial Rails & Tollbooth Moat"},
    {"ticker": "MA", "name": "Mastercard Inc.", "theme": "Global Payment Network & Cross-Border Tollbooth"},
    {"ticker": "SPGI", "name": "S&P Global", "theme": "Credit Ratings & Benchmarks Oligopoly"},
    {"ticker": "LIN", "name": "Linde plc", "theme": "Industrial Gases & Take-or-Pay Contract Monopoly"},
    {"ticker": "ETN", "name": "Eaton Corporation", "theme": "Power Infrastructure & Grid Electrification"},
    {"ticker": "BA.L", "name": "BAE Systems", "theme": "Global Aerospace & Sovereign Defense Backlog"},
]
