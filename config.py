"""Configuration module for Trading 212 Daily Recommendation & Rebalancing Engine."""

import os
from pathlib import Path
from typing import Dict, Any, List, Set


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


def _build_accounts() -> Dict[str, Dict[str, Any]]:
    """Builds account configuration dictionary from simple single-user or multi-account env vars."""
    accounts: Dict[str, Dict[str, Any]] = {}

    primary_name = (os.getenv("ACCOUNT_NAME") or "Main").strip() or "Main"
    upper_tag = primary_name.upper().replace(" ", "_")

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
        sec_tag = sec_label.upper().replace(" ", "_")
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

    return accounts


ACCOUNTS: Dict[str, Dict[str, Any]] = _build_accounts()

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

# --- CAPITAL ALLOCATION & REBALANCING ---
REBALANCE_CONFIG = {
    "min_cash_reserve_gbp": _safe_float(os.getenv("MIN_CASH_RESERVE_GBP"), 1500.0),  # Protected cash withdrawal reserve
    "min_cash_buffer_pct": 0.05,    # 5% additional buffer above reserve
    "max_position_pct": 0.15,       # Maximum 15% of total portfolio in one asset
    "min_order_gbp": 300.0,         # Minimum tranche size in GBP
    "max_new_ideas_to_buy": 2,      # Limit concurrent new entries to keep portfolio focused
    "target_crash_shield_pct": 0.15, # Target >=15% in uncorrelated / low-beta crash shields (Gold, Berkshire, Healthcare, Cash)
    "max_leveraged_etf_pct": 0.08,   # Maximum 8% cap on 3x daily leveraged instruments to eliminate crash wipeout risk
}

# --- OPTIONAL EXCLUDED TICKERS ---
# Set EXCLUDED_TICKERS="TSLA,COIN" in .env if you want the engine to never buy specific tickers.
_excluded_raw = os.getenv("EXCLUDED_TICKERS", "")
EXCLUDED_TICKERS: Set[str] = {
    t.strip().upper() for t in _excluded_raw.split(",") if t.strip()
}

# --- CURATED 5-YEAR HIGH-SORTINO & CRASH-PROTECTED WATCHLIST ---
# Data-driven screening for highest 5-year Sortino ratio, lowest default risk (Altman Z),
# superior forensic earnings quality (Piotroski F-Score >= 8), and crisis-alpha crash protection.
WATCHLIST = [
    # Core All-Weather Foundations & Crisis Alpha Shields
    {"ticker": "SGLN.L", "name": "iShares Physical Gold ETC", "theme": "Physical Gold & Crisis Alpha Shield"},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway", "theme": "Fortress Cash Ballast & Underwriting"},
    {"ticker": "VWRP.L", "name": "Vanguard FTSE All-World ETF", "theme": "Global Equities Core Foundation"},
    {"ticker": "VUAA.L", "name": "Vanguard S&P 500 UCITS ETF", "theme": "US Large-Cap Core ETF (London)"},
    {"ticker": "CNDX.L", "name": "iShares Nasdaq 100", "theme": "US Large-Cap Core ETF (London)"},
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
