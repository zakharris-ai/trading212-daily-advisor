"""Event radar: scheduled market events and your holdings' results dates, and what (if anything) to do about them.

Evidence behind the guidance (deliberately conservative):
  * Company results: on average shares earn slightly MORE around scheduled results, not less (the "earnings
    announcement premium": Frazzini & Lamont 2007; Savor & Wilson 2016), so selling because results are due costs
    return on average. What results add is single-stock gap risk. Hence: no new buys in the 5 days before results (the
    engine's pre-earnings gate), hold existing positions through them, and place trims that are due anyway BEFORE the
    report.
  * US midterm elections: the S&P 500 was higher 12 months after all 19 midterms since 1950 (mean +15.4%, worst +2.0%;
    all 12-month windows since 1950: 75% positive, mean +9.4%; computed from ^GSPC closes, 1 Nov 2026). Nineteen data
    points are too few to trade on, but they argue against selling ahead of the vote.
  * Central-bank and data days (Fed, Bank of England, CPI, jobs): no reliable edge from trading around them; they
    mainly raise day-to-day volatility. The plan never trades on them.

Macro dates were verified on 1 Nov 2026 against the official calendars: federalreserve.gov (FOMC, updated
28 Oct 2026), bankofengland.co.uk (MPC dates), bls.gov (CPI and Employment Situation schedules) and the ONS release
calendar (UK CPI). Extend MACRO_SCHEDULE once a year when the agencies publish the next year's dates.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - every supported Python ships zoneinfo
    ZoneInfo = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LONDON = "Europe/London"
NEW_YORK = "America/New_York"
DEFAULT_HORIZON_DAYS = 14

# --- Recurring scheduled events -------------------------------------------------------------------------------
# 'time' is local to 'tz' (converted to UK time for each date, so US daylight-saving gaps are handled).
MACRO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "fomc": {
        "kind": "central_bank", "title": "US Federal Reserve interest-rate decision", "time": "14:00", "tz": NEW_YORK,
        "impact": "high", "what_to_do": "Nothing.",
        "why": "Fed days can move tech shares sharply either way; there is no reliable edge from trading around them.",
    },
    "boe": {
        "kind": "central_bank", "title": "Bank of England interest-rate decision", "time": "12:00", "tz": LONDON,
        "impact": "medium", "what_to_do": "Nothing.",
        "why": "Bank Rate moves the yield on cash, money-market funds and short gilts much more than share prices.",
    },
    "us_cpi": {
        "kind": "data", "title": "US CPI inflation", "time": "08:30", "tz": NEW_YORK, "impact": "high",
        "ref_month": True, "what_to_do": "Nothing.",
        "why": "Inflation surprises move interest-rate expectations, which hit growth and tech shares hardest; "
               "no reliable edge from trading around the release.",
    },
    "us_jobs": {
        "kind": "data", "title": "US jobs report", "time": "08:30", "tz": NEW_YORK, "impact": "medium",
        "ref_month": True, "what_to_do": "Nothing.",
        "why": "Can move markets 1-2% on the day; the plan never trades on data releases.",
    },
    "uk_cpi": {
        "kind": "data", "title": "UK CPI inflation", "time": "07:00", "tz": LONDON, "impact": "low",
        "ref_month": True, "what_to_do": "Nothing.",
        "why": "Mainly moves gilts and the pound; a stronger pound slightly lowers the £ value of US holdings.",
    },
}

# Announcement dates (FOMC: the second day of the meeting, when the statement is released).
MACRO_SCHEDULE: Dict[str, Tuple[str, ...]] = {
    "fomc": ("2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15",
             "2027-10-27", "2027-12-08"),
    "boe": ("2026-11-05", "2026-12-17", "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-17", "2027-07-29",
            "2027-09-16", "2027-11-04", "2027-12-16"),
    "us_cpi": ("2026-11-10", "2026-12-10", "2027-01-13", "2027-02-11", "2027-03-10", "2027-04-13", "2027-05-12",
               "2027-06-10", "2027-07-14", "2027-08-11", "2027-09-14", "2027-10-14", "2027-11-10", "2027-12-10"),
    "us_jobs": ("2026-11-06", "2026-12-04", "2027-01-08", "2027-02-05", "2027-03-05", "2027-04-02", "2027-05-07",
                "2027-06-04", "2027-07-02", "2027-08-06", "2027-09-03", "2027-10-08", "2027-11-05", "2027-12-03"),
    "uk_cpi": ("2026-11-18",),
}

# One-off events. 'lead_days' shows an event earlier than the normal horizon (e.g. tax-year deadlines).
ONE_OFF_EVENTS: Tuple[Dict[str, Any], ...] = (
    {
        "date": "2026-11-03", "kind": "election", "title": "US midterm elections", "impact": "high",
        "when_label": "results overnight, UK early Wednesday",
        "what_to_do": "Nothing. Hold, and don't trade on the result.",
        "why": "The S&P 500 was higher 12 months after all 19 midterms since 1950 (average +15%, worst +2%), but "
               "19 data points are too few to bet on. Expect a day or two of bigger swings; the monthly "
               "circuit-breaker handles a real downtrend.",
    },
    {
        "date": "2026-11-26", "kind": "market_closed", "title": "US markets closed (Thanksgiving)", "impact": "low",
        "what_to_do": "US orders wait for the next session.",
        "why": "London is open as normal. On Friday 27 Nov US markets close early, at 18:00 UK.",
    },
    {
        "date": "2026-12-24", "kind": "market_closed", "title": "Christmas Eve: short trading day", "impact": "low",
        "what_to_do": "Avoid placing orders; liquidity is thin.",
        "why": "London closes at 12:30 and US markets at 18:00 UK.",
    },
    {
        "date": "2026-12-25", "kind": "market_closed", "title": "Christmas: London and US markets closed",
        "impact": "low", "what_to_do": "Nothing.", "why": "",
    },
    {
        "date": "2026-12-31", "kind": "market_closed", "title": "New Year's Eve: London closes at 12:30", "impact": "low",
        "what_to_do": "Avoid placing London orders in the afternoon.", "why": "",
    },
    {
        "date": "2027-01-01", "kind": "market_closed", "title": "New Year's Day: London and US markets closed",
        "impact": "low", "what_to_do": "Nothing.", "why": "",
    },
    {
        "date": "2027-04-05", "kind": "tax", "title": "Last day of the 2026/27 tax year", "impact": "medium",
        "lead_days": 45,
        "what_to_do": "Use this year's ISA allowance (£20,000 per person) before midnight; consider Bed & ISA for "
                      "shares held outside an ISA (e.g. vested RSUs).",
        "why": "Unused allowance is lost. From 6 April 2027 tax on savings interest outside ISAs rises by 2 points "
               "(Budget, 28 Oct 2026), which makes ISA space more valuable.",
    },
)

# Companies whose results move a whole part of a typical tech-heavy portfolio, not just their own shares.
BELLWETHER_NOTES: Dict[str, str] = {
    "NVDA": "Nvidia's results also move semiconductor and Nasdaq-100 funds (e.g. SMGB, CNDX) and chip stocks "
            "such as TSM, MRVL and ASML.",
}

_EXCHANGE_SUFFIX_HOURS: Dict[str, Tuple[str, str, str, str]] = {
    # suffix: (label, local open, local close, tz)
    ".L": ("London", "08:00", "16:30", LONDON),
    ".AS": ("Amsterdam", "09:00", "17:30", "Europe/Amsterdam"),
    ".PA": ("Paris", "09:00", "17:30", "Europe/Paris"),
    ".DE": ("Frankfurt (Xetra)", "09:00", "17:30", "Europe/Berlin"),
    ".SW": ("Zurich", "09:00", "17:20", "Europe/Zurich"),
}
_US_HOURS = ("US", "09:30", "16:00", NEW_YORK)


# --- Small helpers --------------------------------------------------------------------------------------------
def _parse_date(value: Any) -> Optional[date]:
    """date / datetime / 'YYYY-MM-DD' / ISO timestamp / 'Nov 02, 2026' / '2 Nov 2026' -> date (None if unparseable)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    s = str(value).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        pass
    for fmt in ("%b %d, %Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _to_uk_hhmm(d: date, hhmm: str, tz: str) -> Optional[str]:
    """Local 'HH:MM' on date d in tz -> 'HH:MM' UK time (DST-aware). None if it cannot be converted."""
    try:
        h, m = (int(x) for x in str(hhmm).split(":")[:2])
    except (ValueError, TypeError):
        return None
    if tz == LONDON:
        return f"{h:02d}:{m:02d}"
    if ZoneInfo is None:  # pragma: no cover
        return None
    try:
        local = datetime(d.year, d.month, d.day, h, m, tzinfo=ZoneInfo(tz))
        return local.astimezone(ZoneInfo(LONDON)).strftime("%H:%M")
    except Exception:  # pragma: no cover - tz database missing
        return None


def day_label(d: date, as_of: date) -> str:
    """'Today', 'Tomorrow' or e.g. 'Tue 3 Nov'."""
    if d == as_of:
        return "Today"
    if d == as_of + timedelta(days=1):
        return "Tomorrow"
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"


def _prev_month_name(d: date) -> str:
    first = d.replace(day=1)
    return (first - timedelta(days=1)).strftime("%B")


def _root(ticker: str) -> str:
    t = str(ticker or "").upper().strip()
    return t.split(".")[0] if "." in t else t


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def exchange_hours(ticker: str) -> Tuple[str, str, str, str]:
    """(label, local open, local close, tz) for a Yahoo-style ticker; US when there is no known suffix."""
    t = str(ticker or "").upper()
    for suffix, hours in _EXCHANGE_SUFFIX_HOURS.items():
        if t.endswith(suffix.upper()):
            return hours
    return _US_HOURS


def _shift_hhmm(hhmm: str, minutes: int) -> str:
    h, m = (int(x) for x in hhmm.split(":"))
    total = h * 60 + m + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


# --- Macro calendar -------------------------------------------------------------------------------------------
def upcoming_macro_events(as_of: date, horizon_days: int = DEFAULT_HORIZON_DAYS) -> List[Dict[str, Any]]:
    """Scheduled macro events from as_of (inclusive) to as_of + horizon_days (one-offs may set a longer lead)."""
    out: List[Dict[str, Any]] = []
    end = as_of + timedelta(days=max(0, int(horizon_days)))
    for key, dates in MACRO_SCHEDULE.items():
        tpl = MACRO_TEMPLATES.get(key)
        if not tpl:
            continue
        for ds in dates:
            d = _parse_date(ds)
            if d is None or d < as_of or d > end:
                continue
            title = tpl["title"] + (f" ({_prev_month_name(d)})" if tpl.get("ref_month") else "")
            out.append({
                "date": d.isoformat(), "kind": tpl["kind"], "title": title, "impact": tpl["impact"],
                "time_uk": _to_uk_hhmm(d, tpl["time"], tpl["tz"]), "what_to_do": tpl["what_to_do"],
                "why": tpl["why"], "action_needed": False, "source": key,
            })
    for ev in ONE_OFF_EVENTS:
        d = _parse_date(ev.get("date"))
        lead = int(ev.get("lead_days", horizon_days) or horizon_days)
        if d is None or d < as_of or d > as_of + timedelta(days=max(lead, horizon_days)):
            continue
        out.append({
            "date": d.isoformat(), "kind": ev["kind"], "title": ev["title"], "impact": ev.get("impact", "low"),
            "time_uk": ev.get("time_uk"), "when_label": ev.get("when_label"), "what_to_do": ev.get("what_to_do", ""),
            "why": ev.get("why", ""), "action_needed": ev.get("kind") == "tax", "source": "one_off",
        })
    return out


def _london_closures(as_of: date, horizon_days: int) -> List[Dict[str, Any]]:
    """England & Wales bank holidays in the window (the London Stock Exchange is closed)."""
    try:
        from strategy_engine import uk_bank_holidays
    except Exception:  # pragma: no cover - engine always importable in the app
        return []
    end = as_of + timedelta(days=horizon_days)
    days = set()
    for y in {as_of.year, end.year}:
        try:
            days |= {d for d in uk_bank_holidays(y) if as_of <= d <= end}
        except Exception:  # pragma: no cover
            continue
    static = {_parse_date(e["date"]) for e in ONE_OFF_EVENTS if e.get("kind") == "market_closed"}
    return [
        {"date": d.isoformat(), "kind": "market_closed", "title": "UK bank holiday: London Stock Exchange closed",
         "impact": "low", "time_uk": None, "what_to_do": "London orders wait for the next business day.", "why": "",
         "action_needed": False, "source": "uk_bank_holiday"}
        for d in sorted(days) if d not in static
    ]


# --- Plan events (circuit-breaker check, buy day) -------------------------------------------------------------
def _last_london_business_day(year: int, month: int) -> date:
    try:
        from strategy_engine import is_london_business_day
    except Exception:  # pragma: no cover
        def is_london_business_day(x: date) -> bool:  # type: ignore[misc]
            return x.weekday() < 5
    nxt = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    d = nxt - timedelta(days=1)
    while not is_london_business_day(d):
        d -= timedelta(days=1)
    return d


def _breaker_cut_text() -> str:
    try:
        from strategy_engine import BUCKET_SATELLITE, _allocation_cfg
        cfg = _allocation_cfg()
        floor = cfg.get("risk_off_satellite_pct")
        if cfg.get("rebalance_policy") == "off":
            return ""
        if floor is not None and floor < cfg["targets"][BUCKET_SATELLITE] - 1e-9:
            return f"cut satellites to {floor * 100.0:.0f}% of the household"
    except Exception:
        pass
    return "trim satellites above plan to cash"


def circuit_breaker_event(as_of: date, tech_regime: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The next month-end circuit-breaker check, with the Nasdaq-100 level that would flip it (when known)."""
    r = tech_regime if isinstance(tech_regime, dict) else {}
    state = r.get("regime")
    if state not in ("RISK_ON", "RISK_OFF"):
        return None
    check = _last_london_business_day(as_of.year, as_of.month)
    if check < as_of:  # pragma: no cover - as_of is never after the month's last business day in practice
        nxt = as_of.replace(day=28) + timedelta(days=4)
        check = _last_london_business_day(nxt.year, nxt.month)
    nxt_month = check.replace(day=28) + timedelta(days=4)
    try:
        from strategy_engine import first_london_business_day
        next_email = first_london_business_day(nxt_month.year, nxt_month.month)
    except Exception:  # pragma: no cover
        next_email = check + timedelta(days=1)
    proxy = str(r.get("proxy") or "Nasdaq-100")
    trip = r.get("trip_level")
    latest = r.get("latest_close")
    dist = r.get("trip_distance_pct")
    cut = _breaker_cut_text()
    level_txt = ""
    if trip and latest and dist is not None:
        fmt = (lambda v: f"{v:,.0f}") if max(trip, latest) >= 1000 else (lambda v: f"{v:,.2f}")
        level_txt = f" (about {fmt(trip)} on {proxy}, {dist:+.1f}% from the latest close of {fmt(latest)})"
    if state == "RISK_ON":
        what = (f"Nothing now. If the Nasdaq-100 closes {day_label(check, as_of)} below its 10-month average"
                f"{level_txt}, the {day_label(next_email, as_of)} email will tell you to {cut or 'act'}.")
        impact = "high" if dist is not None and dist > -3.0 else "medium"
    else:
        what = (f"The breaker is ON. If the Nasdaq-100 closes {day_label(check, as_of)} back above its 10-month "
                f"average{level_txt}, it switches off and the cash is reinvested by the normal buy rules.")
        impact = "high"
    why = "The only scheduled moment the plan can switch to crash-protection mode; intra-month swings don't count."
    if state == "RISK_ON" and dist is not None and dist > -3.0:
        why += (f" It is close: a fall of about {abs(dist):.1f}% by {day_label(check, as_of)} would trip it, and that "
                f"email would then contain large sell instructions.")
    return {
        "date": check.isoformat(), "kind": "plan", "title": "Tech circuit-breaker check (month-end close)",
        "impact": impact, "time_uk": "16:30", "what_to_do": what,
        "why": why,
        "action_needed": False, "source": "circuit_breaker",
    }


def buy_day_event(as_of: date, horizon_days: int) -> Optional[Dict[str, Any]]:
    try:
        from strategy_engine import first_london_business_day
    except Exception:  # pragma: no cover
        return None
    this_month = first_london_business_day(as_of.year, as_of.month)
    if this_month >= as_of:
        d = this_month
    else:
        nxt = as_of.replace(day=28) + timedelta(days=4)
        d = first_london_business_day(nxt.year, nxt.month)
    if d > as_of + timedelta(days=horizon_days):
        return None
    return {
        "date": d.isoformat(), "kind": "plan", "title": "Monthly buy day", "impact": "low", "time_uk": None,
        "what_to_do": ("Follow today's buy list, if any: cash above your reserve is invested today." if d == as_of
                       else "Follow that day's email: any cash above your reserve is invested then."),
        "why": "Buying once a month avoids reacting to daily noise; sells and risk checks still run every day.",
        "action_needed": False, "source": "buy_day",
    }


# --- Holdings' results dates ----------------------------------------------------------------------------------
def _earnings_when(research: Dict[str, Any], ticker: str) -> Tuple[Optional[str], str]:
    """(UK time 'HH:MM' or None, label) from research['earnings_ts_utc'] / ['earnings_is_estimate']."""
    ts = research.get("earnings_ts_utc")
    label = ""
    time_uk = None
    if ts and ZoneInfo is not None:
        try:
            dt_utc = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            uk = dt_utc.astimezone(ZoneInfo(LONDON))
            time_uk = uk.strftime("%H:%M")
            if exchange_hours(ticker)[0] == "US":
                ny = dt_utc.astimezone(ZoneInfo(NEW_YORK))
                mins = ny.hour * 60 + ny.minute
                label = ("after the US close" if mins >= 16 * 60
                         else "before the US open" if mins < 9 * 60 + 30 else "during US trading")
        except (ValueError, TypeError):
            time_uk = None
    if research.get("earnings_is_estimate"):
        label = (label + "; " if label else "") + "date is an estimate"
    return time_uk, label


def holdings_earnings_events(
    results_by_account: Dict[str, Dict[str, Any]],
    as_of: date,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    household_basis_gbp: float = 0.0,
) -> List[Dict[str, Any]]:
    """One item per held company reporting within the window, across accounts, with order-aware guidance."""
    end = as_of + timedelta(days=horizon_days)
    by_root: Dict[str, Dict[str, Any]] = {}
    order_roots: Dict[str, Dict[str, set]] = {}
    for acc_name, acc in (results_by_account or {}).items():
        if not isinstance(acc, dict):
            continue
        plan = acc.get("rebalancing_plan") if isinstance(acc.get("rebalancing_plan"), dict) else {}
        for side, key in (("sell", "sell_orders"), ("buy", "buy_orders")):
            for o in plan.get(key) or []:
                if isinstance(o, dict) and not o.get("is_pending"):
                    order_roots.setdefault(_root(o.get("ticker", "")), {"sell": set(), "buy": set()})[side].add(acc_name)
        for pos in acc.get("analyses") or []:
            if not isinstance(pos, dict) or pos.get("is_earmarked") or _num(pos.get("quantity"), 0.0) <= 0:
                continue
            research = pos.get("research") if isinstance(pos.get("research"), dict) else {}
            d = _parse_date(research.get("earnings_date_str"))
            if d is None and pos.get("days_to_earnings") is not None:
                try:
                    d = as_of + timedelta(days=int(pos["days_to_earnings"]))
                except (TypeError, ValueError):
                    d = None
            if d is None or d < as_of or d > end:
                continue
            ticker = str(pos.get("ticker", ""))
            root = _root(ticker)
            item = by_root.setdefault(root, {
                "date": d.isoformat(), "ticker": ticker, "root": root, "name": pos.get("name") or ticker,
                "symbol": pos.get("search_symbol") or root, "accounts": {}, "research": research,
            })
            item["accounts"][acc_name] = item["accounts"].get(acc_name, 0.0) + max(0.0, _num(pos.get("value_gbp"), 0.0))
    out: List[Dict[str, Any]] = []
    for root, item in by_root.items():
        d = _parse_date(item["date"])
        time_uk, when = _earnings_when(item["research"], item["ticker"])
        value = sum(item["accounts"].values())
        weight = value / household_basis_gbp if household_basis_gbp > 0 else 0.0
        orders = order_roots.get(root, {"sell": set(), "buy": set()})
        when_txt = " ".join(x for x in (day_label(d, as_of), time_uk and f"{time_uk} UK", when and f"({when})") if x)
        if orders["sell"]:
            what = f"Place today's sell before the results ({when_txt})."
            why = ("The trim is due anyway; doing it first removes the results-day gap risk at no extra cost.")
            action = True
        elif orders["buy"]:
            what = f"Results are due {when_txt}: place the buy after they are out."
            why = "Buying just before results adds gap risk for no expected reward."
            action = True
        else:
            what = "Hold. Don't sell because results are due, and don't add before them."
            why = ("On average shares earn slightly more around results, not less; the engine blocks new buys in "
                   "the 5 days before them.")
            action = False
        note = BELLWETHER_NOTES.get(root)
        if note:
            why = f"{why} {note}"
        holders = ", ".join(f"{a} £{v:,.0f}" for a, v in item["accounts"].items())
        out.append({
            "date": item["date"], "kind": "earnings",
            "title": f"{item['name']} ({item['symbol']}) results",
            "impact": "high" if (weight >= 0.03 or note) else "medium",
            "time_uk": time_uk, "when_label": when or None, "what_to_do": what, "why": why,
            "action_needed": action, "ticker": item["ticker"], "root": root, "holders": holders,
            "value_gbp": round(value, 2), "household_pct": round(weight * 100.0, 1),
            "source": "earnings",
        })
    return out


# --- Order timing guidance ------------------------------------------------------------------------------------
def fetch_next_earnings(ticker: str) -> Dict[str, Any]:
    """Next results date for a ticker not analysed by the engine (e.g. vested shares held outside Trading 212).

    Returns {'earnings_date_str': 'Feb 03, 2027', 'earnings_ts_utc': ISO or None, 'earnings_is_estimate': bool or None}
    or {} when Yahoo has nothing; never raises.
    """
    try:
        import yfinance as yf
        from research_agent import DeepResearchAgent
        tk = yf.Ticker(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        target: Optional[date] = None
        cal = getattr(tk, "calendar", None)
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            ed = cal["Earnings Date"]
            target = _parse_date(ed[0] if isinstance(ed, (list, tuple)) else ed)
        if target is None and info.get("earningsTimestamp"):
            target = datetime.fromtimestamp(float(info["earningsTimestamp"]), tz=timezone.utc).date()
        if target is None:
            return {}
        e_str = target.strftime("%b %d, %Y")
        ts, est = DeepResearchAgent._earnings_timing(info, e_str)
        return {"earnings_date_str": e_str, "earnings_ts_utc": ts, "earnings_is_estimate": est}
    except Exception as e:  # network / parsing: the radar simply omits the item
        logger.warning("Results date unavailable for %s: %s", ticker, e)
        return {}


def outside_earnings_events(
    outside_rows: Optional[List[Dict[str, Any]]],
    as_of: date,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    household_basis_gbp: float = 0.0,
    fetcher: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Results dates for vested holdings outside Trading 212 (they count in the household but can't be traded here)."""
    fetch = fetcher or fetch_next_earnings
    by_ticker: Dict[str, Dict[str, Any]] = {}
    for row in outside_rows or []:
        if not isinstance(row, dict) or _num(row.get("vested_gbp"), 0.0) <= 0:
            continue
        t = str(row.get("ticker") or "").upper().strip()
        if not t:
            continue
        agg = by_ticker.setdefault(t, {"value": 0.0, "accounts": set()})
        agg["value"] += _num(row.get("vested_gbp"), 0.0)
        if row.get("account"):
            agg["accounts"].add(str(row["account"]))
    out: List[Dict[str, Any]] = []
    for t, agg in by_ticker.items():
        research = fetch(t) or {}
        d = _parse_date(research.get("earnings_date_str"))
        if d is None or d < as_of or d > as_of + timedelta(days=horizon_days):
            continue
        time_uk, when = _earnings_when(research, t)
        weight = agg["value"] / household_basis_gbp if household_basis_gbp > 0 else 0.0
        owners = ", ".join(sorted(agg["accounts"])) or "household"
        out.append({
            "date": d.isoformat(), "kind": "earnings", "title": f"{t} results (vested shares held outside Trading 212)",
            "impact": "high" if weight >= 0.05 else "medium", "time_uk": time_uk, "when_label": when or None,
            "what_to_do": "Nothing in the ISAs. Don't trade on the result; keep to your plan for diversifying these "
                          "shares (e.g. sell on vest, Bed & ISA).",
            "why": "Single-company results can move a share 5-10% overnight. A planned schedule beats reacting to "
                   "one report.",
            "action_needed": False, "ticker": t, "root": _root(t), "holders": f"{owners} £{agg['value']:,.0f} (outside T212)",
            "value_gbp": round(agg["value"], 2), "household_pct": round(weight * 100.0, 1), "source": "earnings_outside",
        })
    return out


def order_timing_lines(results_by_account: Dict[str, Dict[str, Any]], as_of: date,
                       todays_events: Iterable[Dict[str, Any]] = ()) -> List[str]:
    """Plain-English 'when to place today's orders' lines, grouped by exchange."""
    groups: Dict[Tuple[str, str, str, str], List[str]] = {}
    for acc in (results_by_account or {}).values():
        plan = (acc or {}).get("rebalancing_plan") if isinstance(acc, dict) else None
        if not isinstance(plan, dict):
            continue
        for key in ("sell_orders", "buy_orders"):
            for o in plan.get(key) or []:
                if not isinstance(o, dict) or o.get("is_pending"):
                    continue
                hours = exchange_hours(o.get("ticker", ""))
                sym = str(o.get("search_symbol") or _root(o.get("ticker", "")))
                if sym not in groups.setdefault(hours, []):
                    groups[hours].append(sym)
    lines: List[str] = []
    for (label, open_l, close_l, tz), syms in groups.items():
        open_uk = _to_uk_hhmm(as_of, _shift_hhmm(open_l, 15), tz)
        close_uk = _to_uk_hhmm(as_of, _shift_hhmm(close_l, -15), tz)
        real_close = _to_uk_hhmm(as_of, close_l, tz)
        if not (open_uk and close_uk):
            continue
        tail = (f" If this email reaches you later, place them the next business day after {open_uk}: orders placed "
                f"outside market hours wait for the open, when prices can gap and spreads are widest." if label != "US" else
                " Orders placed outside US hours wait for the next open, which can gap.")
        lines.append(f"{label}-listed ({', '.join(syms)}): place between {open_uk} and {close_uk} UK "
                     f"(the market closes at {real_close}).{tail}")
    for ev in todays_events:
        if ev.get("impact") == "high" and ev.get("time_uk") and ev.get("kind") in ("data", "central_bank"):
            lines.append(f"{ev['title']} lands at {ev['time_uk']} UK today: avoid placing orders in the half-hour after it.")
    return lines


# --- Assembly -------------------------------------------------------------------------------------------------
_IMPACT_ORDER = {"high": 0, "medium": 1, "low": 2}


def build_event_radar(
    results_by_account: Dict[str, Dict[str, Any]],
    tech_regime: Optional[Dict[str, Any]] = None,
    as_of: Optional[date] = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    household_basis_gbp: float = 0.0,
    outside_rows: Optional[List[Dict[str, Any]]] = None,
    earnings_fetcher: Optional[Any] = None,
) -> Dict[str, Any]:
    """Everything scheduled in the next `horizon_days` that matters for the household, with what to do about it.

    Also stamps an 'event_note' on today's orders for companies reporting within two days, so the TL;DR says to
    place them before the results. Never raises on malformed engine data (the caller still wraps it). Results dates of
    vested shares held outside Trading 212 are looked up with `earnings_fetcher` (default: Yahoo Finance).
    """
    if as_of is None:
        try:
            import config
            as_of = config.london_date()
        except Exception:  # pragma: no cover
            as_of = date.today()
    horizon_days = max(1, int(horizon_days or DEFAULT_HORIZON_DAYS))
    items: List[Dict[str, Any]] = []
    items += upcoming_macro_events(as_of, horizon_days)
    items += _london_closures(as_of, horizon_days)
    cb = circuit_breaker_event(as_of, tech_regime)
    if cb and _parse_date(cb["date"]) <= as_of + timedelta(days=max(horizon_days, 31)):
        items.append(cb)
    bd = buy_day_event(as_of, horizon_days)
    if bd:
        items.append(bd)
    earnings = holdings_earnings_events(results_by_account, as_of, horizon_days, household_basis_gbp)
    items += earnings
    try:
        items += outside_earnings_events(outside_rows, as_of, horizon_days, household_basis_gbp, earnings_fetcher)
    except Exception as e:  # pragma: no cover - defensive: outside holdings never block the radar
        logger.warning("Outside-holding results dates skipped: %s", e)
    for it in items:
        d = _parse_date(it["date"])
        it["day_label"] = day_label(d, as_of) if d else it["date"]
        it["days_away"] = (d - as_of).days if d else None
    items.sort(key=lambda it: (it["date"], it.get("time_uk") or "99:99", _IMPACT_ORDER.get(it.get("impact"), 3)))

    # Stamp today's orders for companies reporting within two days
    soon = {e["root"]: e for e in earnings if e.get("days_away") is not None and e["days_away"] <= 2}
    for acc in (results_by_account or {}).values():
        plan = (acc or {}).get("rebalancing_plan") if isinstance(acc, dict) else None
        if not isinstance(plan, dict):
            continue
        for key, verb in (("sell_orders", "before"), ("buy_orders", "after")):
            for o in plan.get(key) or []:
                if not isinstance(o, dict) or o.get("is_pending"):
                    continue
                ev = soon.get(_root(o.get("ticker", "")))
                if ev:
                    when = " ".join(x for x in (ev["day_label"], ev.get("time_uk") and f"{ev['time_uk']} UK") if x)
                    o["event_note"] = f"results due {when}: place this {verb} them"

    todays = [it for it in items if it.get("days_away") == 0]
    actions = [it for it in items if it.get("action_needed")]
    n = len(items)
    if not n:
        headline = f"Nothing scheduled in the next {horizon_days} days needs your attention."
    elif actions:
        headline = (f"{n} scheduled event{'s' if n != 1 else ''} in the next {horizon_days} days; "
                    f"{len(actions)} need{'s' if len(actions) == 1 else ''} action (marked ⚠️). For the rest, the "
                    f"best move is to stick to the plan.")
    else:
        headline = (f"{n} scheduled event{'s' if n != 1 else ''} in the next {horizon_days} days; none needs action. "
                    f"The best move is to stick to the plan.")
    return {
        "as_of": as_of.isoformat(),
        "horizon_days": horizon_days,
        "items": items,
        "headline": headline,
        "order_timing": order_timing_lines(results_by_account, as_of, todays),
        "footnote": ("Scheduled events only. Surprises (e.g. geopolitics) are handled by the daily risk rules and the "
                     "monthly circuit-breaker. Dates: Fed, Bank of England, BLS and ONS calendars; company dates "
                     "from Yahoo Finance, which can change."),
    }


def radar_text_lines(radar: Optional[Dict[str, Any]], max_items: int = 14) -> List[str]:
    """Console lines for the radar (shared by main.py and tests)."""
    if not isinstance(radar, dict) or not radar.get("items") and not radar.get("headline"):
        return []
    lines = [f"📅 EVENT RADAR (next {radar.get('horizon_days', DEFAULT_HORIZON_DAYS)} days): {radar.get('headline', '')}"]
    for it in (radar.get("items") or [])[:max_items]:
        when = " ".join(x for x in (it.get("day_label"), it.get("time_uk") and f"{it['time_uk']} UK",
                                    it.get("when_label") and f"({it['when_label']})") if x)
        flag = "⚠️ " if it.get("action_needed") else ""
        held = f" · held: {it['holders']}" if it.get("holders") else ""
        lines.append(f"  • {flag}{when} — {it.get('title')}{held} ➔ {it.get('what_to_do')}")
    for line in radar.get("order_timing") or []:
        lines.append(f"  ⏱️ {line}")
    return lines
