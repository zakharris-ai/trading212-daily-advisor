"""Interactive Web Dashboard Generator for Trading 212 Institutional Engine.

Produces a standalone, zero-dependency, ultra-responsive HTML/JS dashboard
displaying real-time action cards, thematic MPT risk meters, position forensic
tables, and historical equity progression.
"""

from datetime import datetime
import html as _html
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional
import config as _config
from config import WATCHLIST, EXCLUDED_TICKERS, ACCOUNTS, REBALANCE_CONFIG

# Funds, gold and cash lines: single-company metrics (quality checklist, Altman Z) and the mechanical R:R do not apply
_FUND_TICKER_KEYS = (
    "VUAA", "VUSA", "VUAG", "CSPX", "VWRP", "VWRA", "SMGB", "CNDX", "CNX1",
    "DFND", "QQQ3", "SGLN", "IGLN", "GLD", "CSH2",
)
_GOLD_TICKER_KEYS = ("SGLN", "IGLN", "GLD")


def _num(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Finite float from `val` (a real 0 stays 0); `default` only when val is missing, NaN or non-numeric."""
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _first_num(*vals: Any) -> Optional[float]:
    """First value that is a finite number (explicit None checks, so 0 is kept)."""
    for v in vals:
        n = _num(v)
        if n is not None:
            return n
    return None


def _gbp(val: Optional[float], decimals: int = 0) -> str:
    """'£1,234' / '-£1,234', or 'N/A' when missing."""
    if val is None:
        return "N/A"
    sign = "-" if val < 0 else ""
    return f"{sign}£{abs(val):,.{decimals}f}"


def _pct(val: Optional[float], decimals: int = 0, signed: bool = True) -> str:
    if val is None:
        return "N/A"
    return f"{val:+.{decimals}f}%" if signed else f"{val:.{decimals}f}%"


def _regime_badge_class(text: Any, score: Optional[float] = None) -> str:
    """Badge colour that follows the actual regime text (emoji/keywords), falling back to the macro score."""
    t = str(text or "")
    up = t.upper()
    if "⚪" in t or "UNAVAILABLE" in up:
        return "badge-slate"
    if "🔴" in t or "RISK-OFF" in up or "DEFCON 1" in up:
        return "badge-rose"
    if "🟠" in t or "DEFCON 2" in up:
        return "badge-orange"
    if "🟡" in t or "DEFCON 3" in up:
        return "badge-amber"
    if "🟢" in t:
        return "badge-emerald"
    if score is not None:
        if score >= 58:
            return "badge-emerald"
        if score >= 42:
            return "badge-amber"
        if score >= 28:
            return "badge-orange"
        return "badge-rose"
    return "badge-slate"


def _is_isa_wrapper(val: Any) -> bool:
    if isinstance(val, dict):
        val = val.get("wrapper") or val.get("type") or val.get("account_type")
    return "ISA" in str(val or "").upper()


def _wrapper_footer_line(account_names: Iterable[str]) -> str:
    """'Accounts: Stocks & Shares ISA (no CGT on trades)' when config.ACCOUNT_WRAPPERS says ISA for every account."""
    wrappers = getattr(_config, "ACCOUNT_WRAPPERS", {}) or {}
    if not isinstance(wrappers, dict):
        return ""
    names = [n for n in account_names if n and n != "ALL"]
    if names and all(_is_isa_wrapper(wrappers.get(n)) for n in names):
        return "Accounts: Stocks & Shares ISA (no CGT on trades)"
    return ""


def _reserve_status(plan: Optional[Dict[str, Any]], cash_info: Optional[Dict[str, Any]], acc_name: str) -> Dict[str, float]:
    """Reserve floor vs what free cash actually funds (C5 keys first, legacy keys as fallback)."""
    plan = plan or {}
    cash_info = cash_info or {}
    floor = _num(plan.get("reserve_floor_gbp"))
    if floor is None:
        floor = _num(plan.get("buffer_reserve_gbp"))
        if floor is None or floor <= 0:
            floor = _num(
                (ACCOUNTS.get(acc_name) or {}).get("min_cash_reserve_gbp"),
                1500.0,
            )
    free = _num(plan.get("available_cash_gbp"))
    if free is None:
        free = _first_num(cash_info.get("free_gbp"), cash_info.get("free"))
    if free is None:
        free = 0.0
    funded = _num(plan.get("reserve_funded_gbp"))
    if funded is None:
        funded = max(0.0, min(floor, free))
    shortfall = _num(plan.get("reserve_shortfall_gbp"))
    if shortfall is None:
        shortfall = max(0.0, floor - funded)
    return {"floor": floor, "funded": funded, "shortfall": max(0.0, shortfall), "free": free}


def _reserve_text(rs: Dict[str, float]) -> str:
    if rs["shortfall"] > 0.005:
        return (
            f"⚠️ UNFUNDED — £{rs['shortfall']:,.0f} short "
            f"(free cash covers £{rs['funded']:,.0f} of the £{rs['floor']:,.0f} floor)"
        )
    return f"✅ Funded from free cash (£{rs['funded']:,.0f} of £{rs['floor']:,.0f})"


def _earmark_line(plan: Optional[Dict[str, Any]], analyses: Optional[List[Dict[str, Any]]] = None) -> str:
    """'🔒 Ring-fenced (excluded from portfolio maths): CSH2.L £X' from plan earmarked_gbp / earmarked_tickers.

    Falls back to analyses flagged `is_earmarked` when the plan does not carry the earmark keys.
    """
    plan = plan or {}
    amount = _num(plan.get("earmarked_gbp"))
    tickers = plan.get("earmarked_tickers") or []
    if amount is None and not tickers and analyses:
        marked = [a for a in analyses if isinstance(a, dict) and a.get("is_earmarked")]
        if marked:
            tickers = [str(a.get("ticker", "")) for a in marked]
            amount = sum(_num(a.get("value_gbp"), 0.0) for a in marked)
    if isinstance(tickers, dict):
        parts = []
        for t, v in tickers.items():
            v_num = _num(v)
            parts.append(f"{t} £{v_num:,.0f}" if v_num is not None else str(t))
        body = ", ".join(parts)
    else:
        if isinstance(tickers, str):
            tickers = [tickers]
        names = []
        for t in tickers:
            if isinstance(t, dict):
                t = t.get("ticker") or t.get("symbol") or ""
            if t:
                names.append(str(t))
        if isinstance(tickers, (set, frozenset)):
            names.sort()
        body = ", ".join(names)
        if amount is not None and (amount > 0 or names):
            body = f"{body} £{amount:,.0f}".strip()
    if not body or (not tickers and not (amount is not None and amount > 0)):
        return ""
    return f"🔒 Ring-fenced (excluded from portfolio maths): {body}"


def _outside_holdings_line(consolidated_view: Optional[Dict[str, Any]]) -> str:
    cv = consolidated_view or {}
    vested = _num(cv.get("outside_vested_gbp_total"))
    unvested = _num(cv.get("outside_unvested_gbp_total"))
    has_amounts = (vested is not None and vested > 0) or (unvested is not None and unvested > 0)
    if not cv.get("outside_holdings") and not has_amounts:
        return ""
    vested_v = vested if vested is not None else 0.0
    unvested_v = unvested if unvested is not None else 0.0
    labels = sorted({
        str(row.get("ticker") or row.get("name") or "").strip()
        for row in (cv.get("outside_holdings") or [])
        if isinstance(row, dict)
    } - {""})
    label_txt = f" ({_html.escape(', '.join(labels))})" if labels else ""
    return (
        f"Holdings outside Trading 212{label_txt}: vested £{vested_v:,.0f} (counted in concentration caps)"
        f" + unvested £{unvested_v:,.0f} (not counted)"
    )


def _crash_delta_text(delta: Optional[float]) -> str:
    """Signed C6 crash_delta_gbp: negative means the plan adds 2022-style stress."""
    if delta is None:
        return ""
    if delta < -0.5:
        return f"plan adds £{abs(delta):,.0f} of 2022-style stress"
    if delta > 0.5:
        return f"plan removes £{delta:,.0f} of 2022-style stress"
    return "plan leaves 2022-style stress unchanged"


def _risk_kpi_texts(rp: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Pre-rendered risk KPI strings (one source of truth for the server render and the JS account switch)."""
    rp = rp or {}

    def arrow(cur: Optional[float], post: Optional[float], fmt) -> str:
        if cur is None and post is None:
            return "N/A"
        if cur is None:
            return f"after plan {fmt(post)}"
        if post is None or abs(post - cur) < 1e-9:
            return fmt(cur)
        return f"{fmt(cur)} ➔ {fmt(post)}"

    cur_sort, post_sort = _num(rp.get("current_sortino_5y")), _num(rp.get("post_sortino_5y"))
    cur_cagr, post_cagr = _num(rp.get("current_cagr_5y_pct")), _num(rp.get("post_cagr_5y_pct"))
    cur_beta, post_beta = _num(rp.get("current_beta_5y")), _num(rp.get("post_beta_5y"))
    cur_st, post_st = _num(rp.get("current_stress_2022_pct")), _num(rp.get("post_stress_2022_pct"))
    cur_st_gbp, post_st_gbp = _num(rp.get("current_stress_2022_gbp")), _num(rp.get("post_stress_2022_gbp"))
    cur_sh, post_sh = _num(rp.get("current_shield_pct")), _num(rp.get("post_shield_pct"))
    cur_def, post_def = _num(rp.get("current_defensive_pct")), _num(rp.get("post_defensive_pct"))
    cur_def_gbp, post_def_gbp = _num(rp.get("current_defensive_gbp")), _num(rp.get("post_defensive_gbp"))

    sortino = arrow(cur_sort, post_sort, lambda v: f"{v:.2f}")
    if sortino != "N/A":
        sortino += " Sortino"

    cagr_parts = []
    if cur_cagr is not None:
        cagr_parts.append(f"now {cur_cagr:+.0f}%/yr")
    if post_cagr is not None:
        cagr_parts.append(f"after plan {post_cagr:+.0f}%/yr")
    cagr = "Est. 5Y CAGR (blended hist./model estimate, info only): " + (" ➔ ".join(cagr_parts) if cagr_parts else "N/A")
    beta_txt = arrow(cur_beta, post_beta, lambda v: f"{v:.2f}")
    if beta_txt != "N/A":
        cagr += f" · Beta {beta_txt}"

    crash = arrow(cur_st, post_st, lambda v: f"{v:+.1f}%")
    crash_sub = "2022-style sim: " + arrow(cur_st_gbp, post_st_gbp, lambda v: _gbp(v))
    delta_txt = _crash_delta_text(_num(rp.get("crash_delta_gbp")))
    if delta_txt:
        crash_sub += f" · {delta_txt}"

    shield = arrow(cur_sh, post_sh, lambda v: f"{v:.1f}%")
    defensive = arrow(cur_def, post_def, lambda v: f"{v:.0f}%")
    def_gbp = arrow(cur_def_gbp, post_def_gbp, lambda v: _gbp(v))
    defensive_txt = f"True defensive (cash, money-market, gold): {defensive}"
    if def_gbp != "N/A":
        defensive_txt += f" ({def_gbp})"
    shield_sub = str(rp.get("crash_readiness_grade") or "N/A")

    bear, base, bull = (_num(rp.get(f"projected_5y_{k}_gbp")) for k in ("bear", "base", "bull"))
    if bear is not None or base is not None or bull is not None:
        proj = f"Bear {_gbp(bear)} · Base {_gbp(base)} · Bull {_gbp(bull)}"
        rates = [_num(rp.get(f"projected_5y_{k}_cagr_pct")) for k in ("bear", "base", "bull")]
        proj_sub = "5-yr scenarios: " + " · ".join(
            f"{name} {_pct(r)}/yr" for name, r in zip(("bear", "base", "bull"), rates)
        )
    else:
        cons, target = _num(rp.get("projected_5y_conservative_gbp")), _num(rp.get("projected_5y_target_gbp"))
        if cons is None and target is None:
            proj, proj_sub = "N/A", "5-year projection unavailable"
        else:
            proj = f"{_gbp(cons)} – {_gbp(target)}"
            proj_sub = "Legacy conservative–target range (no bear case)"

    return {
        "sortino": sortino,
        "cagr": cagr,
        "crash": crash,
        "crash_sub": crash_sub,
        "shield": shield,
        "shield_sub": shield_sub,
        "defensive": defensive_txt,
        "proj": proj,
        "proj_sub": proj_sub,
    }


def _all_stage2_verified(analyses: List[Dict[str, Any]]) -> bool:
    """True only when every position is confirmed weekly Stage 2 on non-degraded data."""
    if not analyses:
        return False
    return all(
        a.get("is_weekly_stage_2") is True and a.get("data_ok") is not False
        for a in analyses
        if isinstance(a, dict)
    )


def _signal_prefix(action_text: str, is_trim: bool) -> str:
    """Avoid double emoji when the action text already carries one (e.g. '🔴 SELL (LEVERAGE POLICY — 0% CAP)')."""
    if action_text.startswith(("🔴", "🟠", "🟢", "🟡", "🔒", "⚪", "🏦", "⏳")):
        return ""
    return "🟠 " if is_trim else "🔴 "


# --- Household strategic allocation (Phase 1) display helpers --------------------------------------------
# Shared by the dashboard, the email digest (mailer.py) and the console / plain-text summary (main.py).
# Every helper is None-safe: missing or malformed engine keys produce empty values, never an exception.
_BUCKET_ORDER = ("core", "satellite", "defensive")
_BUCKET_LABELS = {
    "core": "Core global index",
    "satellite": "Satellites (single stocks & thematic funds)",
    "defensive": "Defensive (cash, money-market, gold, short gilts)",
}
_BUCKET_SHORT_LABELS = {"core": "Core", "satellite": "Satellites", "defensive": "Defensive"}
_BUCKET_ALIASES = {"satellites": "satellite", "defence": "defensive", "defense": "defensive"}
_MIX_STATUS_TEXT = {"under": "Under target", "over": "Over target", "in_band": "In band"}
_MIX_EXPLAINER = (
    "New money goes to whichever part of the mix is furthest below target. Nothing profitable is sold to rebalance."
)
_BIG_TECH_SOURCE_LABELS = {
    "direct": "Direct",
    "fund": "Fund look-through",
    "outside": "Vested, outside T212",
}


def _bucket_key(val: Any) -> str:
    """'core' / 'satellite' / 'defensive' for an engine bucket value; '' when missing or unknown."""
    key = str(val or "").strip().lower()
    key = _BUCKET_ALIASES.get(key, key)
    return key if key in _BUCKET_ORDER else ""


def _bucket_label(val: Any) -> str:
    """Badge text for a buy order's bucket: 'CORE' / 'SATELLITE' / 'DEFENSIVE'; '' when unknown."""
    return _bucket_key(val).upper()


def _share_text(val: Optional[float], decimals: int = 1) -> str:
    """Share of household wealth: '21.9%', whole numbers drop the decimals ('65%'); 'N/A' when missing."""
    if val is None:
        return "N/A"
    if abs(val - round(val)) < 1e-9:
        return f"{val:.0f}%"
    return f"{val:.{decimals}f}%"


def _household_mix_view(consolidated_view: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Display-ready consolidated_view["household_mix"]; {} when absent or unusable.

    rows: [{key, label, now_pct, target_pct, post_pct, status ('under' / 'over' / 'in_band' / ''), out_of_band}].
    The engine's status is trusted; without it the status is derived from out_of_band or the +/- band.
    post_pct is None when the engine applied no plans (post_trade_available False).
    """
    hm = consolidated_view.get("household_mix") if isinstance(consolidated_view, dict) else None
    if not isinstance(hm, dict):
        return {}
    band = _num(hm.get("band_pp"))
    post_ok = hm.get("post_trade_available") is not False
    buckets = hm.get("buckets")
    rows: List[Dict[str, Any]] = []
    for b in buckets if isinstance(buckets, list) else []:
        if not isinstance(b, dict):
            continue
        key = _bucket_key(b.get("bucket"))
        label = str(b.get("label") or _BUCKET_LABELS.get(key) or b.get("bucket") or "").strip()
        if not label:
            continue
        now = _num(b.get("current_pct"))
        target = _num(b.get("target_pct"))
        status = str(b.get("status") or "").strip().lower()
        if status not in _MIX_STATUS_TEXT:
            status = ""
            if now is not None and target is not None:
                out_of_band = b.get("out_of_band")
                if not isinstance(out_of_band, bool):
                    out_of_band = band is not None and abs(now - target) > band
                status = ("under" if now < target else "over") if out_of_band else "in_band"
        rows.append({
            "key": key,
            "label": label,
            "now_pct": now,
            "target_pct": target,
            "post_pct": _num(b.get("post_trade_pct")) if post_ok else None,
            "status": status,
            "out_of_band": status in ("under", "over"),
        })
    summary = str(hm.get("summary") or "").strip()
    if not rows and not summary:
        return {}
    return {
        "rows": rows,
        "summary": summary,
        "band_pp": band,
        "basis_gbp": _num(hm.get("basis_gbp")),
        "outside_included_gbp": _num(hm.get("outside_included_gbp")),
        "excluded_cash_gbp": _num(hm.get("excluded_cash_gbp")),
        "post_trade_available": post_ok,
    }


def _mix_basis_text(mix: Optional[Dict[str, Any]]) -> str:
    """'Basis £200,000: Trading 212 investable money (earmarked funds excluded) + £20,000 vested ... · band ±5pp'."""
    mix = mix or {}
    parts: List[str] = []
    basis = mix.get("basis_gbp")
    if basis is not None and basis > 0:
        txt = f"Basis {_gbp(basis)}: Trading 212 investable money (earmarked funds excluded)"
        outside = mix.get("outside_included_gbp")
        if outside is not None and outside >= 0.5:
            txt += f" + {_gbp(outside)} vested shares held outside Trading 212"
        parts.append(txt)
    excluded = mix.get("excluded_cash_gbp")
    if excluded is not None and excluded >= 0.5:
        parts.append(f"{_gbp(excluded)} temporary reserve top-up cash not counted")
    band = mix.get("band_pp")
    if band is not None:
        parts.append(f"band ±{band:g}pp")
    return " · ".join(parts)


def _big_tech_view(consolidated_view: Optional[Dict[str, Any]], top_n: int = 8) -> Dict[str, Any]:
    """Display-ready consolidated_view["big_tech"]; {} when absent or without an exposure figure.

    line: '47% of household wealth (cap 30%)', plus ' — no new big-tech buys' when over the cap.
    rows: the top_n breakdown rows by exposure (source labelled direct / fund look-through / outside).
    """
    bt = consolidated_view.get("big_tech") if isinstance(consolidated_view, dict) else None
    if not isinstance(bt, dict):
        return {}
    basis = _num(bt.get("basis_gbp"))
    exposure_gbp = _num(bt.get("exposure_gbp"))
    pct = _num(bt.get("exposure_pct"))
    if pct is None and exposure_gbp is not None and basis is not None and basis > 0:
        pct = exposure_gbp / basis * 100.0
    if pct is None:
        return {}
    cap = _num(bt.get("cap_pct"))
    over_cap = bt.get("over_cap")
    if not isinstance(over_cap, bool):
        over_cap = cap is not None and pct > cap
    line = f"{pct:.0f}% of household wealth"
    if cap is not None:
        line += f" (cap {_share_text(cap)})"
    if over_cap:
        line += " — no new big-tech buys"

    breakdown = bt.get("breakdown")
    rows: List[Dict[str, Any]] = []
    for r in breakdown if isinstance(breakdown, list) else []:
        if not isinstance(r, dict):
            continue
        ticker = str(r.get("ticker") or "").strip()
        exposure = _num(r.get("exposure_gbp"))
        if not ticker or exposure is None:
            continue
        source = str(r.get("source") or "").strip().lower()
        rows.append({
            "account": str(r.get("account") or "").strip(),
            "ticker": ticker,
            "source": source,
            "source_label": _BIG_TECH_SOURCE_LABELS.get(source, source.capitalize() or "N/A"),
            "value_gbp": _num(r.get("value_gbp")),
            "weight": _num(r.get("lookthrough_weight")),
            "exposure_gbp": exposure,
        })
    rows.sort(key=lambda r: r["exposure_gbp"], reverse=True)
    return {
        "pct": pct,
        "cap_pct": cap,
        "over_cap": over_cap,
        "post_pct": _num(bt.get("post_trade_exposure_pct")),
        "exposure_gbp": exposure_gbp,
        "basis_gbp": basis,
        "line": line,
        "rows": rows[:max(0, int(top_n))],
        "total_rows": len(rows),
        "note": str(bt.get("note") or "").strip(),
    }


def _tracking_text(consolidated_view: Optional[Dict[str, Any]]) -> str:
    """Plain-English since-tracking-began line from consolidated_view["tracking_performance"]; '' when absent."""
    tp = consolidated_view.get("tracking_performance") if isinstance(consolidated_view, dict) else None
    if not isinstance(tp, dict):
        return ""
    return str(tp.get("text") or "").strip()


def _allocation_view(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Display-ready rebalancing_plan["household_allocation"] for one account; {} when absent or empty.

    cadence: the engine's cadence_note (falls back to the next buy day when buys wait for it), notes: list of
    plain-English strings, split_text: 'Core £3,900 · Defensive £1,120' from by_bucket_gbp (non-zero buckets,
    only on days buys are allowed).
    """
    ha = plan.get("household_allocation") if isinstance(plan, dict) else None
    if not isinstance(ha, dict):
        return {}
    cadence = str(ha.get("cadence_note") or "").strip()
    next_day = str(ha.get("next_buy_day") or "").strip()
    if not cadence and ha.get("buys_allowed") is False and next_day:
        cadence = f"New buys wait for the next monthly buy day ({next_day})."
    raw_notes = ha.get("notes")
    if isinstance(raw_notes, str):
        raw_notes = [raw_notes]
    notes = [
        str(n).strip()
        for n in (raw_notes if isinstance(raw_notes, (list, tuple)) else [])
        if n is not None and not isinstance(n, (dict, list)) and str(n).strip()
    ]
    by_bucket = ha.get("by_bucket_gbp")
    split_parts: List[str] = []
    if isinstance(by_bucket, dict) and ha.get("buys_allowed") is not False:
        for key in _BUCKET_ORDER:
            amount = _num(by_bucket.get(key))
            if amount is not None and amount >= 0.5:
                split_parts.append(f"{_BUCKET_SHORT_LABELS[key]} {_gbp(amount)}")
    split_text = " · ".join(split_parts)
    if not (cadence or notes or split_text):
        return {}
    return {"cadence": cadence, "notes": notes, "split_text": split_text, "next_buy_day": next_day}


def _lookthrough_text(weight: Optional[float]) -> str:
    """Fund look-through weight as a share: 0.35 -> '35%', 1.0 -> '100%'; 'N/A' when missing."""
    return f"{weight * 100.0:.0f}%" if weight is not None else "N/A"


def _household_mix_panel_html(
    consolidated_view: Optional[Dict[str, Any]],
    results_by_account: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Server-rendered dashboard 'Household mix' panel; '' when the engine supplied none of the Phase 1 data.

    Bucket table (now / target / after today's plan) with status chips, the since-tracking-began line and the
    rebalancing explainer; US big-tech exposure vs the cap with the top 8 breakdown rows; per-account buy-cadence
    notes. Household-wide, so it is static HTML (no JSON payload). Every engine-supplied string is HTML-escaped.
    """
    esc = _html.escape
    mix = _household_mix_view(consolidated_view)
    big_tech = _big_tech_view(consolidated_view, top_n=8)
    tracking = _tracking_text(consolidated_view)

    alloc_blocks: List[str] = []
    for acc_name, acc_data in (results_by_account or {}).items():
        alloc = _allocation_view((acc_data or {}).get("rebalancing_plan") if isinstance(acc_data, dict) else None)
        if not alloc:
            continue
        parts = [f"<b>👤 {esc(str(acc_name))}:</b> {esc(alloc['cadence'] or 'No cadence note today.')}"]
        if alloc["split_text"]:
            parts.append(f'<div class="mix-sub">New money split: {esc(alloc["split_text"])}</div>')
        if alloc["notes"]:
            parts.append("<ul>" + "".join(f"<li>{esc(n)}</li>" for n in alloc["notes"]) + "</ul>")
        alloc_blocks.append(f'<div class="mix-acc">{"".join(parts)}</div>')

    if not (mix or big_tech or tracking or alloc_blocks):
        return ""

    cards: List[str] = []
    mix_rows = mix.get("rows") or []
    if mix or tracking:
        body: List[str] = []
        basis_txt = _mix_basis_text(mix)
        if basis_txt:
            body.append(f'<div class="mix-sub">{esc(basis_txt)}</div>')
        if mix_rows:
            tr_html = ""
            for r in mix_rows:
                status_txt = _MIX_STATUS_TEXT.get(r["status"], "")
                chip = f'<span class="chip chip-{r["status"]}">{status_txt}</span>' if status_txt else ""
                post_txt = _share_text(r["post_pct"]) if r["post_pct"] is not None else "—"
                tr_html += (
                    f'<tr><td><div class="mix-label">{esc(r["label"])}</div>{chip}</td>'
                    f'<td class="mix-num"><b>{_share_text(r["now_pct"])}</b></td>'
                    f'<td class="mix-num">{_share_text(r["target_pct"])}</td>'
                    f'<td class="mix-num">{post_txt}</td></tr>'
                )
            body.append(
                '<table class="mix-table"><thead><tr><th>Bucket</th><th class="mix-num">Now</th>'
                '<th class="mix-num">Target</th><th class="mix-num">After today\'s plan</th></tr></thead>'
                f"<tbody>{tr_html}</tbody></table>"
            )
            if not mix.get("post_trade_available", True):
                body.append('<div class="mix-sub">After-plan figures are unavailable today (no plan applied).</div>')
        elif mix.get("summary"):
            body.append(f'<div class="mix-line">{esc(mix["summary"])}</div>')
        if tracking:
            body.append(f'<div class="mix-line">📈 {esc(tracking)}</div>')
        if mix_rows:
            body.append(f'<div class="mix-explainer">💡 {esc(_MIX_EXPLAINER)}</div>')
        cards.append(
            '<div class="mix-card"><div class="mix-head">⚖️ Core / Satellites / Defensive</div>'
            + "".join(body) + "</div>"
        )

    if big_tech:
        line_color = "var(--rose)" if big_tech["over_cap"] else "var(--emerald)"
        sub_parts = []
        if big_tech["post_pct"] is not None:
            sub_parts.append(f"After today's plan: {_share_text(big_tech['post_pct'])}")
        sub_parts.append("Counts fund look-through and vested shares held outside Trading 212")
        body = [
            f'<div class="mix-head">🏢 US big tech: <span style="color: {line_color};">{esc(big_tech["line"])}</span></div>',
            f'<div class="mix-sub">{esc(" · ".join(sub_parts))}</div>',
        ]
        if big_tech["rows"]:
            tr_html = "".join(
                f'<tr><td><b>{esc(r["ticker"])}</b><div class="mix-sub-cell">{esc(r["account"] or "—")}</div></td>'
                f'<td>{esc(r["source_label"])}</td><td class="mix-num">{_gbp(r["value_gbp"])}</td>'
                f'<td class="mix-num">{_lookthrough_text(r["weight"])}</td>'
                f'<td class="mix-num"><b>{_gbp(r["exposure_gbp"])}</b></td></tr>'
                for r in big_tech["rows"]
            )
            body.append(
                '<table class="mix-table"><thead><tr><th>Holding</th><th>Source</th>'
                '<th class="mix-num">Value</th><th class="mix-num">Look-through</th>'
                '<th class="mix-num">Big tech</th></tr></thead>'
                f"<tbody>{tr_html}</tbody></table>"
            )
        foot = []
        if big_tech["total_rows"] > len(big_tech["rows"]):
            foot.append(f"Top {len(big_tech['rows'])} of {big_tech['total_rows']} holdings shown")
        if big_tech["note"]:
            foot.append(big_tech["note"])
        if foot:
            body.append(f'<div class="mix-sub">{esc(" · ".join(foot))}</div>')
        cards.append('<div class="mix-card">' + "".join(body) + "</div>")

    grid_html = f'\n    <div class="mix-grid">\n      {"".join(cards)}\n    </div>' if cards else ""
    cadence_html = (
        '\n    <div class="mix-card mix-wide"><div class="mix-head">🗓️ New money today, by account</div>'
        + "".join(alloc_blocks) + "</div>"
        if alloc_blocks else ""
    )
    return f"""
    <!-- Household Mix (strategic allocation: core / satellites / defensive) -->
    <div class="section-title">
      <span>🧭 Household Mix — all accounts + vested outside shares, as one household</span>
    </div>{grid_html}{cadence_html}"""


def generate_dashboard_html(
    results_by_account: Dict[str, Dict[str, Any]],
    consolidated_view: Dict[str, Any],
    macro_regime: Dict[str, Any],
    gbp_rate: float,
    history: Optional[List[Dict[str, Any]]] = None,
    watchlist_scan: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Renders the comprehensive interactive HTML dashboard.

    `watchlist_scan` (optional, C10): per-ticker analysis dicts for the WATCHLIST (e.g. analyze_position()
    results). Cards show their real values; without it every card reads N/A / NOT SCANNED.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_date = datetime.now().strftime("%Y-%m-%d")
    consolidated_view = consolidated_view or {}
    macro_regime = macro_regime or {}
    results_by_account = results_by_account or {}
    model_version = getattr(_config, "MODEL_VERSION", "dev")
    accounts_title_label = " & ".join(_html.escape(str(k)) for k in results_by_account.keys() if k != "ALL") or "Portfolio"
    excluded_badge_html = (
        f'<span class="badge badge-indigo">Excluded: {_html.escape(", ".join(sorted(EXCLUDED_TICKERS)))}</span>'
        if EXCLUDED_TICKERS
        else '<span class="badge badge-indigo">5-Year High-Sortino Universe</span>'
    )
    gbp_rate_val = _num(gbp_rate)
    gbp_rate_txt = f"1 GBP = ${gbp_rate_val:.4f} USD" if gbp_rate_val is not None else "GBP/USD N/A"

    tot_net_worth = _num(consolidated_view.get("total_net_worth_gbp"), 0.0)
    tot_invested = _num(consolidated_view.get("total_invested_gbp"), 0.0)
    tot_cash = _num(consolidated_view.get("total_liquid_cash_gbp"), 0.0)
    tot_blocked = _num(consolidated_view.get("total_blocked_cash_gbp"), 0.0)
    tot_ppl = _num(consolidated_view.get("total_ppl_gbp"), 0.0)
    dod = consolidated_view.get("day_over_day") or {}

    dod_gbp = _num(dod.get("combined_delta_gbp"), 0.0)
    dod_pct = _num(dod.get("combined_delta_pct"), 0.0)
    dod_sign = "+" if dod_gbp >= 0 else ""
    dod_color = "#10b981" if dod_gbp >= 0 else "#ef4444"

    # Macro parameters (null-safe, explicit None checks; inputs listed in macro_missing_keys display as N/A)
    macro_missing_keys = set(macro_regime.get("macro_missing_keys") or [])

    def _mval(key: str) -> Optional[float]:
        if key in macro_missing_keys:
            return None
        return _num(macro_regime.get(key))

    vix = _mval("vix")
    macro_score = _num(macro_regime.get("macro_score"))
    climate = macro_regime.get("risk_climate") or "⚪ MACRO DATA UNAVAILABLE (NEUTRAL)"
    crash_defcon = macro_regime.get("crash_defcon") or "⚪ DEFCON N/A"
    us10y = _mval("us10y_yield")
    curve_spread = _mval("yield_curve_spread")
    term_struct = _mval("vix_term_structure")
    oil = _mval("crude_oil_price")
    spy_dist = _mval("spy_dist_days")
    qqq_dist = _mval("qqq_dist_days")
    hb_ratio = _mval("high_beta_ratio")
    semi_ratio = _mval("semi_leadership_ratio")

    climate_badge_cls = _regime_badge_class(climate, macro_score)
    defcon_badge_cls = _regime_badge_class(crash_defcon, macro_score)
    score_txt = f" ({macro_score:.0f}/100)" if macro_score is not None else ""
    vix_txt = f"{vix:.1f}" if vix is not None else "N/A"
    term_txt = f"{term_struct:.2f}x term" if term_struct is not None else "term N/A"
    us10y_txt = f"{us10y:.2f}%" if us10y is not None else "N/A"
    spread_txt = f"{curve_spread:+.2f}%" if curve_spread is not None else "N/A"
    spy_dist_txt = f"{spy_dist:.0f}d" if spy_dist is not None else "N/A"
    qqq_dist_txt = f"{qqq_dist:.0f}d" if qqq_dist is not None else "N/A"
    hb_txt = f"{hb_ratio:.2f}x" if hb_ratio is not None else "N/A"
    semi_txt = f"{semi_ratio:.3f}x" if semi_ratio is not None else "N/A"
    oil_txt = f"${oil:.1f}" if oil is not None else "N/A"
    macro_missing = [str(m) for m in (macro_regime.get("macro_missing") or [])]
    partial_macro_html = ""
    if macro_regime.get("macro_data_ok") is not False and macro_missing:
        partial_macro_html = f"""
      <div class="macro-item">
        <span class="macro-label">Partial data:</span>
        <span class="macro-val" style="color: var(--amber);">{_html.escape(", ".join(macro_missing))} unavailable (scored neutral)</span>
      </div>"""

    # Historical Performance Data for SVG Chart
    history_points = []
    if history:
        for entry in history[-14:]:  # last 14 sessions
            accs = entry.get("accounts") or {}
            point = {
                "date": str(entry.get("date", "")),
                "total": _num(entry.get("combined_total"), 0.0),
                "ppl": _num(entry.get("total_ppl"), 0.0),
            }
            for acc_key, acc_val in (accs.items() if isinstance(accs, dict) else []):
                point.setdefault(str(acc_key).lower(), _num(acc_val, 0.0))
            for acc_key in results_by_account.keys():
                point.setdefault(str(acc_key).lower(), 0.0)
            history_points.append(point)
    n_sessions = len(history_points)
    sessions_label = f"(Trailing {n_sessions} Session{'s' if n_sessions != 1 else ''})"

    # Prepare positions data across accounts
    all_positions = []
    for acc_name, acc_data in results_by_account.items():
        plan = acc_data.get("rebalancing_plan") or {}
        analyses = acc_data.get("analyses") or []
        for pos in analyses:
            research = pos.get("research") or {}
            p_dict = {
                "account": acc_name,
                "ticker": pos.get("ticker", ""),
                "symbol": pos.get("search_symbol", pos.get("ticker", "")),
                "name": pos.get("name", pos.get("ticker", "")),
                "price": _num(pos.get("price"), 0.0),
                "currency": pos.get("currency", "USD"),
                "avg_price": _num(pos.get("avg_price"), 0.0),
                "quantity": _num(pos.get("quantity"), 0.0),
                "value_gbp": _num(pos.get("value_gbp"), 0.0),
                "unrealized_pnl": _num(pos.get("unrealized_pnl"), 0.0),
                "weight_pct": _num(pos.get("portfolio_weight_pct"), 0.0),
                "signal": pos.get("signal") or "",
                "action_short": pos.get("action_short", "HOLD"),
                "plan_status": pos.get("plan_status") or "HOLD",
                "badge_color": pos.get("plan_badge_color", "#6366f1"),
                "conviction_score": int(_num(pos.get("conviction_score"), 50.0)),
                "piotroski": _first_num(pos.get("piotroski_score"), research.get("piotroski_score")),
                "altman_z": _first_num(pos.get("altman_z"), research.get("altman_z")),
                "roic": _first_num(pos.get("roic"), research.get("roic")),
                "roic_label": pos.get("roic_label") or research.get("roic_label", "N/A"),
                "insider_label": pos.get("insider_label") or research.get("insider_label", "Neutral"),
                "options_pcr_vol": _first_num(pos.get("options_pcr_vol"), research.get("options_pcr_vol")),
                "options_pcr_label": pos.get("options_pcr_label") or research.get("options_pcr_label", "N/A"),
                "inst_flow_label": pos.get("institutional_flow_label") or research.get("institutional_flow_label", "Stable"),
                "stage": pos.get("minervini_stage") or "N/A",
                "trailing_stop": _num(pos.get("trailing_stop"), 0.0),
                "target_price": _num(pos.get("target_price"), 0.0),
                "theme": pos.get("theme", "Growth"),
                "macro_cluster": pos.get("macro_cluster", "Technology"),
                "earnings_streak_label": pos.get("earnings_streak_label") or research.get("earnings_streak_label", "N/A"),
                "fcf_yield_pct": _first_num(pos.get("fcf_yield_pct"), research.get("fcf_yield_pct")),
                "institutional_ownership_pct": _first_num(pos.get("institutional_ownership_pct"), research.get("institutional_ownership_pct")),
                "rs_alpha": _num(pos.get("rs_alpha"), 0.0),
                "bearish_divergence": bool(pos.get("bearish_divergence", False)),
                # Explicit None checks: a genuine 0 stays 0 and a missing value renders as N/A (no fake defaults)
                "risk_reward": _num(pos.get("risk_reward_ratio")),
                "cagr_5y_pct": _num(pos.get("cagr_5y_pct")),
                "sortino_5y": _num(pos.get("sortino_5y")),
                "crash_2022_ret_pct": _num(pos.get("crash_2022_ret_pct")),
                "crash_role_label": pos.get("crash_role_label") or "",
                "data_ok": pos.get("data_ok"),
            }
            if p_dict["avg_price"] > 0:
                p_dict["gain_pct"] = round(((p_dict["price"] - p_dict["avg_price"]) / p_dict["avg_price"]) * 100.0, 1)
            else:
                p_dict["gain_pct"] = 0.0
            all_positions.append(p_dict)

    # Sort positions by value descending
    all_positions.sort(key=lambda x: x["value_gbp"], reverse=True)

    # Calculate dynamic accounts metadata, reserve floor funding (C5) and pre-rendered risk KPIs (C6)
    accounts_meta = {}
    reserve_by_account: Dict[str, Dict[str, float]] = {}
    tot_reserve = 0.0
    tot_reserve_shortfall = 0.0
    short_accounts: List[str] = []
    for acc_name, acc_data in results_by_account.items():
        plan = acc_data.get("rebalancing_plan") or {}
        analyses = acc_data.get("analyses") or []
        cash_info = acc_data.get("cash") or {}
        account_shares = consolidated_view.get("account_shares") or {}
        acc_share = account_shares.get(acc_name) or {}

        invested_gbp = sum(_num(pos.get("value_gbp"), 0.0) for pos in analyses)
        cash_free_gbp = _first_num(cash_info.get("free_gbp"), cash_info.get("free"))
        if cash_free_gbp is None:
            cash_free_gbp = 0.0

        total_val = _num(acc_share.get("total_value_gbp"), 0.0)
        if total_val <= 0:
            total_val = _num(cash_info.get("total"), 0.0)
        if total_val <= 0:
            total_val = invested_gbp + cash_free_gbp

        share_pct = _num(acc_share.get("share_of_total_pct"), 0.0)
        if share_pct <= 0 and tot_net_worth > 0:
            share_pct = round((total_val / tot_net_worth) * 100.0, 1)

        rs = _reserve_status(plan, cash_info, acc_name)
        reserve_by_account[acc_name] = rs
        tot_reserve += rs["floor"]
        tot_reserve_shortfall += rs["shortfall"]
        if rs["shortfall"] > 0.005:
            short_accounts.append(acc_name)

        risk_5y = plan.get("portfolio_5y_risk_profile") or {}
        accounts_meta[acc_name] = {
            "total_value_gbp": total_val,
            "invested_gbp": invested_gbp,
            "cash_free_gbp": cash_free_gbp,
            "reserve_floor": rs["floor"],
            "reserve_funded": rs["funded"],
            "reserve_shortfall": rs["shortfall"],
            "reserve_text": _reserve_text(rs),
            "share_pct": share_pct,
            "risk_5y": risk_5y,
            "kpi": _risk_kpi_texts(risk_5y),
            "data_degraded_reason": plan.get("data_degraded_reason"),
        }

    if tot_reserve <= 0:
        tot_reserve = sum(
            _num(a.get("min_cash_reserve_gbp"), 1500.0) for k, a in ACCOUNTS.items()
        )

    if tot_reserve_shortfall > 0.005:
        reserve_all_text = f"⚠️ UNFUNDED — £{tot_reserve_shortfall:,.0f} short ({', '.join(short_accounts)})"
    else:
        reserve_all_text = "✅ Funded from free cash in every account"
    reserve_color = "#f59e0b" if tot_reserve_shortfall > 0.005 else "#38bdf8"

    consolidated_risk = consolidated_view.get("portfolio_5y_risk_profile") or {}
    accounts_meta["ALL"] = {
        "total_value_gbp": tot_net_worth,
        "invested_gbp": tot_invested,
        "cash_free_gbp": tot_cash,
        "reserve_floor": tot_reserve,
        "reserve_shortfall": tot_reserve_shortfall,
        "reserve_text": reserve_all_text,
        "share_pct": 100.0,
        "risk_5y": consolidated_risk,
        "kpi": _risk_kpi_texts(consolidated_risk),
    }
    all_kpi = accounts_meta["ALL"]["kpi"]

    # Data-quality banner (degraded plans or unavailable macro inputs)
    dq_msgs: List[str] = []
    for acc_name, acc_data in results_by_account.items():
        reason = (acc_data.get("rebalancing_plan") or {}).get("data_degraded_reason")
        if reason:
            dq_msgs.append(f"{acc_name}: {reason}")
    if macro_regime.get("macro_data_ok") is False:
        miss_txt = ", ".join(macro_missing) if macro_missing else "key inputs"
        dq_msgs.append(f"Macro: {miss_txt} unavailable — regime treated as neutral, not risk-on")
    data_quality_banner_html = ""
    if dq_msgs:
        dq_items = "".join(f"<li>{_html.escape(str(m))}</li>" for m in dq_msgs)
        data_quality_banner_html = f"""
    <div class="dq-banner">
      <div class="dq-title">⚠️ DATA QUALITY WARNING — treat today's output with caution</div>
      <ul>{dq_items}</ul>
    </div>"""

    # Household notes: ring-fenced (earmarked) money per account and outside holdings
    household_notes: List[str] = []
    for acc_name, acc_data in results_by_account.items():
        line = _earmark_line(acc_data.get("rebalancing_plan"), acc_data.get("analyses"))
        if line:
            household_notes.append(f"👤 {acc_name}: {line}")
    outside_line = _outside_holdings_line(consolidated_view)
    if outside_line:
        household_notes.append(outside_line)
    household_notes_html = ""
    if household_notes:
        notes_items = "".join(f'<div class="note-item">{n}</div>' for n in household_notes)
        household_notes_html = f'\n    <div class="household-notes">{notes_items}</div>'

    # Household mix panel (Phase 1 strategic allocation); empty when the engine supplies none of the data
    household_mix_panel_html = _household_mix_panel_html(consolidated_view, results_by_account)

    # Build dynamic action cards and status badges per account
    action_cards_by_account: Dict[str, List[str]] = {}
    action_badges_by_account: Dict[str, Dict[str, str]] = {}

    for acc_name, acc_data in results_by_account.items():
        plan = acc_data.get("rebalancing_plan") or {}
        analyses = acc_data.get("analyses") or []
        actionable_sells = [s for s in plan.get("sell_orders", []) if not s.get("is_pending")]
        pending_sells = plan.get("pending_sell_orders", [])
        buy_orders = plan.get("buy_orders", [])
        rs = reserve_by_account.get(acc_name) or _reserve_status(plan, acc_data.get("cash"), acc_name)
        reserve_val = rs["floor"]
        avail_cash = rs["free"]

        cards = []
        step = 1

        degraded_reason = plan.get("data_degraded_reason")
        if degraded_reason:
            cards.append(f"""
        <div class="action-card trim">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: var(--amber);">👤 [{acc_name.upper()}] ⚠️ DATA DEGRADED</span>
            <span class="action-badge" style="background: var(--amber-glow); color: var(--amber);">CHECK INPUTS</span>
          </div>
          <div class="action-why"><b>Reason:</b> {_html.escape(str(degraded_reason))}</div>
        </div>""")

        if not actionable_sells and not buy_orders and not pending_sells:
            action_badges_by_account[acc_name] = {
                "text": "✅ ALL SET — SIT BACK & RELAX",
                "class": "badge-emerald"
            }
            status_parts = []
            if _all_stage2_verified(analyses):
                status_parts.append("All positions are in confirmed weekly Stage 2 uptrends.")
            if rs["shortfall"] > 0.005:
                status_parts.append(
                    f"⚠️ Cash reserve UNFUNDED — £{rs['shortfall']:,.0f} short of the £{reserve_val:,.0f} floor "
                    f"(£{avail_cash:,.2f} free cash), so it is not available to withdraw until rebuilt."
                )
                card_badge = '<span class="action-badge" style="background: var(--amber-glow); color: var(--amber);">⚠️ RESERVE UNFUNDED</span>'
            else:
                status_parts.append(
                    f"Cash reserve floor of £{reserve_val:,.0f} is funded from free cash (£{avail_cash:,.2f} free)."
                )
                card_badge = '<span class="action-badge" style="background: var(--emerald-glow); color: var(--emerald);">PORTFOLIO BALANCED</span>'
            status_txt = " ".join(status_parts)
            instruction_txt = (
                "✨ No action required today."
                if rs["shortfall"] > 0.005
                else "✨ No action required today. Portfolio is fully balanced and compounding."
            )
            cards.append(f"""
        <div class="action-card reserve">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: var(--emerald);">👤 [{acc_name.upper()}] ✅ ALL SET — SIT BACK & RELAX</span>
            {card_badge}
          </div>
          <div class="action-instruction">{instruction_txt}</div>
          <div class="action-why"><b>Status:</b> {status_txt}</div>
        </div>""")
        else:
            has_trims = any("TRIM" in s.get("action", "") for s in actionable_sells)
            badge_text = "🟠 PROFIT HARVESTING & REALLOCATION ACTIVE" if has_trims else ("🔴 CAPITAL REALLOCATION ACTIVE" if actionable_sells else "🟢 CAPITAL DEPLOYMENT ACTIVE")
            action_badges_by_account[acc_name] = {
                "text": badge_text,
                "class": "badge-amber" if (has_trims or actionable_sells) else "badge-emerald"
            }

            for pso in pending_sells:
                lim_p = _num(pso.get("limit_price"), 0.0)
                curr = pso.get("currency", "USD")
                curr_sym = "$" if curr == "USD" else "£"
                lim_str = f" @ limit {curr_sym}{lim_p:,.2f}" if lim_p > 0 else ""
                pso_val = _num(pso.get("estimated_gbp"), 0.0)
                s_name = pso.get("name", pso.get("ticker"))
                s_sym = pso.get("search_symbol", pso.get("ticker"))
                shares_amt = _num(pso.get("shares"), 0.0)
                cards.append(f"""
        <div class="action-card" style="border-left: 5px solid var(--primary); background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), var(--card-bg) 60%);">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: var(--primary);">👤 [{acc_name.upper()}] ⏳ ORDER ALREADY QUEUED IN TRADING 212</span>
            <span class="action-badge" style="background: rgba(59, 130, 246, 0.15); color: var(--primary);">NO ACTION NEEDED</span>
          </div>
          <div class="action-instruction">SELL {s_name} ({s_sym}) ~{shares_amt:,.2f} shares{lim_str}</div>
          <div class="action-why"><b>Status:</b> Order is resting in Trading 212 awaiting execution (+£{pso_val:,.0f} cash when filled). No duplicate action needed.</div>
        </div>""")

            for so in actionable_sells:
                action_txt = str(so.get("action") or "SELL")
                is_trim = "TRIM" in action_txt
                card_class = "trim" if is_trim else "sell"
                h_color = "var(--amber)" if is_trim else "var(--rose)"
                b_bg = "var(--amber-glow)" if is_trim else "var(--rose-glow)"
                b_col = "var(--amber)" if is_trim else "var(--rose)"
                act_label = f"STEP {step}: {_signal_prefix(action_txt, is_trim)}{action_txt}"
                est_val = _num(so.get("estimated_gbp"), 0.0)
                cards.append(f"""
        <div class="action-card {card_class}">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: {h_color};">👤 [{acc_name.upper()}] {act_label}</span>
            <span class="action-badge" style="background: {b_bg}; color: {b_col};">BANKS +£{est_val:,.0f} CASH</span>
          </div>
          <div class="action-instruction">{so.get('app_instruction', '')}</div>
          <div class="action-why"><b>Why:</b> {so.get('plain_english_why', '')}</div>
        </div>""")
                step += 1

            for bo in buy_orders:
                b_sym = bo.get("search_symbol", bo.get("ticker"))
                b_name = bo.get("name", bo.get("ticker"))
                b_est = _num(bo.get("estimated_gbp"), 0.0)
                b_why = bo.get("plain_english_why", "")
                b_inst = bo.get("app_instruction", "")
                act_label = f"STEP {step}: 🟢 BUY {b_name} ({b_sym})"
                b_bucket = _bucket_key(bo.get("bucket"))
                bucket_badge = (
                    f' <span class="bucket-badge bucket-{b_bucket}">{b_bucket.upper()}</span>' if b_bucket else ""
                )
                b_alloc = str(bo.get("allocation_reason") or "").strip()
                alloc_html = (
                    f'\n          <div class="action-why alloc-why"><b>Allocation:</b> {_html.escape(b_alloc)}</div>'
                    if b_alloc else ""
                )
                cards.append(f"""
        <div class="action-card buy">
          <div class="action-header">
            <span style="font-weight: 800; font-size: 14px; color: var(--emerald);">👤 [{acc_name.upper()}] {act_label}{bucket_badge}</span>
            <span class="action-badge" style="background: var(--emerald-glow); color: var(--emerald);">REINVEST £{b_est:,.0f}</span>
          </div>
          <div class="action-instruction">{b_inst}</div>
          <div class="action-why"><b>Data Rationale:</b> {b_why}</div>{alloc_html}
        </div>""")
                step += 1

        action_cards_by_account[acc_name] = cards

    all_cards = []
    any_actions_active = False
    for acc_name, cards in action_cards_by_account.items():
        all_cards.extend(cards)
        if any("STEP" in c for c in cards):
            any_actions_active = True

    if not any_actions_active:
        action_badges_by_account["ALL"] = {
            "text": "✅ ALL SET — SIT BACK & RELAX",
            "class": "badge-emerald"
        }
    else:
        action_badges_by_account["ALL"] = {
            "text": "🟠 ACTIONS ACTIVE (SELECT ACCOUNT TAB)",
            "class": "badge-amber"
        }
    action_cards_by_account["ALL"] = all_cards

    # Build dynamic account tabs (the consolidated "ALL" view is the first button, never an account tab)
    tabs_buttons_html = f"""
      <button class="tab-btn active" onclick="switchAccount('ALL')">
        <span>🌐 Consolidated (£{tot_net_worth:,.0f})</span>
      </button>"""
    for acc_name, meta in accounts_meta.items():
        if acc_name == "ALL":
            continue
        tabs_buttons_html += f"""
      <button class="tab-btn" onclick="switchAccount('{acc_name}')">
        <span>👤 {acc_name}'s Account (£{meta['total_value_gbp']:,.0f} • {meta['share_pct']:.1f}%)</span>
      </button>"""

    # JSON payloads for client interactivity
    positions_json = json.dumps(all_positions)
    history_json = json.dumps(history_points)
    accounts_meta_json = json.dumps(accounts_meta, default=str)
    action_cards_json = json.dumps(action_cards_by_account)
    action_badges_json = json.dumps(action_badges_by_account)

    # Thematic Clusters (cap label follows the engine's configured cluster cap, C9)
    thematic_clusters = consolidated_view.get("thematic_breakdown") or []
    cap_raw = _num(REBALANCE_CONFIG.get("max_cluster_pct", 0.24), 0.24)
    cluster_cap_pct = cap_raw * 100.0 if cap_raw <= 1.0 else cap_raw

    # Watchlist cards: real per-ticker values from the scan (C10)
    scan_index: Dict[str, Dict[str, Any]] = {}
    for entry in (watchlist_scan or []):
        if not isinstance(entry, dict):
            continue
        for key in (entry.get("ticker"), entry.get("yf_ticker"), entry.get("search_symbol")):
            if key:
                scan_index.setdefault(str(key).upper(), entry)

    footer_wrapper_line = _wrapper_footer_line(list(results_by_account.keys()) or list(ACCOUNTS.keys()))
    footer_wrapper_html = f"<div>{footer_wrapper_line}</div>" if footer_wrapper_line else ""

    # HTML Document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading 212 Institutional Engine — Wealth & Rebalancing Terminal</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --card-hover: #172236;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.15);
      --amber: #f59e0b;
      --amber-glow: rgba(245, 158, 11, 0.15);
      --rose: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.15);
      --indigo: #6366f1;
      --purple: #8b5cf6;
      --cyan: #06b6d4;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
      -webkit-font-smoothing: antialiased;
    }}
    .max-container {{ max-width: 1400px; margin: 0 auto; }}
    
    /* Header Bar */
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }}
    .title-group h1 {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .title-group p {{
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge-emerald {{ background: var(--emerald-glow); color: var(--emerald); border: 1px solid var(--emerald); }}
    .badge-amber {{ background: var(--amber-glow); color: var(--amber); border: 1px solid var(--amber); }}
    .badge-rose {{ background: var(--rose-glow); color: var(--rose); border: 1px solid var(--rose); }}
    .badge-indigo {{ background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid #6366f1; }}
    .badge-orange {{ background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid #f97316; }}
    .badge-slate {{ background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid #64748b; }}

    /* Data-quality banner & household notes */
    .dq-banner {{
      background: rgba(245, 158, 11, 0.10);
      border: 1px solid var(--amber);
      border-radius: 12px;
      padding: 12px 18px;
      margin-bottom: 20px;
      color: #fde68a;
      font-size: 13px;
    }}
    .dq-title {{ font-weight: 800; margin-bottom: 6px; color: var(--amber); }}
    .dq-banner ul {{ margin-left: 18px; }}
    .household-notes {{
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 10px 16px;
      margin: -12px 0 24px 0;
      font-size: 12.5px;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    /* Household mix (strategic allocation: core / satellites / defensive) */
    .mix-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .mix-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 20px;
      overflow-x: auto;
    }}
    .mix-card.mix-wide {{ margin-bottom: 28px; }}
    .mix-head {{ font-size: 14px; font-weight: 800; margin-bottom: 6px; }}
    .mix-sub {{ font-size: 12px; color: var(--text-muted); margin: 4px 0 8px 0; }}
    .mix-line {{ font-size: 13px; margin-top: 10px; }}
    .mix-explainer {{
      font-size: 12.5px;
      color: var(--text-muted);
      margin-top: 10px;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px;
      padding: 8px 12px;
    }}
    .mix-table {{ font-size: 12.5px; margin-top: 6px; }}
    .mix-table th {{ cursor: default; padding: 8px 10px; }}
    .mix-table td {{ padding: 9px 10px; vertical-align: top; }}
    .mix-num {{ text-align: right; white-space: nowrap; }}
    .mix-label {{ font-weight: 700; color: #fff; }}
    .mix-sub-cell {{ font-size: 11px; color: var(--text-muted); font-weight: 400; margin-top: 2px; }}
    .mix-acc {{ font-size: 13px; padding: 8px 0; border-top: 1px solid rgba(255, 255, 255, 0.05); }}
    .mix-acc ul {{ margin: 4px 0 0 18px; color: var(--text-muted); font-size: 12.5px; }}
    .chip {{
      display: inline-block;
      margin-top: 4px;
      padding: 1px 8px;
      border-radius: 9999px;
      font-size: 10.5px;
      font-weight: 800;
      white-space: nowrap;
    }}
    .chip-under {{ background: var(--amber-glow); color: var(--amber); border: 1px solid var(--amber); }}
    .chip-over {{ background: var(--rose-glow); color: var(--rose); border: 1px solid var(--rose); }}
    .chip-in_band {{ background: var(--emerald-glow); color: var(--emerald); border: 1px solid var(--emerald); }}
    .bucket-badge {{
      display: inline-block;
      margin-left: 6px;
      padding: 1px 8px;
      border-radius: 6px;
      font-size: 10.5px;
      font-weight: 800;
      letter-spacing: 0.04em;
      vertical-align: middle;
    }}
    .bucket-core {{ background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid #6366f1; }}
    .bucket-satellite {{ background: rgba(139, 92, 246, 0.15); color: #c4b5fd; border: 1px solid #8b5cf6; }}
    .bucket-defensive {{ background: rgba(6, 182, 212, 0.15); color: #67e8f9; border: 1px solid #06b6d4; }}
    .alloc-why {{ margin-top: 6px; color: #a5b4fc; }}

    /* Macro Marquee */
    .macro-bar {{
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 24px;
      font-size: 13px;
    }}
    .macro-item {{ display: flex; align-items: center; gap: 8px; }}
    .macro-label {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .macro-val {{ font-weight: 700; color: #fff; }}

    /* KPI Cards Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 20px;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: #2d3b55;
    }}
    .kpi-title {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}
    .kpi-sub {{
      font-size: 12px;
      margin-top: 6px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    /* View Mode Tabs */
    .tabs-bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
      overflow-x: auto;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
      white-space: nowrap;
    }}
    .tab-btn:hover {{
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }}
    .tab-btn.active {{
      background: #1e293b;
      color: #fff;
      border-color: #3b82f6;
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.2);
    }}

    /* Section Layout */
    .section-title {{
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.01em;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    /* Action Cards */
    .action-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}
    .action-card {{
      background: var(--card-bg);
      border-radius: 14px;
      padding: 20px;
      border: 1px solid var(--card-border);
      position: relative;
    }}
    .action-card.trim {{
      border-left: 5px solid var(--amber);
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.06), var(--card-bg) 60%);
    }}
    .action-card.buy {{
      border-left: 5px solid var(--emerald);
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.06), var(--card-bg) 60%);
    }}
    .action-card.reserve {{
      border-left: 5px solid var(--cyan);
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.06), var(--card-bg) 60%);
    }}
    .action-card.sell {{
      border-left: 5px solid var(--rose);
      background: linear-gradient(135deg, rgba(244, 63, 94, 0.06), var(--card-bg) 60%);
    }}
    .action-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .action-badge {{
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .action-instruction {{
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 14px;
      color: #fff;
      font-weight: 600;
      margin: 12px 0;
    }}
    .action-why {{
      font-size: 12.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }}

    /* Table */
    .table-container {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      overflow-x: auto;
      margin-bottom: 32px;
    }}
    .table-toolbar {{
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      border-bottom: 1px solid var(--card-border);
    }}
    .search-input {{
      background: #090d16;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      width: 260px;
      outline: none;
    }}
    .search-input:focus {{ border-color: var(--primary); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      padding: 12px 16px;
      background: #0c121e;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--card-border);
      cursor: pointer;
      user-select: none;
    }}
    th:hover {{ color: #fff; }}
    td {{
      padding: 14px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .ticker-cell {{ font-weight: 800; font-size: 14px; display: flex; align-items: center; gap: 8px; }}
    .company-cell {{ color: var(--text-muted); font-size: 12px; }}

    /* MPT Cluster Cards */
    .mpt-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}
    .mpt-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .mpt-header {{ display: flex; justify-content: space-between; font-size: 13px; font-weight: 700; margin-bottom: 10px; }}
    .progress-track {{
      background: #1e293b;
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 8px;
      position: relative;
    }}
    .progress-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s ease; }}
    .mpt-meta {{ display: flex; justify-content: space-between; font-size: 11.5px; color: var(--text-muted); }}

    /* Equity Curve Chart */
    .chart-box {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 32px;
    }}
    .chart-container {{
      width: 100%;
      height: 220px;
      position: relative;
    }}
    svg.equity-svg {{
      width: 100%;
      height: 100%;
      overflow: visible;
    }}

    /* Watchlist Grid */
    .watch-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .watch-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      transition: border-color 0.15s ease;
    }}
    .watch-card:hover {{ border-color: #3b82f6; }}
    .watch-top {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }}
    .watch-title {{ font-size: 15px; font-weight: 800; }}
    .watch-theme {{ font-size: 11px; color: var(--text-muted); margin-bottom: 12px; }}
    .watch-metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: #090d16;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 11.5px;
    }}
    .watch-m-item {{ display: flex; flex-direction: column; gap: 2px; }}
    .watch-m-label {{ color: var(--text-muted); font-size: 10px; text-transform: uppercase; }}
    .watch-m-val {{ font-weight: 700; color: #fff; }}

    /* Footer */
    .footer {{
      border-top: 1px solid var(--card-border);
      padding-top: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 12px;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>
  <div class="max-container">

    <!-- Header -->
    <header class="header">
      <div class="title-group">
        <h1>
          <span>🔬 Institutional Quantitative Dashboard</span>
          <span class="badge badge-emerald">LIVE ENGINE</span>
        </h1>
        <p>{accounts_title_label} Trading 212 Portfolio • Daily Buy/Sell Recommendations, Trims, & Risk Parity Engine</p>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 13px; font-weight: 700; color: #fff;">Anchor FX: {gbp_rate_txt}</div>
        <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 2px;">Last Audit: {now_str}</div>
      </div>
    </header>
{data_quality_banner_html}

    <!-- Macro Telemetry Bar -->
    <div class="macro-bar">
      <div class="macro-item">
        <span class="macro-label">Backdrop:</span>
        <span class="badge {climate_badge_cls}">{climate}{score_txt}</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Crash Defcon:</span>
        <span class="badge {defcon_badge_cls}">{crash_defcon}</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">VIX:</span>
        <span class="macro-val">{vix_txt} ({term_txt})</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">10Y Yield:</span>
        <span class="macro-val">{us10y_txt} (10Y–3M Spread: {spread_txt})</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Dist. Days:</span>
        <span class="macro-val">SPY {spy_dist_txt} / QQQ {qqq_dist_txt}</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">High-Beta:</span>
        <span class="macro-val">{hb_txt}</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Semi Lead:</span>
        <span class="macro-val">{semi_txt}</span>
      </div>
      <div class="macro-item">
        <span class="macro-label">Crude Oil:</span>
        <span class="macro-val">{oil_txt}</span>
      </div>{partial_macro_html}
    </div>

    <!-- Top KPI Row -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">
          <span>Consolidated Net Worth</span>
          <span>💎</span>
        </div>
        <div class="kpi-value" id="kpiNetWorth">£{tot_net_worth:,.2f}</div>
        <div class="kpi-sub">
          <span style="color: {dod_color}; font-weight: 700;">{dod_sign}£{dod_gbp:,.2f} ({dod_pct:+.2f}%)</span>
          <span>vs yesterday</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Total Invested Equity</span>
          <span>📈</span>
        </div>
        <div class="kpi-value" id="kpiInvested">£{tot_invested:,.2f}</div>
        <div class="kpi-sub">
          <span>Across {len(all_positions)} audited positions</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Liquid Free Cash</span>
          <span>💰</span>
        </div>
        <div class="kpi-value" id="kpiCash">£{tot_cash:,.2f}</div>
        <div class="kpi-sub">
          <span>Pending Blocked: £{tot_blocked:,.2f}</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Protected Cash Floor</span>
          <span>🛡️</span>
        </div>
        <div class="kpi-value" style="color: {reserve_color};" id="kpiReserve">£{tot_reserve:,.2f}</div>
        <div class="kpi-sub" id="kpiReserveSub">
          <span>{reserve_all_text}</span>
        </div>
      </div>
    </div>
{household_notes_html}
{household_mix_panel_html}

    <!-- Account View Selector Tabs -->
    <div class="tabs-bar">
{tabs_buttons_html}
    </div>

    <!-- 5-Year Horizon Crash-Protection & Stress-Test Banner -->
    <div class="kpi-grid" style="margin-bottom: 24px;">
      <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.35);">
        <div class="kpi-title">
          <span>5-Year Sortino & CAGR</span>
          <span>🚀</span>
        </div>
        <div class="kpi-value" style="color: var(--emerald); font-size: 22px;" id="kpiSortino5Y">{all_kpi['sortino']}</div>
        <div class="kpi-sub" id="kpiCagr5Y">
          <span>{all_kpi['cagr']}</span>
        </div>
      </div>

      <div class="kpi-card" style="border-color: rgba(244, 63, 94, 0.35);">
        <div class="kpi-title">
          <span>2022 Crash Stress-Test</span>
          <span>🛡️</span>
        </div>
        <div class="kpi-value" style="color: var(--amber); font-size: 22px;" id="kpiCrash2022">{all_kpi['crash']}</div>
        <div class="kpi-sub" id="kpiCrash2022Sub">
          <span>{all_kpi['crash_sub']}</span>
        </div>
      </div>

      <div class="kpi-card" style="border-color: rgba(59, 130, 246, 0.35);">
        <div class="kpi-title">
          <span>All-Weather Crash Shield</span>
          <span>🏰</span>
        </div>
        <div class="kpi-value" style="color: var(--primary); font-size: 22px;" id="kpiShield5Y">{all_kpi['shield']}</div>
        <div class="kpi-sub" id="kpiShield5YSub">
          <span>{all_kpi['shield_sub']}</span>
        </div>
        <div class="kpi-sub" id="kpiDefensive5Y">
          <span>{all_kpi['defensive']}</span>
        </div>
      </div>

      <div class="kpi-card" style="border-color: rgba(139, 92, 246, 0.35);">
        <div class="kpi-title">
          <span>5-Year (2031) Projection</span>
          <span>🔮</span>
        </div>
        <div class="kpi-value" style="color: #c4b5fd; font-size: 17px;" id="kpiProj5Y">{all_kpi['proj']}</div>
        <div class="kpi-sub" id="kpiProj5YSub">
          <span>{all_kpi['proj_sub']}</span>
        </div>
      </div>
    </div>

    <!-- 2-Minute Action Center -->
    <div class="section-title">
      <span>⚡ 2-Minute Action Center — What You Need To Do Today</span>
      <span class="badge {action_badges_by_account['ALL']['class']}" id="actionStatusBadge">{action_badges_by_account['ALL']['text']}</span>
    </div>

    <div class="action-grid" id="actionCardsContainer">
      <!-- Generated dynamically by JS according to selected account -->
    </div>

    <!-- MPT Thematic Concentration Meters -->
    <div class="section-title">
      <span>🌐 Modern Portfolio Theory (MPT) Thematic Risk Exposure</span>
      <span style="font-size: 12px; color: var(--text-muted); font-weight: normal;">({cluster_cap_pct:.0f}% Maximum Cluster Cap)</span>
    </div>

    <div class="mpt-grid">
"""
    for tb in thematic_clusters:
        c_name = tb.get("cluster", "Unclassified")
        c_val = _num(tb.get("value_gbp"), 0.0)
        c_pct = _num(tb.get("weight_pct"), 0.0)
        is_sat = c_pct >= cluster_cap_pct
        bar_color = "#ef4444" if is_sat else ("#10b981" if c_pct >= 15.0 else "#f59e0b")
        status_label = f"🔴 AT/ABOVE {cluster_cap_pct:.0f}% CAP" if is_sat else ("🟢 CORE PILLAR" if c_pct >= 15.0 else "🟡 UNDERWEIGHTED")
        html += f"""
      <div class="mpt-card">
        <div class="mpt-header">
          <span>{c_name}</span>
          <span style="color: {bar_color};">{c_pct:.1f}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: {min(100.0, c_pct * 2.5)}%; background: {bar_color};"></div>
        </div>
        <div class="mpt-meta">
          <span>£{c_val:,.0f} value</span>
          <span style="font-weight: 700; color: {bar_color};">{status_label}</span>
        </div>
      </div>
"""

    html += f"""
    </div>

    <!-- Historical Net Worth Equity Curve -->
    <div class="chart-box">
      <div class="section-title" style="margin-bottom: 12px;">
        <span>📈 Historical Portfolio Equity Curve & Trajectory</span>
        <span style="font-size: 12px; color: var(--text-muted); font-weight: normal;">{sessions_label}</span>
      </div>
      <div class="chart-container" id="chartContainer">
        <!-- Rendered via inline SVG -->
      </div>
    </div>

    <!-- Forensic Holdings Dossier Table -->
    <div class="section-title">
      <span>🔬 Active Holdings Forensic Quality & Technical Audit</span>
    </div>

    <div class="table-container">
      <div class="table-toolbar">
        <input type="text" class="search-input" id="tableSearch" placeholder="🔍 Search holding or symbol..." oninput="filterTable()">
        <div style="display: flex; gap: 8px; font-size: 12px; color: var(--text-muted); align-items: center;">
          <span>Showing <b id="visibleCount">15</b> positions</span>
        </div>
      </div>
      <table id="holdingsTable">
        <thead>
          <tr>
            <th onclick="sortTable('ticker')">Holding</th>
            <th onclick="sortTable('account')">Account</th>
            <th onclick="sortTable('price')">Price</th>
            <th onclick="sortTable('gain_pct')">Gain / Return</th>
            <th onclick="sortTable('unrealized_pnl')">P&L (£)</th>
            <th onclick="sortTable('value_gbp')">Value (GBP)</th>
            <th onclick="sortTable('weight_pct')">Weight</th>
            <th onclick="sortTable('sortino_5y')">5Y Sortino</th>
            <th onclick="sortTable('crash_2022_ret_pct')">2022 Crash</th>
            <th onclick="sortTable('piotroski')">Quality Checklist (0-9)</th>
            <th onclick="sortTable('altman_z')">Altman Z</th>
            <th onclick="sortTable('risk_reward')">R:R</th>
            <th>Signal & Strategy Status</th>
          </tr>
        </thead>
        <tbody id="tableBody">
          <!-- Populated by JS -->
        </tbody>
      </table>
    </div>

    <!-- Curated Cross-Sector Watchlist -->
    <div class="section-title">
      <span>🎯 Curated Data-Driven Watchlist (Highest Probability, Lowest Risk)</span>
      {excluded_badge_html}
    </div>

    <div class="watch-grid">
"""
    for w in WATCHLIST:
        t = w["ticker"]
        name = w["name"]
        theme = w["theme"]
        entry = scan_index.get(str(t).upper())
        research = (entry or {}).get("research") or {}
        t_up = str(t).upper()
        is_gold = any(k in t_up for k in _GOLD_TICKER_KEYS)
        is_fund = bool(research.get("is_etf") or (entry or {}).get("is_etf")) or any(k in t_up for k in _FUND_TICKER_KEYS)
        fund_na = "N/A (gold)" if is_gold else "N/A (fund)"

        data_ok = (entry or {}).get("data_ok")
        if data_ok is None and entry is not None:
            data_ok = research.get("data_ok")
        if entry is None:
            badge_html = '<span class="badge badge-slate">NOT SCANNED</span>'
        elif data_ok is True:
            badge_html = '<span class="badge badge-indigo">SCREENED</span>'
        elif data_ok is False:
            badge_html = '<span class="badge badge-amber">DATA UNAVAILABLE</span>'
        else:
            badge_html = '<span class="badge badge-slate">UNVERIFIED</span>'

        if entry is None or data_ok is False:
            quality_txt = solvency_txt = trend_txt = rr_txt = "N/A"
        else:
            if is_fund:
                quality_txt = solvency_txt = fund_na
            else:
                p_score = _first_num(entry.get("piotroski_score"), research.get("piotroski_score"))
                quality_txt = f"{p_score:.0f}/9" if p_score is not None else "N/A"
                z_val = _first_num(entry.get("altman_z"), research.get("altman_z"))
                solvency_txt = f"Altman Z {z_val:.1f}" if z_val is not None else "N/A"
            stage2 = entry.get("is_weekly_stage_2")
            trend_label = str(entry.get("weekly_trend_label") or "").split("(")[0].strip()
            if trend_label:
                trend_txt = trend_label
            elif stage2 is True:
                trend_txt = "Weekly Stage 2"
            elif stage2 is False:
                trend_txt = "Not Weekly Stage 2"
            else:
                trend_txt = "N/A"
            rr_val = _num(entry.get("risk_reward_ratio"))
            if is_fund:
                rr_txt = fund_na
            else:
                rr_txt = f"R:R {rr_val:.1f}:1" if rr_val is not None else "N/A"

        html += f"""
      <div class="watch-card">
        <div class="watch-top">
          <div>
            <div class="watch-title">{t}</div>
            <div style="font-size: 12px; color: #fff;">{name}</div>
          </div>
          {badge_html}
        </div>
        <div class="watch-theme">{theme}</div>
        <div class="watch-metrics">
          <div class="watch-m-item">
            <span class="watch-m-label">Quality Checklist (0-9)</span>
            <span class="watch-m-val">{quality_txt}</span>
          </div>
          <div class="watch-m-item">
            <span class="watch-m-label">Solvency</span>
            <span class="watch-m-val">{solvency_txt}</span>
          </div>
          <div class="watch-m-item">
            <span class="watch-m-label">Trend Structure</span>
            <span class="watch-m-val">{trend_txt}</span>
          </div>
          <div class="watch-m-item">
            <span class="watch-m-label">Expectancy</span>
            <span class="watch-m-val">{rr_txt}</span>
          </div>
        </div>
      </div>
"""

    html += f"""
    </div>

    <!-- Footer -->
    <footer class="footer">
      <div>Trading 212 Daily Recommendation & Rebalancing Engine • {accounts_title_label} • Model version: {_html.escape(str(model_version))}</div>
      <div>Capital Isolation Enforced: Execute orders directly inside your Trading 212 app.</div>
      {footer_wrapper_html}
    </footer>

  </div>

  <script>
    const POSITIONS = {positions_json};
    const HISTORY = {history_json};
    const ACCOUNTS_META = {accounts_meta_json};
    const ACTION_CARDS = {action_cards_json};
    const ACTION_BADGES = {action_badges_json};
    let currentAccount = 'ALL';
    let sortCol = 'value_gbp';
    let sortAsc = false;

    function formatGBP(val) {{
      return '£' + Number(val ?? 0).toLocaleString('en-GB', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
    }}

    function setText(id, val) {{
      const el = document.getElementById(id);
      if (el) el.textContent = val ?? 'N/A';
    }}

    function setSpan(id, val) {{
      const el = document.getElementById(id);
      if (!el) return;
      const span = document.createElement('span');
      span.textContent = val ?? 'N/A';
      el.replaceChildren(span);
    }}

    function switchAccount(acc) {{
      currentAccount = acc;
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.textContent.includes(acc === 'ALL' ? 'Consolidated' : acc));
      }});
      updateKPIs();
      renderActionCards();
      renderTable();
      renderChart();
    }}

    function updateKPIs() {{
      const kpiNetWorth = document.getElementById('kpiNetWorth');
      const kpiInvested = document.getElementById('kpiInvested');
      const kpiCash = document.getElementById('kpiCash');
      const kpiReserve = document.getElementById('kpiReserve');

      if (currentAccount === 'ALL') {{
        kpiNetWorth.textContent = '£{tot_net_worth:,.2f}';
        kpiInvested.textContent = '£{tot_invested:,.2f}';
        kpiCash.textContent = '£{tot_cash:,.2f}';
        kpiReserve.textContent = '£{tot_reserve:,.2f}';
      }} else if (ACCOUNTS_META[currentAccount]) {{
        const acc = ACCOUNTS_META[currentAccount];
        kpiNetWorth.textContent = formatGBP(acc.total_value_gbp);
        kpiInvested.textContent = formatGBP(acc.invested_gbp);
        kpiCash.textContent = formatGBP(acc.cash_free_gbp);
        kpiReserve.textContent = formatGBP(acc.reserve_floor);
      }}

      const accMeta = ACCOUNTS_META[currentAccount] ?? ACCOUNTS_META['ALL'];
      if (accMeta) {{
        const unfunded = (accMeta.reserve_shortfall ?? 0) > 0.005;
        kpiReserve.style.color = unfunded ? '#f59e0b' : '#38bdf8';
        setSpan('kpiReserveSub', accMeta.reserve_text);
      }}
      if (accMeta && accMeta.kpi) {{
        const k = accMeta.kpi;
        setText('kpiSortino5Y', k.sortino);
        setSpan('kpiCagr5Y', k.cagr);
        setText('kpiCrash2022', k.crash);
        setSpan('kpiCrash2022Sub', k.crash_sub);
        setText('kpiShield5Y', k.shield);
        setSpan('kpiShield5YSub', k.shield_sub);
        setSpan('kpiDefensive5Y', k.defensive);
        setText('kpiProj5Y', k.proj);
        setSpan('kpiProj5YSub', k.proj_sub);
      }}
    }}

    function renderActionCards() {{
      const container = document.getElementById('actionCardsContainer');
      const badge = document.getElementById('actionStatusBadge');

      const cardsList = ACTION_CARDS[currentAccount] || [];
      container.innerHTML = cardsList.join('\\n');

      const badgeInfo = ACTION_BADGES[currentAccount] || {{ text: '✅ ALL SET', class: 'badge-emerald' }};
      if (badge) {{
        badge.className = 'badge ' + badgeInfo.class;
        badge.textContent = badgeInfo.text;
      }}
    }}

    function renderTable() {{
      const tbody = document.getElementById('tableBody');
      const searchVal = (document.getElementById('tableSearch').value || '').toLowerCase();
      let filtered = POSITIONS.filter(p => {{
        if (currentAccount !== 'ALL' && p.account !== currentAccount) return false;
        if (searchVal && !p.ticker.toLowerCase().includes(searchVal) && !p.name.toLowerCase().includes(searchVal)) return false;
        return true;
      }});

      filtered.sort((a, b) => {{
        let vA = a[sortCol];
        let vB = b[sortCol];
        const aMissing = (vA === null || vA === undefined);
        const bMissing = (vB === null || vB === undefined);
        if (aMissing || bMissing) return aMissing === bMissing ? 0 : (aMissing ? 1 : -1);
        if (typeof vA === 'string') vA = vA.toLowerCase();
        if (typeof vB === 'string') vB = vB.toLowerCase();
        if (vA < vB) return sortAsc ? -1 : 1;
        if (vA > vB) return sortAsc ? 1 : -1;
        return 0;
      }});

      document.getElementById('visibleCount').textContent = filtered.length;

      const naSpan = '<span style="color:#64748b">N/A</span>';
      const normSig = (s) => String(s ?? '').toUpperCase().replace(/[^A-Z0-9%]+/g, ' ').trim();

      let html = '';
      for (const p of filtered) {{
        const pnlColor = p.unrealized_pnl >= 0 ? '#10b981' : '#f43f5e';
        const pnlSign = p.unrealized_pnl >= 0 ? '+' : '';
        const gainSign = p.gain_pct >= 0 ? '+' : '';
        
        const pioBadge = (p.piotroski !== null && p.piotroski !== undefined) ? `<span class="badge ${{p.piotroski >= 8 ? 'badge-emerald' : 'badge-amber'}}">${{p.piotroski}}/9</span>` : naSpan;
        const altBadge = (p.altman_z !== null && p.altman_z !== undefined) ? `<span style="font-weight:700; color:${{p.altman_z >= 3.0 ? '#10b981' : '#f59e0b'}}">${{p.altman_z.toFixed(1)}}</span>` : naSpan;
        const sortVal = p.sortino_5y ?? null;
        const cagrVal = p.cagr_5y_pct ?? null;
        const cagrSpan = cagrVal !== null ? ` <span style="font-size:10px; color:#94a3b8;">(${{cagrVal >= 0 ? '+' : ''}}${{cagrVal.toFixed(0)}}%/y)</span>` : '';
        const sort5Badge = sortVal !== null ? `<span style="font-weight:700; color:${{sortVal >= 1.5 ? '#10b981' : (sortVal >= 0.9 ? '#38bdf8' : '#f43f5e')}}">${{sortVal.toFixed(2)}}</span>${{cagrSpan}}` : naSpan + cagrSpan;
        const crashVal = p.crash_2022_ret_pct ?? null;
        const crash22Color = crashVal === null ? '#64748b' : (crashVal >= 0 ? '#10b981' : (crashVal >= -20 ? '#38bdf8' : '#f43f5e'));
        const crash22Badge = crashVal !== null ? `<span style="font-weight:700; color:${{crash22Color}}">${{crashVal >= 0 ? '+' : ''}}${{crashVal.toFixed(1)}}%</span>` : naSpan;
        const rrVal = p.risk_reward ?? null;
        const rrCell = rrVal !== null ? `<b>${{rrVal.toFixed(1)}}:1</b>` : naSpan;
        const signalLine = (p.signal && normSig(p.signal) !== normSig(p.plan_status)) ? `<div style="font-size: 10.5px; color: #94a3b8; margin-top: 4px;">${{p.signal}}</div>` : '';

        let statusBadgeClass = 'badge-indigo';
        if (p.plan_status.includes('TRIM')) statusBadgeClass = 'badge-amber';
        if (p.plan_status.includes('BUY')) statusBadgeClass = 'badge-emerald';
        if (p.plan_status.includes('SELL')) statusBadgeClass = 'badge-rose';

        html += `
          <tr>
            <td>
              <div class="ticker-cell">
                <span>${{p.symbol}}</span>
                <span style="font-size: 11px; font-weight: normal; color: var(--text-muted);">(${{p.ticker}})</span>
              </div>
              <div class="company-cell">
                <span>${{p.name}}</span>
                ${{p.crash_role_label ? `<span style="font-size: 10px; background: rgba(16, 185, 129, 0.12); color: #34d399; padding: 1px 6px; border-radius: 4px; margin-left: 6px; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.3);">${{p.crash_role_label}}</span>` : ''}}
                ${{p.earnings_streak_label && p.earnings_streak_label !== 'N/A' ? `<span style="font-size: 10px; background: #f0fdf4; color: #166534; padding: 1px 6px; border-radius: 4px; margin-left: 6px; font-weight: 700; border: 1px solid #bbf7d0;">${{p.earnings_streak_label}}</span>` : ''}}
                ${{p.fcf_yield_pct !== null && p.fcf_yield_pct !== undefined ? `<span style="font-size: 10px; background: #f8fafc; color: #334155; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 700; border: 1px solid #e2e8f0;">FCF: ${{p.fcf_yield_pct}}%</span>` : ''}}
                ${{p.roic !== null && p.roic !== undefined && p.roic > 0 ? `<span style="font-size: 10px; background: #ecfdf5; color: #047857; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 700; border: 1px solid #bbf7d0;">ROIC: ${{p.roic.toFixed(1)}}%</span>` : ''}}
                ${{p.options_pcr_vol !== null && p.options_pcr_vol !== undefined && p.options_pcr_vol < 0.70 ? `<span style="font-size: 10px; background: #e0f2fe; color: #0369a1; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 700; border: 1px solid #bae6fd;">Call Skew (${{p.options_pcr_vol.toFixed(2)}})</span>` : ''}}
                ${{p.insider_label && p.insider_label.includes('Buying') ? `<span style="font-size: 10px; background: #f0fdf4; color: #166534; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 800; border: 1px solid #bbf7d0;">👥 Insider Buying</span>` : ''}}
                ${{p.bearish_divergence ? `<span style="font-size: 10px; background: #fef3c7; color: #b45309; padding: 1px 6px; border-radius: 4px; margin-left: 4px; font-weight: 800; border: 1px solid #fde68a;">⚠️ Bearish Div</span>` : ''}}
              </div>
            </td>
            <td><span class="badge" style="background:#1e293b; color:#94a3b8;">${{p.account}}</span></td>
            <td><b>${{p.currency === 'USD' ? '$' : '£'}}${{p.price.toFixed(2)}}</b></td>
            <td style="color: ${{pnlColor}}; font-weight: 700;">${{gainSign}}${{p.gain_pct.toFixed(1)}}%</td>
            <td style="color: ${{pnlColor}}; font-weight: 700;">${{pnlSign}}£${{Math.abs(p.unrealized_pnl).toLocaleString('en-GB', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}</td>
            <td><b>£${{p.value_gbp.toLocaleString('en-GB', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}</b></td>
            <td>${{p.weight_pct.toFixed(1)}}%</td>
            <td>${{sort5Badge}}</td>
            <td>${{crash22Badge}}</td>
            <td>${{pioBadge}}</td>
            <td>${{altBadge}}</td>
            <td>${{rrCell}}</td>
            <td>
              <span class="badge ${{statusBadgeClass}}">${{p.plan_status}}</span>
              ${{signalLine}}
            </td>
          </tr>
        `;
      }}
      tbody.innerHTML = html;
    }}

    function filterTable() {{
      renderTable();
    }}

    function sortTable(col) {{
      if (sortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortCol = col;
        sortAsc = false;
      }}
      renderTable();
    }}

    function renderChart() {{
      const container = document.getElementById('chartContainer');
      if (!HISTORY || HISTORY.length < 2) {{
        container.innerHTML = '<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 40px;">Accumulating historical session records...</div>';
        return;
      }}

      const w = container.clientWidth || 1000;
      const h = 200;
      const padL = 60;
      const padR = 20;
      const padT = 20;
      const padB = 30;

      const key = currentAccount === 'ALL' ? 'total' : currentAccount.toLowerCase();
      const vals = HISTORY.map(d => d[key]);
      const minV = Math.min(...vals) * 0.995;
      const maxV = Math.max(...vals) * 1.005;

      const pts = HISTORY.map((d, i) => {{
        const x = padL + (i / (HISTORY.length - 1)) * (w - padL - padR);
        const y = padT + (1 - (d[key] - minV) / (maxV - minV)) * (h - padT - padB);
        return {{ x, y, date: d.date, val: d[key] }};
      }});

      const pathD = pts.reduce((acc, p, i) => i === 0 ? `M ${{p.x}} ${{p.y}}` : `${{acc}} L ${{p.x}} ${{p.y}}`, '');
      const fillD = `${{pathD}} L ${{pts[pts.length-1].x}} ${{h - padB}} L ${{pts[0].x}} ${{h - padB}} Z`;

      let dots = '';
      for (const p of pts) {{
        dots += `<circle cx="${{p.x}}" cy="${{p.y}}" r="4" fill="#3b82f6" stroke="#fff" stroke-width="2" />`;
        dots += `<text x="${{p.x}}" y="${{h - 10}}" text-anchor="middle" font-size="10" fill="#64748b">${{p.date.slice(5)}}</text>`;
      }}

      container.innerHTML = `
        <svg class="equity-svg" viewBox="0 0 ${{w}} ${{h}}">
          <defs>
            <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.35"/>
              <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
          <line x1="${{padL}}" y1="${{padT}}" x2="${{w - padR}}" y2="${{padT}}" stroke="#1e293b" stroke-dasharray="3,3" />
          <line x1="${{padL}}" y1="${{h - padB}}" x2="${{w - padR}}" y2="${{h - padB}}" stroke="#1e293b" />
          <text x="${{padL - 10}}" y="${{padT + 5}}" text-anchor="end" font-size="10" fill="#64748b">£${{Math.round(maxV).toLocaleString()}}</text>
          <text x="${{padL - 10}}" y="${{h - padB}}" text-anchor="end" font-size="10" fill="#64748b">£${{Math.round(minV).toLocaleString()}}</text>
          <path d="${{fillD}}" fill="url(#chartGrad)" />
          <path d="${{pathD}}" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" />
          ${{dots}}
        </svg>
      `;
    }}

    window.addEventListener('resize', renderChart);
    // Initial boot
    updateKPIs();
    renderActionCards();
    renderTable();
    setTimeout(renderChart, 50);
  </script>
</body>
</html>
"""
    return html
