"""Survivorship-free household crash study (evidence for CRASH_SCENARIOS and REBALANCE_POLICY).

Usage:
    ./.venv/bin/python3 scripts/crash_study.py [--out crash_study.md] [--cache .crash_study_cache]

Data (all free; downloaded once into --cache):
  * Kenneth French Data Library (CRSP-based, includes delisted firms): US market + T-bills, 12 industries
    (Business Equipment = tech, Health), 2x3 size x prior-return (momentum) and size x operating profitability
    portfolios, developed-markets market factor.
  * Bank of England IADB: month-end USD per GBP (XUMLUSS) and monthly-average Bank Rate (IUMABEDR).
  * datahub.io core/gold-prices (London gold, USD/oz, monthly) spliced with Yahoo GC=F / GLD; Yahoo ^SOX (semis).

Method: monthly, GBP total returns, after fund fees/turnover drags and 0.15% per unit of rebalancing turnover;
weights decided at month-end t earn month t+1 (no look-ahead); +/-5pp drift bands; satellite trend rule = sleeve
total-return index vs its 10-month simple moving average. Textbook parameters only; nothing is fitted.
Pre-registered decision rule: highest CAGR subject to (1) max drawdown no deeper than 100% equity and (2) no
chronological split trailing 100% equity by more than 1.5 pp/yr.
"""

from __future__ import annotations

import argparse
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
BOE = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/1975"
       "&Dateto=now&SeriesCodes=XUMLUSS,IUMABEDR&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
GOLD = "https://datahub.io/core/gold-prices/r/monthly.csv"
DRAG = {"core": 0.0020, "gold": 0.0012, "cash": 0.0010, "tech": 0.0030, "qm": 0.0055}
TRADE_COST = 0.0015
CRISES = {"Dot-com": ("2000-03-31", "2002-09-30"), "GFC": ("2007-10-31", "2009-02-28"), "2022": ("2021-12-31", "2022-12-31")}


def _get(cache: Path, name: str, url: str) -> bytes:
    path = cache / name
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    r = requests.get(url, timeout=60, headers=UA)
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content


def french_monthly(cache: Path, zip_name: str, value_weighted: bool = True) -> pd.DataFrame:
    """First monthly block of a French CSV zip (the value-weighted one when the file has several), as decimals."""
    archive = zipfile.ZipFile(io.BytesIO(_get(cache, zip_name, FRENCH + zip_name)))
    text = archive.read(archive.namelist()[0]).decode("latin1")
    lines, blocks, title, i = text.splitlines(), [], "", 0
    while i < len(lines):
        s = lines[i].strip()
        if s and not s[:1].isdigit() and "," not in s:
            title = s
        elif s.startswith(","):
            header = [h.strip() for h in s.split(",")][1:]
            rows, j = [], i + 1
            while j < len(lines) and lines[j].split(",")[0].strip().isdigit():
                rows.append([p.strip() for p in lines[j].split(",")])
                j += 1
            if rows and len(rows[0][0]) == 6:
                idx = pd.PeriodIndex([r[0] for r in rows], freq="M").to_timestamp(how="end").normalize()
                df = pd.DataFrame([[float(x) for x in r[1:1 + len(header)]] for r in rows], index=idx, columns=header)
                blocks.append((title.lower(), df.mask(df <= -99.0) / 100.0))
            i = j
            continue
        i += 1
    if value_weighted:
        for t, df in blocks:
            if "value weight" in t and "monthly" in t:
                return df
    return blocks[0][1]


def build_panel(cache: Path) -> pd.DataFrame:
    ff = french_monthly(cache, "F-F_Research_Data_Factors_CSV.zip", value_weighted=False)
    ind = french_monthly(cache, "12_Industry_Portfolios_CSV.zip")
    mom = french_monthly(cache, "6_Portfolios_ME_Prior_12_2_CSV.zip")
    op = french_monthly(cache, "6_Portfolios_ME_OP_2x3_CSV.zip")
    dev = french_monthly(cache, "Developed_3_Factors_CSV.zip", value_weighted=False)
    usd = pd.DataFrame({
        "us_mkt": ff["Mkt-RF"] + ff["RF"], "tech": ind["BusEq"], "health": ind["Hlth"],
        "qm": 0.5 * mom["BIG HiPRIOR"] + 0.5 * op["BIG HiOP"], "dev_mkt": dev["Mkt-RF"] + dev["RF"],
    })
    g = pd.read_csv(io.StringIO(_get(cache, "gold.csv", GOLD).decode("utf-8")))
    g.index = pd.PeriodIndex(g["Date"], freq="M").to_timestamp(how="end").normalize()
    gold_level = pd.to_numeric(g["Price"], errors="coerce").dropna()
    # Splice in month-end futures (from Aug 2000) and the GLD ETF (from Nov 2004): investable month-end prices
    for ticker in ("GC=F", "GLD"):
        path = cache / f"yf_{ticker.replace('=', '_')}.csv"
        if path.exists():
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            try:
                import yfinance as yf
                df = yf.download(ticker, start="2000-01-01", interval="1d", auto_adjust=True, progress=False, threads=False)
                c = df["Close"]
                c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
                c.index = pd.DatetimeIndex(c.index).tz_localize(None)
                s = c.dropna().groupby(c.dropna().index.to_period("M")).last()
                s.index = s.index.to_timestamp(how="end").normalize()
                s.to_frame("Close").to_csv(path)
            except Exception as e:  # keep the datahub series if Yahoo is unavailable
                print(f"{ticker} unavailable ({e}); gold stays on the datahub monthly series.")
                continue
        start = s.index.min()
        base = gold_level.loc[:start]
        if not base.empty and start in s.index:
            gold_level = pd.concat([gold_level.loc[:start - pd.offsets.MonthEnd(1)], s * (base.iloc[-1] / s.loc[start])])
    usd["gold"] = gold_level.pct_change()
    # PHLX Semiconductor index (price only; semis yield ~1%) for the semiconductor sleeve of the crash replays
    sox_path = cache / "yf__SOX.csv"
    try:
        if sox_path.exists():
            sox = pd.read_csv(sox_path, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            import yfinance as yf
            df = yf.download("^SOX", start="1994-01-01", interval="1d", auto_adjust=True, progress=False, threads=False)
            c = df["Close"]
            c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            c.index = pd.DatetimeIndex(c.index).tz_localize(None)
            sox = c.dropna().groupby(c.dropna().index.to_period("M")).last()
            sox.index = sox.index.to_timestamp(how="end").normalize()
            sox.to_frame("Close").to_csv(sox_path)
        usd["semis"] = sox.pct_change()
    except Exception as e:
        print(f"^SOX unavailable ({e}); semis column omitted.")
    boe = pd.read_csv(io.StringIO(_get(cache, "boe.csv", BOE).decode("latin1")))
    boe.index = pd.to_datetime(boe["DATE"], format="%d %b %Y").dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    fx = pd.to_numeric(boe["XUMLUSS"], errors="coerce").pct_change()
    usd = usd.loc["1975-02-28":]
    gbp = (1 + usd).div(1 + fx.reindex(usd.index), axis=0) - 1
    gbp["cash"] = pd.to_numeric(boe["IUMABEDR"], errors="coerce").reindex(gbp.index) / 1200.0
    return gbp


@dataclass
class Mix:
    name: str
    core: float
    sat: float
    defensive: float
    sat_kind: str = "tech"
    trend_to: Optional[str] = None  # None | 'cash' | 'core'


def run(m: Mix, s: pd.DataFrame, core_col: str) -> pd.Series:
    a = pd.DataFrame({"core": s[core_col] - DRAG["core"] / 12, "sat": s[m.sat_kind] - DRAG[m.sat_kind] / 12,
                      "def": s["gold"] - DRAG["gold"] / 12, "cash": s["cash"] - DRAG["cash"] / 12}).dropna()
    level = (1 + a["sat"]).cumprod()
    on = (level > level.rolling(10).mean()).astype(float)

    def target(t) -> Dict[str, float]:
        w = {"core": m.core, "sat": m.sat, "def": m.defensive, "cash": 0.0}
        if m.trend_to and on.get(t, 1.0) < 0.5:
            w["cash" if m.trend_to == "cash" else "core"] += w["sat"]
            w["sat"] = 0.0
        return w

    dates = a.index[12:]
    w, prev, out = target(a.index[11]), on.get(a.index[11], 1.0), []
    for t in dates:
        r = a.loc[t]
        gross = sum(w[k] * r[k] for k in w)
        tot = sum(w[k] * (1 + r[k]) for k in w)
        w = {k: w[k] * (1 + r[k]) / tot for k in w}
        tgt, regime, cost = target(t), on.get(t, 1.0), 0.0
        if (m.trend_to and regime != prev) or any(abs(w[k] - tgt[k]) > 0.05 for k in w):
            cost = sum(abs(w[k] - tgt[k]) for k in w) * TRADE_COST
            w = dict(tgt)
        prev = regime
        out.append(gross - cost)
    return pd.Series(out, index=dates, name=m.name)


def stats(r: pd.Series) -> Dict[str, float]:
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    dd = (eq / eq.cummax() - 1).min()
    crises = {}
    for k, (a, b) in CRISES.items():
        seg = r[(r.index > a) & (r.index <= b)]
        crises[k] = (float(np.prod(1 + seg)) - 1) * 100 if len(seg) else float("nan")
    return {"cagr": cagr * 100, "max_dd": dd * 100, **crises}


def sector_crash_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for k, (a, b) in CRISES.items():
        seg = p[(p.index > a) & (p.index <= b)]
        rows[k] = {c: round((float(np.prod(1 + seg[c].dropna())) - 1) * 100, 1)
                   for c in ("dev_mkt", "us_mkt", "tech", "semis", "health", "gold", "cash") if c in seg}
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    out = ["| | " + " | ".join(cols) + " |", "|:---|" + "---:|" * len(cols)]
    for idx, row in df.iterrows():
        out.append(f"| {idx} | " + " | ".join(f"{v:.1f}" if isinstance(v, (int, float)) and v == v else str(v) for v in row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=".crash_study_cache")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)
    p = build_panel(cache)
    lines = ["# Household crash study (survivorship-free, GBP, after costs)", "",
             "## Sector returns in each crisis window (GBP, %)", "", to_markdown(sector_crash_table(p)), ""]
    for label, core_col, start in (("Global core, 1991-2026", "dev_mkt", "1990-07-31"), ("US core, 1976-2026", "us_mkt", "1975-02-28")):
        s = p.loc[start:]
        mixes = [Mix("100% equity core", 1.0, 0.0, 0.0)]
        for core, sat, de, tag in ((0.75, 0.20, 0.05, "75/20/5"), (0.50, 0.45, 0.05, "50/45/5"), (0.28, 0.70, 0.02, "28/70/2")):
            mixes += [Mix(f"{tag} tech static", core, sat, de), Mix(f"{tag} tech trend->cash", core, sat, de, trend_to="cash"),
                      Mix(f"{tag} quality+momentum static", core, sat, de, sat_kind="qm")]
        res = {m.name: stats(run(m, s, core_col)) for m in mixes}
        lines += [f"## {label}", "", to_markdown(pd.DataFrame(res).T), ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
