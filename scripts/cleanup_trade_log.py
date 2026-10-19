#!/usr/bin/env python3
"""One-off data hygiene for the trade log and the position date maps (review finding B13, 18 Oct 2026).

What it does (all inside data/):

1. trade_log.json: removes duplicate events, i.e. records with the same account, ticker, date, action and
   shares (the traded quantity and the resulting position size), keeping the FIRST occurrence. Tickers are
   compared, and stored, in canonical Yahoo form via t212_client.resolve_canonical_ticker, so a stale
   broker-format record (e.g. 'ASMLa_EQ') and its canonical twin ('ASML.AS') count as the same event.
2. position_entries.json / position_trims.json: migrates stale ticker keys to canonical tickers
   (e.g. 'ASMLa_EQ' -> 'ASML.AS'). When a migrated key collides with an existing key the LATEST date wins,
   the same rule portfolio_storage applies when it reads these files.
3. Prints a before/after summary.

Safe to re-run: a second run reports zero changes. A file is only rewritten when its content changes, using
the same JSON formatting as portfolio_storage. Use --dry-run to preview without writing anything. The
originals remain recoverable from git (e.g. `git show HEAD:data/trade_log.json`).

Usage (from the repo root):
    ./.venv/bin/python3 scripts/cleanup_trade_log.py [--dry-run] [--data-dir PATH]
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from t212_client import resolve_canonical_ticker  # noqa: E402

# Share quantities are compared at 1e-8 so float noise (0.46392764000000007 vs 0.46392764) never splits a duplicate.
SHARE_DECIMALS = 8


def _canon(ticker: Any) -> str:
    """Canonical Yahoo-style ticker ('' stays '')."""
    raw = str(ticker or "").strip()
    return resolve_canonical_ticker(raw) if raw else raw


def _shares(val: Any) -> Optional[float]:
    """Share quantity rounded for comparison; None when missing or not a finite number."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, SHARE_DECIMALS)


def _event_key(rec: Dict[str, Any], canonical: bool) -> Tuple[Any, ...]:
    """(account, ticker, date, action, traded shares, resulting position shares) identity of a trade event."""
    ticker = _canon(rec.get("ticker")) if canonical else str(rec.get("ticker") or "")
    traded = rec.get("shares_added", rec.get("shares_sold"))
    position = rec.get("new_total_shares", rec.get("remaining_shares"))
    return (rec.get("account"), ticker, rec.get("date"), rec.get("action"), _shares(traded), _shares(position))


def dedupe_trade_log(log: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Removes duplicate trade events (first occurrence kept) and canonicalises the kept records' tickers.

    Returns:
        (cleaned log, stats) where stats holds exact_duplicates (identical raw ticker), canonical_duplicates
        (only identical once tickers are canonicalised), tickers_canonicalised and ticker_renames.
    """
    seen_exact = set()
    seen_canonical = set()
    cleaned: List[Any] = []
    stats: Dict[str, Any] = {
        "exact_duplicates": 0,
        "canonical_duplicates": 0,
        "tickers_canonicalised": 0,
        "ticker_renames": Counter(),
    }
    for rec in log:
        if not isinstance(rec, dict):
            cleaned.append(rec)  # unrecognised entry: kept untouched
            continue
        exact_key = _event_key(rec, canonical=False)
        canonical_key = _event_key(rec, canonical=True)
        if exact_key in seen_exact:
            stats["exact_duplicates"] += 1
            continue
        seen_exact.add(exact_key)
        if canonical_key in seen_canonical:
            stats["canonical_duplicates"] += 1
            continue
        seen_canonical.add(canonical_key)

        canonical_ticker = _canon(rec.get("ticker"))
        if canonical_ticker and canonical_ticker != rec.get("ticker"):
            stats["ticker_renames"][(rec.get("ticker"), canonical_ticker)] += 1
            stats["tickers_canonicalised"] += 1
            rec = dict(rec)
            rec["ticker"] = canonical_ticker
        cleaned.append(rec)
    return cleaned, stats


def _later(a: Any, b: Any) -> Any:
    """The later of two 'YYYY-MM-DD' dates (a non-empty value beats an empty one)."""
    if not a:
        return b
    if not b:
        return a
    return a if str(a) >= str(b) else b


def migrate_date_map(data: Any) -> Tuple[Any, List[str], int]:
    """Canonicalises {account: {ticker: date}} keys; keys that map to the same ticker keep the latest date.

    Returns:
        (migrated map, one human-readable line per changed ticker group, number of stale keys migrated)
    """
    if not isinstance(data, dict):
        return data, [], 0
    changes: List[str] = []
    stale_keys = 0
    migrated: Dict[str, Any] = {}
    for acc, acc_map in data.items():
        if not isinstance(acc_map, dict):
            migrated[acc] = acc_map
            continue
        groups: Dict[str, List[Tuple[str, Any]]] = {}  # canonical key -> [(original key, date)], file order
        for ticker, value in acc_map.items():
            groups.setdefault(_canon(ticker) or ticker, []).append((ticker, value))
        new_map: Dict[str, Any] = {}
        for key, members in groups.items():
            kept = members[0][1]
            for _, value in members[1:]:
                kept = _later(kept, value)
            new_map[key] = kept
            stale = sum(1 for ticker, _ in members if ticker != key)
            if stale or len(members) > 1:
                stale_keys += stale
                sources = ", ".join(f"{ticker!r} ({value})" for ticker, value in members)
                note = "; latest date kept" if len(members) > 1 else ""
                changes.append(f"{acc}: {sources} -> {key!r} = {kept}{note}")
        migrated[acc] = new_map
    return migrated, changes, stale_keys


def _load_json(path: Path, default: Any) -> Tuple[Any, bool]:
    """(parsed JSON, existed). Raises on unreadable/malformed files so nothing is ever overwritten blindly."""
    if not path.is_file():
        return default, False
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), True


def _write_json(path: Path, data: Any) -> None:
    """Atomic write with portfolio_storage's formatting (indent=2, ensure_ascii=False, no trailing newline)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _action_counts(log: List[Any]) -> Counter:
    return Counter(str(r.get("action")) for r in log if isinstance(r, dict))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="De-duplicate the trade log and migrate stale ticker keys.")
    parser.add_argument("--dry-run", action="store_true", help="print the summary without writing any file")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"), help="data directory (default: ./data)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    trade_log_path = data_dir / "trade_log.json"
    entries_path = data_dir / "position_entries.json"
    trims_path = data_dir / "position_trims.json"

    try:
        trade_log, has_log = _load_json(trade_log_path, [])
        entries, has_entries = _load_json(entries_path, {})
        trims, has_trims = _load_json(trims_path, {})
    except (OSError, ValueError) as e:
        print(f"ERROR: could not read the data files ({e}); nothing was changed.", file=sys.stderr)
        return 1
    if not isinstance(trade_log, list):
        print(f"ERROR: {trade_log_path} is not a JSON list; nothing was changed.", file=sys.stderr)
        return 1

    cleaned_log, stats = dedupe_trade_log(trade_log)
    new_entries, entry_changes, entry_stale = migrate_date_map(entries)
    new_trims, trim_changes, trim_stale = migrate_date_map(trims)

    to_write = []
    if has_log and cleaned_log != trade_log:
        to_write.append((trade_log_path, cleaned_log))
    if has_entries and new_entries != entries:
        to_write.append((entries_path, new_entries))
    if has_trims and new_trims != trims:
        to_write.append((trims_path, new_trims))

    removed = stats["exact_duplicates"] + stats["canonical_duplicates"]
    print(f"Trade log hygiene ({trade_log_path})")
    print(f"  records before .............................. {len(trade_log)}")
    print(f"  exact duplicates removed .................... {stats['exact_duplicates']}")
    print(f"  duplicates removed after canonical tickers .. {stats['canonical_duplicates']}")
    print(f"  records removed in total .................... {removed}")
    print(f"  records after ............................... {len(cleaned_log)}")
    print(f"  ticker fields canonicalised ................. {stats['tickers_canonicalised']}")
    for (old, new), count in sorted(stats["ticker_renames"].items()):
        print(f"      {old!r} -> {new!r} x{count}")
    before_actions, after_actions = _action_counts(trade_log), _action_counts(cleaned_log)
    print("  by action (before -> after):")
    for action in sorted(set(before_actions) | set(after_actions)):
        print(f"      {action}: {before_actions.get(action, 0)} -> {after_actions.get(action, 0)}")

    def _key_count(m: Any) -> int:
        return sum(len(v) for v in m.values() if isinstance(v, dict)) if isinstance(m, dict) else 0

    print("Position date maps")
    for label, old_map, new_map, changes, stale, existed in (
        (entries_path.name, entries, new_entries, entry_changes, entry_stale, has_entries),
        (trims_path.name, trims, new_trims, trim_changes, trim_stale, has_trims),
    ):
        if not existed:
            print(f"  {label}: file not found (skipped)")
        elif not changes:
            print(f"  {label}: no stale keys ({_key_count(old_map)} keys)")
        else:
            print(
                f"  {label}: {stale} stale key(s) migrated; keys {_key_count(old_map)} -> {_key_count(new_map)}"
            )
            for change in changes:
                print(f"      {change}")

    if args.dry_run:
        print("Dry run: no files written." + (f" Would write: {', '.join(p.name for p, _ in to_write)}" if to_write else ""))
        return 0
    for path, data in to_write:
        _write_json(path, data)
    print("Files written: " + (", ".join(p.name for p, _ in to_write) if to_write else "none (already clean)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
