# Trading 212 Daily Advisor: Autonomous Operations & Reliability Rules

## 1. Autonomous Scheduling & Laptop-Off Guarantee
- **Cloud-First Execution**: The daily pipeline runs autonomously in GitHub Actions (`.github/workflows/daily_trading_digest.yml`) so it executes reliably when your laptop is off, or can be run locally on demand via `./run.sh`.
- **Target Dispatch**: Exactly ONE morning email digest delivered daily at 09:00 Europe/London time.

## 2. GitHub Actions Workflow Invariants
- **No `||` Fallbacks in Workflow Env**: Never write `${{ secrets.VAR || 'fallback' }}` in workflow files. In GitHub Actions expressions, `||` is a boolean operator that yields string `"true"`, crashing downstream `int()` and `float()` parsers in Python.
- **Defensive Python Parsing**: All environment variables representing numbers or booleans must be parsed via `_safe_int(os.getenv(...), default)` and `_safe_float(os.getenv(...), default)` in `config.py`.
- **Offset Cron Schedules & Multi-Tick Morning Retries**: Workflows are scheduled off peak minute `:00` across 15-minute retry intervals (`5 8`, `20 8`, `35 8`, `50 8` for BST; `5 9`, `20 9`, `35 9`, `50 9` for GMT). The deduplication gatekeeper (`data/last_email_sent.json`) guarantees exactly ONE dispatch per calendar day.
- **Resilient Gatekeeper & Deduplication**:
  - The gatekeeper allows execution if `LONDON_HOUR >= 9` while checking `data/last_email_sent.json` to guarantee exactly 1 dispatch per calendar day.
- **Single-Line Python in YAML Steps**: Python commands inside workflow `run: |` blocks avoid multiline strings with column-0 indentation.
- **Never `git add` Gitignored Files**: Never include files from `.gitignore` (e.g., `latest_digest.html`) in workflow `git add` commands.

## 3. Financial Research & Broker Ticker Normalization
- **Exchange Suffix Mapping**: When ingesting Trading 212 broker tickers, map European suffixes cleanly:
  - London: `_?[lL]_EQ` -> `.L`
  - Deutsche Boerse Xetra: `_?[dD]_EQ` -> `.DE`
  - Euronext Amsterdam: `_?(as|nl|a|AS|NL|A)_EQ` -> `.AS` (e.g. `ASMLa_EQ` -> `ASML.AS`)
  - Euronext Paris: `_?(fp|pa|FP|PA)_EQ` -> `.PA`
  - Swiss Exchange: `_?[sS]_EQ` -> `.SW`
- **Fallback Stripping**: Unrecognized `_?[a-zA-Z0-9]+_EQ` suffixes are stripped to prevent 404 lookup failures on financial data providers.
- **Currency Conversion**: Non-GBP assets (EUR, USD, GBp) pass through `to_gbp()` with live currency conversion rates.

## 4. Ground-Truth Broker Cash Accounting & Buy Invariants
- **Live Cash Balance is Authoritative**: Never treat `free_cash == 0.0` as an API failure or trigger for fallback values. When an investor is 100% invested, £0.00 uninvested cash is the accurate ground truth.
- **Strict Fallback Isolation**: `manual_cash` overrides are ONLY engaged when the broker API endpoint completely fails to return a response (`raw_cash is None` or missing `"free"` key). In all cases, default fallback cash is `0.0`.
- **Protected Cash Reserve Guarantee**: Never recommend BUY orders unless uninvested liquid cash exceeds the configured `MIN_CASH_RESERVE_GBP` PLUS the minimum deployment threshold. If cash is below the ring-fenced reserve or £0.00, the rebalancing engine outputs `ALL SET — SIT BACK & RELAX`.
