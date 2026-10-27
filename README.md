# 📈 Trading 212 Daily Buy/Sell & Rebalancing Advisor

An institutional-grade quantitative portfolio engine that connects (read-only) to your **Trading 212** account, audits your live positions and cash balance, and gives you a **2-minute daily action plan** of **what to BUY, SELL, TRIM, or HOLD**.

> 🔒 **100% Non-Custodial & Read-Only**: This tool only *reads* your portfolio and generates recommendations. It never places trades automatically—you review the recommendations and execute orders directly inside your Trading 212 app.

---

## ⚡ 60-Second Quickstart (Any User)

### Step 1: Clone the Repo & Run the Interactive Setup Wizard
```bash
git clone https://github.com/zakharris-ai/trading212-daily-advisor.git
cd trading212-daily-advisor
./setup.sh
```
The `./setup.sh` wizard will:
1. Create an isolated Python virtual environment (`.venv`) and install the pinned dependencies (`yfinance`, `pandas`, `numpy`, `requests`).
2. Walk you through connecting your Trading 212 API key and setting your protected cash reserve floor.
3. Run an immediate verification check and generate your interactive web dashboard.

> 💡 **Want to preview it before creating an API key?**
> Just press **Enter** when `./setup.sh` asks for your API key, or run:
> ```bash
> ./run.sh --dry-run --mock
> ./dashboard.sh
> ```

---

## 🔑 How to Get Your Trading 212 API Key (Takes 30 Seconds)

1. Open the **Trading 212** app on your phone or web browser.
2. Go to **Settings** ⚙️ ➔ **API (Beta)**.
3. Tap **Generate API Key**.
4. Give it a name (e.g., `Daily Advisor`) and enable **Read** permissions for:
   - ✅ **Account Data** (Cash balance)
   - ✅ **Portfolio** (Open positions)
   - ✅ **Orders** (Pending orders)
   - ✅ **History** (Transactions)
5. Copy the **API Key ID** and **Secret Key** into `./setup.sh` (or your `.env` file).

---

## 🖥️ Daily Usage Commands

| What you want to do | Command |
| :--- | :--- |
| **See what to Buy / Sell today** (Terminal + updates Web Dashboard) | `./run.sh --dry-run` |
| **Open the Interactive Web Dashboard** in your browser | `./dashboard.sh` |
| **Serve the Web Dashboard** on `http://localhost:8080` | `./dashboard.sh --serve 8080` |
| **Test with sample portfolio data** (no API key needed) | `./run.sh --dry-run --mock` |
| **Run & send daily email digest** (requires Gmail SMTP in `.env`) | `./run.sh` |
| **Re-send today's digest** (bypasses the once-a-day dedupe) | `./run.sh --force` |
| **Run the offline test suite** | `./.venv/bin/python3 -m unittest discover -s tests -v` |

### `--dry-run` writes no state
A dry run prints the action plan and regenerates the local previews (`latest_digest.html` and `dashboard.html`), but it never:
- detects or records trades (`data/trade_log.json`, `position_entries.json`, `position_trims.json`, `position_closed.json`);
- saves a daily snapshot or updates `data/portfolio_history.json`;
- sends an email or writes the `data/last_email_sent.json` sent-marker;
- appends to the recommendation ledger.

`--mock` runs never call Trading 212 and never record trades, snapshots or ledger entries either.

---

## ☁️ Optional: Free Automated 9:00 AM Daily Email (GitHub Actions)

Want the advisor to email you your 2-minute Buy/Sell summary every morning at **09:00 AM UK time** even when your laptop is off?

1. **Fork or push this repository** to your own **private** GitHub repository. (In a fork, open the **Actions** tab once and enable workflows.)
2. In your GitHub repo, go to **Settings** ➔ **Secrets and variables** ➔ **Actions** ➔ **New repository secret**, and add:
   - `ACCOUNT_NAME`: Your name (e.g., `Alex`)
   - `T212_KEY_ID`: Your Trading 212 API Key ID
   - `T212_SECRET`: Your Trading 212 API Secret Key
   - `MIN_CASH_RESERVE_GBP`: Your protected cash floor in £ (e.g., `1500`)
   - `SMTP_USER`: Your Gmail address (e.g., `you@gmail.com`)
   - `SMTP_PASS`: A 16-character [Google App Password](https://myaccount.google.com/apppasswords)
   - `ALERT_EMAILS`: Where to send the daily report (e.g., `you@gmail.com`)
   - Optional: `T212_ENV` (`demo` for a practice account), `EXCLUDED_TICKERS`, `SMTP_HOST`, `SMTP_PORT`, a second account (`SECOND_ACCOUNT_NAME`, `T212_KEY_ID_2`, `T212_SECRET_2`, `MIN_CASH_RESERVE_2`) and any of the [optional settings](#-optional-settings) below.
3. Go to the **Actions** tab in your GitHub repo, select **Daily Trading Intelligence Digest**, and click **Run workflow** to test it immediately!

GitHub's own `schedule:` trigger is best-effort and can start runs hours late. For a reliable 09:00 email, add the external trigger described next.

---

## ⏰ Reliable 09:00 Delivery (External Scheduler)

The workflow's cron ticks stay in place as a fallback, but GitHub's `schedule:` trigger is best-effort: in practice scheduled runs are often several hours late, and on some days they don't happen at all. An external scheduler such as [cron-job.org](https://cron-job.org) (free) can start the workflow through GitHub's `workflow_dispatch` API at **09:00 Europe/London** every day. Whichever trigger runs first sends the email; every later run reads the sent-marker (`data/last_email_sent.json` on `origin/main`) and skips, so you still get **exactly one email per day**.

If a run fails, you get **one alert email that day** saying which stage failed, with the last log lines and a link to the run (see [How the Daily GitHub Actions Run Works](#️-how-the-daily-github-actions-run-works)).

In the steps below, replace `<your-github-user>/<your-repo>` with your own repository (e.g. your fork).

### Step 1: Create a fine-grained token (one-off)
1. On GitHub, go to **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. **Token name:** e.g. `trading212-digest-dispatch`. **Expiration:** up to 1 year (set a reminder to renew it; when it expires only the GitHub cron fallback runs).
3. **Repository access:** *Only select repositories* → `<your-github-user>/<your-repo>` (that repository only).
4. **Permissions → Repository permissions → Actions: Read and write.** GitHub adds *Metadata: Read-only* automatically; nothing else is needed.
5. Generate the token and copy it (it starts with `github_pat_`).

### Step 2: Test the token from a terminal
```bash
curl -i -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer <your-fine-grained-token>" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -H "Content-Type: application/json" \
  https://api.github.com/repos/<your-github-user>/<your-repo>/actions/workflows/daily_trading_digest.yml/dispatches \
  -d '{"ref":"main"}'
```
- **Expected response:** `HTTP/2 204` with an empty body, and a new `workflow_dispatch` run under **Actions**.
- **The test is safe:** if today's digest has already gone out, that run stops at the gatekeeper ("already been sent") without sending a second email.

### Step 3: Create the cron-job.org job
1. Sign in at [cron-job.org](https://cron-job.org) and open **Dashboard → Cronjobs → Create cronjob**.
2. **Common** tab:
   - **URL:** `https://api.github.com/repos/<your-github-user>/<your-repo>/actions/workflows/daily_trading_digest.yml/dispatches`
   - **Execution schedule:** every day at **09:00**.
3. **Advanced** tab:
   - **Time zone:** `Europe/London` (follows BST/GMT automatically; check your account's default time zone isn't overriding it).
   - **Request method:** `POST`
   - **Headers:** add exactly these:

     | Key | Value |
     |---|---|
     | `Accept` | `application/vnd.github+json` |
     | `Authorization` | `Bearer <your-fine-grained-token>` |
     | `X-GitHub-Api-Version` | `2022-11-28` |
     | `Content-Type` | `application/json` |

   - **Request body:** `{"ref":"main"}`
4. Turn on notifications for failed executions and **Save**. The next morning, **Actions** should show a `workflow_dispatch` run at about 09:00 that sends the email, and later `schedule` runs that log "already been sent" and skip.

**Re-sending a digest:** run the workflow from **Actions → Daily Trading Intelligence Digest → Run workflow** with **force** ticked, or dispatch with the body `{"ref":"main","inputs":{"force":"true"}}`. Without `force`, every run (scheduled or dispatched) respects the once-a-day dedupe.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Bad credentials` | Token mistyped, revoked or expired. | Regenerate the token and update the `Authorization` header. |
| `403 Resource not accessible by personal access token` | Token lacks **Actions: Read and write**, or isn't scoped to this repository. | Edit the token's repository access and permissions. |
| `404 Not Found` | Wrong owner/repo/workflow file name, or the token can't see the (private) repository. | Check the URL and the token's repository access. |
| `422 Unprocessable Entity` | `ref` isn't a branch containing the workflow, or `workflow_dispatch` is missing on that branch. | Use `{"ref":"main"}` and make sure the workflow file is on `main`. |
| `204` but no email | The run skipped (already sent today), failed closed (see below), or the SMTP login failed. | Check for the failure alert email, or open the run's logs under **Actions**; re-send with `force` if needed. |
| Scheduled runs only, hours late | External job not running or wrong time zone. | Check the cron-job.org execution history and its `Europe/London` time zone. |

---

## ⚙️ How the Daily GitHub Actions Run Works

The workflow lives in `.github/workflows/daily_trading_digest.yml`:

1. **Triggers:** cron ticks at :05, :20, :35 and :50 past 08:00 and 09:00 UTC (covering the BST and GMT morning windows) plus a 10:05 UTC catch-all, and manual/external `workflow_dispatch` with an optional boolean `force` input.
2. **Concurrency:** all runs share the `daily-digest` group (`cancel-in-progress: false`), so two runs never send or push at the same time; the job times out after 20 minutes.
3. **Gatekeeper:** if the repository has no Trading 212 key at all (`T212_KEY_ID` and `T212_KEY_ID_2` both unset, e.g. this template or a fork you haven't set up yet), the run is skipped cleanly with a notice, so it stays green and GitHub sends no "Run failed" emails. Otherwise it runs `git fetch origin main` and reads `data/last_email_sent.json` from `origin/main`. Scheduled ticks proceed only when the Europe/London hour is 9 or later and nothing has been sent today; dispatches skip the hour check but still respect the dedupe unless `force` is true.
4. **Offline unit tests** run before the digest; a failing test fails the job before any email is sent.
5. **`main.py`** runs (with `--force` only when the dispatch input `force` is true). Test and digest output is also written to `logs/` (gitignored, never committed) so a failure alert can quote it.
6. **Commit step:** `git add data/ dashboard.html`, then up to 3 attempts of `git pull --rebase origin main` + `git push`.
7. **Failure alert (only when a step fails):** `scripts/failure_alert.py` emails `ALERT_EMAILS` with the stage that failed, whether today's digest had already gone out, the last 40 log lines of the failed step and a link to the run. It uses only the Python standard library (so it works even if the dependency install failed) and sends at most one alert per London day, remembered with the Actions cache key `failure-alert-YYYY-MM-DD`. If SMTP itself is broken the alert can't be emailed either; GitHub's own "Run failed" notification is the backstop.

## 🛑 Fail-Closed Behaviour

When data is missing the engine stops rather than guessing. `main.py` exits with code 1 **before** writing any snapshot, trade log, email, sent-marker or ledger entry when:
- no Trading 212 credentials are configured, or no account returned data;
- Trading 212 positions are unavailable (API error, timeout or malformed response), after the client's own retries with backoff;
- an account reports invested capital but the positions endpoint returns none;
- no live GBP/USD rate is available. The rate comes from Yahoo first, then the ECB reference rate (Frankfurter), then ExchangeRate-API's open endpoint; rates older than 5 days or outside 0.5–3.0 are rejected and a static rate is never assumed. Other currencies (EUR, CHF, …) use the same fallback chain.

It also exits with code 1, without writing the sent-marker, if the email fails to send, so the next scheduled tick retries. Inside the engine, a holding or watchlist idea with failed research, missing weekly data or an unconvertible price is never turned into a BUY; if a held position can't be valued in GBP, that account's plan is marked **DATA DEGRADED** and emits no orders. Missing metrics are shown as `n/a` instead of optimistic defaults.

---

## 🧭 Optional Settings

All optional settings are off by default. Set them in `.env` for local runs, or as repository secrets of the same name for GitHub Actions (the workflow already maps them). Account names are the ones you configured (`ACCOUNT_NAME`, default `Main`; `SECOND_ACCOUNT_NAME`, default `Secondary`).

| Setting (env var) | Default | What it does |
|---|---|---|
| `EARMARKED_TICKERS` | none | Money with a fixed purpose (e.g. `CSH2.L` holding savings you need soon). Earmarked holdings are excluded from the investable book (weights, caps, crash shield, projections) and never traded. Comma-separated; `none` clears. |
| `OUTSIDE_HOLDINGS_JSON` | none | Holdings outside Trading 212 (e.g. shares you hold at another broker, or company shares from a workplace share scheme). They are never traded; vested value counts towards single-name and cluster caps, unvested value is shown for information only. Shape: `{"Main": [{"ticker": "MSFT", "shares_vested": 10, "shares_unvested": 5, "currency": "USD"}]}` (optional keys: `macro_cluster`, `value_hint_usd_vested`, `value_hint_usd_unvested`, `as_of`). Invalid JSON keeps the default and logs a warning. Add the ticker to `EXCLUDED_TICKERS` too if you never want the engine to suggest buying more. |
| `RESERVE_TOPUPS_JSON` | none | Temporarily raises an account's protected cash floor, e.g. while money you plan to withdraw sits as free cash: `{"Main": {"amount_gbp": 2000, "until": "2026-12-31"}}`. Active while the Europe/London date is on or before `until`; lapses automatically afterwards. An unreadable `until` keeps the top-up active, so protected cash is never released by mistake. |
| `ACCOUNT_WRAPPER_<NAME>` (or `ACCOUNT_WRAPPER` / `ACCOUNT_WRAPPER_2` for the first / second account) | not set | Tax wrapper per account (`ISA`, `GIA`, ...). Informational only; when every account is an ISA the dashboard notes that trades carry no CGT. |
| `MAX_LEVERAGED_ETF_PCT` | `0.08` | **Leverage policy:** maximum weight in 3x daily-leveraged ETFs (QQQ3, 3LUS, 3SEM, ...). Holdings above the cap are trimmed back to it and 3x ETFs are never bought. `0` means "exit 3x ETFs": any you hold is flagged as a full SELL. |
| `EXCLUDED_TICKERS` | none | Tickers the engine must never recommend buying (e.g. `TSLA,COIN`). |

Fixed limits in `config.py` (`REBALANCE_CONFIG`):
- `max_position_pct`: 15% per holding (evaluated across combined household wealth when household context is supplied). Not applied to broad index ETFs unless `cap_applies_to_broad_etfs` is `True`.
- `max_cluster_pct`: 24% per correlated macro cluster.
- `max_new_ideas_to_buy`: 2.
- `target_crash_shield_pct`: 5% (env `TARGET_CRASH_SHIELD_PCT`). Crash protection counts **only real defensive assets**: cash, money-market funds, physical gold and short-dated gilts. Low-beta stocks and equity index funds do not count.

`MODEL_VERSION` is stamped on every snapshot and ledger record; bump it whenever you change the scoring rules.

**Household allocation (`ALLOCATION_CONFIG`).** All your accounts plus any vested outside holdings are treated as one portfolio, split into three buckets:
- **Core**: All-World / S&P 500 index funds.
- **Satellites**: single stocks, Nasdaq-100, sector and thematic funds.
- **Defensive**: cash, money-market funds, gold and short gilts.

The default target is **75 / 20 / 5** (Gold-only new defensive buys by default via `DEFENSIVE_GOLD_SHARE=1.0`). New money goes to whichever bucket is furthest below target; buckets more than ±5pp below target go first. Nothing profitable is ever sold to rebalance.

How new money is invested:
- **Core money** buys `VWRP.L`.
- **Defensive money** buys `SGLN.L` (accumulating physical gold; or a split with `IGLS.L` short gilts if `DEFENSIVE_GOLD_SHARE` is set below `1.0`).
- **Satellite money** goes through the stock-selection engine, subject to:
  - a 5% household cap per stock for new buys;
  - a 30% cap on US big tech (AAPL, MSFT, GOOGL/GOOG, AMZN, META, NVDA, AVGO, TSLA) as a share of household wealth. This counts fund look-through, using approximate weights reviewed yearly, plus outside holdings.
- **GBP-quoted lines** are used for new money, which avoids the 0.15% FX fee.

**Buy cadence:** buys happen on the **first London business day of each month** (bank holidays respected), or on any day when deployable cash is at least £2,000. Sells and risk alerts stay daily.

Optional env overrides (all parsed safely):

| Variable | Default |
|---|---|
| `TARGET_CORE_PCT` | `0.75` |
| `TARGET_SATELLITE_PCT` | `0.20` |
| `TARGET_DEFENSIVE_PCT` | `0.05` |
| `DEFENSIVE_GOLD_SHARE` | `1.0` |
| `ALLOCATION_BAND_PP` | `5` |
| `SATELLITE_MAX_SINGLE_PCT` | `0.05` |
| `BIG_TECH_CAP_PCT` | `0.30` |
| `BUY_CADENCE` | `monthly` (or `daily`) |
| `OFF_CYCLE_MIN_DEPLOY_GBP` | `2000` |
| `CORE_BUY_TICKER` | `VWRP.L` |
| `GOLD_BUY_TICKER` | `SGLN.L` |
| `SHORT_GILT_BUY_TICKER` | `IGLS.L` |

Targets that don't sum to 1 are normalised. The research model's forward-CAGR estimate is shown for information only and never affects scores or sizes.

All numeric environment values go through `_safe_float` / `_safe_int`, so blank, `true`, `false` or malformed values fall back to the defaults instead of crashing. If you add your own env mappings to the workflow, never use `${{ a || b }}` fallbacks (see [`GEMINI.md`](./GEMINI.md)).

---

## 📒 Recommendation Ledger & 📈 Benchmark Tracking

**Recommendation ledger:** after every successful live send, `main.py` appends one record per recommended order (sells/trims not already pending in Trading 212, and all buys) to `data/recommendation_ledger.json` with `date`, `model_version`, `account`, `action`, `ticker`, `bucket`, `amount_gbp`, `price`, `currency` and `reason`. Records are deduped on `date|model_version|account|action|ticker`, and an unreadable ledger is never overwritten. It lets you check later whether a model version's recommendations actually added value.

**Benchmark & time-weighted return (TWR):** each daily snapshot (`data/snapshots/YYYY-MM-DD.json`, under `"performance"`) and each `data/portfolio_history.json` record carries:
- `external_flows_gbp`: net deposits minus withdrawals since the previous snapshot (from Trading 212 `DEPOSIT` / `WITHDRAW` transactions, deduped by reference);
- `twr_index_total` and `twr_index_<account>` (e.g. `twr_index_main`): time-weighted return indices based at 100 on the first tracked day, so new money is never counted as performance;
- `benchmark_vwrp_index`: the VWRP.L close rebased to 100 on the same day ("everything in a global index fund"), or `null` when unavailable.

The digest's day-over-day change also excludes deposits and withdrawals. The email, dashboard and console show a "since tracking began: you vs VWRP" line built from these indices.

`scripts/cleanup_trade_log.py` is an idempotent hygiene script for older data folders: it removes duplicate trade events and migrates stale ticker keys (e.g. `ASMLa_EQ` → `ASML.AS`). Preview with `./.venv/bin/python3 scripts/cleanup_trade_log.py --dry-run`.

---

## 🧪 Running the Tests

```bash
./.venv/bin/python3 -m unittest discover -s tests -v
```
- **Fully offline:** stdlib `unittest` with all network access mocked (no Trading 212, Yahoo Finance or SMTP calls); storage tests use a temporary data directory, never `data/`.
- **Independent of your `.env`:** behaviour is exercised through injected inputs, so your own settings never change the result.
- **Runs in CI** before every digest; a failing test blocks the email. Python 3.9+.
- **Workflow rules are tested too:** `tests/test_workflow.py` checks the workflow file's invariants (no `||` inside `${{ }}`, offset retry ticks, the hour ≥ 9 gatekeeper with dedupe, no gitignored files in `git add`, single-line `python -c`, `pipefail` with `tee`, and the failure-alert wiring).

## 🔬 Evidence: Strategy Backtest

```bash
./.venv/bin/python3 scripts/backtest.py --out backtest_results.md [--json backtest_results.json]
```

The backtest is monthly, in GBP, after costs, with no look-ahead. It runs from 2008 to the last complete month and compares:
- 100% global equity (≈ VWRP).
- Core + 15% defensive, rebalanced within ±5pp bands.
- Core + 20% momentum sector satellites + defensive, against a no-momentum control.
- A 10-month SMA trend filter.
- The engine's own weekly trend rule.

It uses textbook parameters only (nothing fitted), needs network access (Yahoo Finance) and writes files only when asked. Its unit tests are offline.

Results for Apr 2008 – Sep 2026:
- Momentum sector selection added nothing over the control (t-stat −0.2).
- A 15% defensive sleeve cut the worst fall from −33% to −23% for about 0.3 pp a year.
- The weekly trend rule cost about 2.8 pp a year at index level, which is why core and defensive buys ignore trend signals.

Past results on one historical path are not a forecast.

---

## 🧠 How the Recommendation Engine Works

1. **30-Second Daily TL;DR**:
   Every run starts with a clear checklist:
   - 🔴 **Sells / Cap Trims to Execute**: Identifies holdings with severe balance-sheet distress (`Altman Z < 1.81` with `Piotroski <= 2`) or leverage-policy breaches, and trims positions back to the 15% household single-position and leverage caps (oversold technical breakdowns are surfaced as HOLD / review alerts rather than auto-sold; cluster-cap breaches block new buys in that cluster).
   - 🟢 **High-Conviction Buys**: Ranks your existing winners and a curated 5-year high-Sortino watchlist by institutional quality, recommending exact £ GBP allocation sizes only when you have surplus cash above your reserve floor.
   - 🟡 **Hold & Compound**: Tells you which positions are in healthy Stage 2 uptrends so you can sit back and let them compound.
2. **Forensic Balance-Sheet Audit (Quality Checklist 0–9 & Altman Z-Score)**:
   Verifies GAAP Net Income, Operating Cash Flow (`OCF > Net Income`), Free Cash Flow, Gross Margin expansion, and solvency (`Altman Z > 3.0`) before approving any buy.
3. **Asymmetric Risk/Reward Gatekeeper ($\ge 2.0:1$)**:
   Calculates upside to the analyst consensus target vs. downside to the reference stop, blocking poor risk/reward entries (funds, gold and cash lines get no R:R).
4. **Earnings Catalyst Protection**:
   Automatically blocks new speculative entries within **5 trading days** of an upcoming earnings report.
5. **Protected Cash Floor (`MIN_CASH_RESERVE_GBP`)**:
   Ring-fences your emergency cash reserve (default **£1,500**, plus any active top-up) so the advisor never recommends BUY orders that dip into your protected cash buffer. The digest shows the floor as **FUNDED** only when free cash actually covers it; otherwise it shows **UNFUNDED** with the shortfall.
