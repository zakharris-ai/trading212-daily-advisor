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
1. Create an isolated Python virtual environment (`.venv`) and install dependencies (`yfinance`, `pandas`, `requests`).
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

---

## ☁️ Optional: Free Automated 9:00 AM Daily Email (GitHub Actions)

Want the advisor to email you your 2-minute Buy/Sell summary every morning at **09:00 AM UK time** even when your laptop is off?

1. **Fork or push this repository** to your own **private** GitHub repository.
2. In your GitHub repo, go to **Settings** ➔ **Secrets and variables** ➔ **Actions** ➔ **New repository secret**, and add:
   - `ACCOUNT_NAME`: Your name (e.g., `Alex`)
   - `T212_KEY_ID`: Your Trading 212 API Key ID
   - `T212_SECRET`: Your Trading 212 API Secret Key
   - `MIN_CASH_RESERVE_GBP`: Your protected cash floor in £ (e.g., `1500`)
   - `SMTP_USER`: Your Gmail address (e.g., `you@gmail.com`)
   - `SMTP_PASS`: A 16-character [Google App Password](https://myaccount.google.com/apppasswords)
   - `ALERT_EMAILS`: Where to send the daily report (e.g., `you@gmail.com`)
3. Go to the **Actions** tab in your GitHub repo, select **Daily Trading Intelligence Digest**, and click **Run workflow** to test it immediately!

---

## 🧠 How the Recommendation Engine Works

1. **30-Second Daily TL;DR**:
   Every run starts with a clear checklist:
   - 🔴 **Sells / Profit Trims to Execute**: Identifies holdings that have broken below their 3x ATR trailing stop, suffered fundamental deterioration, or become dangerously overextended (`>18%` above 50-day EMA with RSI `>75` or single-stock weight `>15%`).
   - 🟢 **High-Conviction Buys**: Ranks your existing winners and a curated 5-year high-Sortino watchlist by institutional quality, recommending exact £ GBP allocation sizes only when you have surplus cash above your reserve floor.
   - 🟡 **Hold & Compound**: Tells you which positions are in healthy Stage 2 uptrends so you can sit back and let them compound.
2. **Forensic Balance-Sheet Audit (Piotroski F-Score 0–9 & Altman Z-Score)**:
   Verifies GAAP Net Income, Operating Cash Flow (`OCF > Net Income`), Free Cash Flow, Gross Margin expansion, and solvency (`Altman Z > 3.0`) before approving any buy.
3. **Asymmetric Risk/Reward Gatekeeper ($\ge 2.0:1$)**:
   Calculates upside to consensus target vs. downside to trailing stop, blocking poor risk/reward entries.
4. **Earnings Catalyst Protection**:
   Automatically blocks new speculative entries within **5 trading days** of an upcoming earnings report.
5. **Protected Cash Floor (`MIN_CASH_RESERVE_GBP`)**:
   Ring-fences your emergency cash reserve (default **£1,500**) so the advisor never recommends BUY orders that dip into your protected cash buffer.
