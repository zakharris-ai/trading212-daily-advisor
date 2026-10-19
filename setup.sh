#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=================================================================="
echo "🚀 Trading 212 Daily Buy/Sell Advisor — Interactive Setup Wizard"
echo "=================================================================="

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $PYTHON_VERSION"

# 2. Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "✅ Virtual environment activated."

# 3. Install Dependencies
echo "📥 Installing dependencies from requirements.txt..."
export PIP_INDEX_URL="https://pypi.org/simple"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed successfully."

# 4. Interactive Configuration Wizard
CONFIGURE_ENV="yes"
if [ -f ".env" ]; then
    echo ""
    echo "ℹ️  An existing .env file was found."
    if [ -t 0 ]; then
        read -r -p "Would you like to re-run the interactive credential setup? [y/N]: " RECONF
        if [[ ! "$RECONF" =~ ^[Yy]$ ]]; then
            CONFIGURE_ENV="no"
        fi
    else
        CONFIGURE_ENV="no"
    fi
fi

if [ "$CONFIGURE_ENV" = "yes" ] && [ -t 0 ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "🔑 STEP 1: Connect Your Trading 212 Account (Read-Only)"
    echo "------------------------------------------------------------------"
    echo "How to get your Trading 212 API Key (takes 30 seconds):"
    echo "  1. Open Trading 212 app -> Settings -> API (Beta)"
    echo "  2. Tap 'Generate API Key'"
    echo "  3. Enable Read permissions: Account Data, Portfolio, Orders, History"
    echo "  4. Copy your API Key ID and Secret Key below"
    echo "------------------------------------------------------------------"
    echo ""

    read -r -p "👤 Enter your first name or portfolio label [default: Main]: " INPUT_NAME
    ACCOUNT_NAME="${INPUT_NAME:-Main}"

    read -r -p "🔑 Enter your Trading 212 API Key ID (or press Enter to test with mock data first): " INPUT_KEY_ID
    T212_KEY_ID="${INPUT_KEY_ID}"

    T212_SECRET=""
    if [ -n "$T212_KEY_ID" ]; then
        read -r -s -p "🔒 Enter your Trading 212 API Secret Key: " INPUT_SECRET
        echo ""
        T212_SECRET="${INPUT_SECRET}"
    fi

    read -r -p "🌐 Account Type — 'live' (Real Money Invest/ISA) or 'demo' (Practice) [default: live]: " INPUT_ENV
    T212_ENV="${INPUT_ENV:-live}"

    read -r -p "🛡️  Protected Cash Reserve Floor in £ GBP (never spent on buy orders) [default: 1500.0]: " INPUT_RESERVE
    MIN_CASH_RESERVE_GBP="${INPUT_RESERVE:-1500.0}"

    read -r -p "🚫 Optional: Any stock tickers to exclude from BUY recommendations? (e.g. TSLA,COIN) [default: none]: " INPUT_EXCLUDED
    EXCLUDED_TICKERS="${INPUT_EXCLUDED:-}"

    SMTP_USER=""
    SMTP_PASS=""
    ALERT_EMAILS=""
    echo ""
    read -r -p "📧 Optional: Configure daily email notifications via Gmail SMTP now? [y/N]: " SETUP_EMAIL
    if [[ "$SETUP_EMAIL" =~ ^[Yy]$ ]]; then
        read -r -p "   Gmail address (SMTP_USER): " SMTP_USER
        read -r -s -p "   16-char Google App Password (SMTP_PASS): " SMTP_PASS
        echo ""
        read -r -p "   Recipient email(s), comma-separated [default: $SMTP_USER]: " INPUT_RECIPIENTS
        ALERT_EMAILS="${INPUT_RECIPIENTS:-$SMTP_USER}"
    fi

    cat > .env <<EOF
# ==============================================================================
# Trading 212 Daily Buy/Sell Advisor Configuration
# ==============================================================================

# --- YOUR PROFILE & TRADING 212 API ---
ACCOUNT_NAME=${ACCOUNT_NAME}
T212_KEY_ID=${T212_KEY_ID}
T212_SECRET=${T212_SECRET}
T212_ENV=${T212_ENV}
CASH_CURRENCY=GBP

# --- PORTFOLIO LIQUIDITY & RISK GUARDRAILS ---
MIN_CASH_RESERVE_GBP=${MIN_CASH_RESERVE_GBP}
EXCLUDED_TICKERS=${EXCLUDED_TICKERS}

# --- OPTIONAL: DAILY EMAIL DIGEST (GMAIL SMTP) ---
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=${SMTP_USER}
SMTP_PASS=${SMTP_PASS}
ALERT_EMAILS=${ALERT_EMAILS}

# --- OPTIONAL: SECOND TRADING 212 ACCOUNT ---
SECOND_ACCOUNT_NAME=
T212_KEY_ID_2=
T212_SECRET_2=
MIN_CASH_RESERVE_2=500.0

# --- OPTIONAL SETTINGS (blank = off; see README "Optional Settings") ---
EARMARKED_TICKERS=
OUTSIDE_HOLDINGS_JSON=
RESERVE_TOPUPS_JSON=
ACCOUNT_WRAPPER=
ACCOUNT_WRAPPER_2=
MAX_LEVERAGED_ETF_PCT=0.08
EOF
    chmod 600 .env
    echo "✅ Saved configuration to .env (permissions locked to 600)."

    echo ""
    echo "------------------------------------------------------------------"
    echo "🧪 STEP 2: Running Initial Verification Check..."
    echo "------------------------------------------------------------------"
    if [ -n "$T212_KEY_ID" ] && [ -n "$T212_SECRET" ]; then
        echo "🔌 Testing live Trading 212 API connection & generating your first recommendation report..."
        python3 main.py --dry-run
    else
        echo "💡 No API key entered yet — running a sample demo audit (--mock) so you can preview the output:"
        python3 main.py --dry-run --mock
    fi
elif [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying .env.example -> .env"
    cp .env.example .env
    chmod 600 .env
    echo "👉 Edit .env with your Trading 212 API credentials or run ./setup.sh in an interactive terminal."
fi

echo ""
echo "=================================================================="
echo "🎉 Setup Complete! Quick Commands:"
echo "  • Daily buy/sell check (terminal):  ./run.sh --dry-run"
echo "  • Open interactive web dashboard:   ./dashboard.sh"
echo "  • Send live email digest:           ./run.sh"
echo "=================================================================="
