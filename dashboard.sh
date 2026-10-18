#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

if [ -d ".venv" ]; then
    source ".venv/bin/activate"
fi

# If dashboard.html does not exist, run a dry-run to generate it first
if [ ! -f "dashboard.html" ]; then
    echo "⚡ Dashboard not yet generated. Running intelligence pipeline to generate initial snapshot..."
    python3 main.py --dry-run
fi

if [[ "$1" == "--serve" ]]; then
    PORT="${2:-8080}"
    echo "🚀 Serving dashboard at http://localhost:$PORT/dashboard.html"
    python3 -m http.server "$PORT"
else
    echo "🌐 Opening Trading 212 Institutional Dashboard in your default browser..."
    if command -v open >/dev/null 2>&1; then
        open "$DIR/dashboard.html"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$DIR/dashboard.html"
    else
        echo "Dashboard available at: file://$DIR/dashboard.html"
    fi
fi
