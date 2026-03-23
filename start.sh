#!/usr/bin/env bash
# ---------------------------------------------------------
# LogAnomalyML - Start Script
# Starts the Flask ML API and Rails web application
# ---------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================"
echo " LogAnomalyML - BGL Log Anomaly Detection System"
echo " MSc AI Dissertation  -  Subhan Jameel - U2926092"
echo "======================================================"

# -- 1. Flask ML API --------------------------------------
echo ""
echo ">  Starting Flask ML API (port 5001) ..."
cd "$SCRIPT_DIR/ml_pipeline"

# Install Python deps if needed
if ! python3 -c "import flask" 2>/dev/null; then
  echo "   Installing Python dependencies ..."
  pip3 install -r requirements.txt -q
fi

# Start API in background (auto-trains on synthetic data if no model found)
python3 api/app.py &
ML_PID=$!
echo "   ML API PID: $ML_PID"

# Wait for API to be ready
echo "   Waiting for ML API to be ready ..."
for i in {1..20}; do
  if curl -s http://localhost:5001/api/v1/health > /dev/null 2>&1; then
    echo "   [OK] ML API ready at http://localhost:5001"
    break
  fi
  sleep 2
done

# -- 2. Rails App -----------------------------------------
echo ""
echo ">  Starting Rails application (port 3000) ..."
cd "$SCRIPT_DIR/log_anomaly_rails"

source ~/.rvm/scripts/rvm 2>/dev/null || true
rvm use 3.3.4 2>/dev/null || true

export ML_API_URL=http://localhost:5001

bundle exec rails server -p 3000 &
RAILS_PID=$!
echo "   Rails PID: $RAILS_PID"

sleep 3

echo ""
echo "======================================================"
echo " [OK]  System running:"
echo "    Dashboard:   http://localhost:3000"
echo "    ML API:      http://localhost:5001/api/v1/health"
echo "======================================================"
echo ""
echo " Press Ctrl+C to stop both services."
echo ""

# Wait and handle shutdown
trap "echo ''; echo 'Stopping services...'; kill $ML_PID $RAILS_PID 2>/dev/null; exit 0" INT TERM
wait
