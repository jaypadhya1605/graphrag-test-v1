#!/bin/bash
set -e
echo "Starting Streamlit Knowledge Graph App..."

# Set required environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Change to the application directory
cd /home/site/wwwroot

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found in $(pwd)"
    ls -la
    exit 1
fi

echo "Found app.py, starting Streamlit..."

# Start Streamlit with explicit configuration
python -m streamlit run app.py \
    --server.port=8000 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
