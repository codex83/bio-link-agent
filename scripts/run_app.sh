#!/bin/bash
# Script to run the Streamlit app

# Get the directory where this script is located (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Change to project root (parent directory)
cd "$SCRIPT_DIR/.."

# Activate virtual environment
source biolink_env/bin/activate

# Verify we're using the right Python
echo "Using Python: $(which python)"
echo "Using Streamlit: $(which streamlit)"

# Check if loguru is installed
python -c "import loguru; print('✓ loguru installed')" || {
    echo "Installing missing dependencies..."
    pip install -r requirements.txt
}

# Run Streamlit
streamlit run app.py

