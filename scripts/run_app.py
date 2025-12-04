#!/usr/bin/env python3
"""
Alternative launcher for Streamlit app that ensures correct environment.
"""

import sys
import os
import subprocess
from pathlib import Path

# Get project root (parent of scripts/ directory)
project_root = Path(__file__).parent.parent.absolute()

# Check if we're in the virtual environment
venv_python = project_root / "biolink_env" / "bin" / "python"
if not venv_python.exists():
    print("ERROR: Virtual environment not found!")
    print("Please run: python3 -m venv biolink_env")
    sys.exit(1)

# Check if we're using the venv Python
if sys.executable != str(venv_python):
    print("WARNING: Not using virtual environment Python!")
    print(f"Current: {sys.executable}")
    print(f"Expected: {venv_python}")
    print("\nPlease run:")
    print(f"  source {project_root}/biolink_env/bin/activate")
    print("  streamlit run app.py")
    sys.exit(1)

# Verify loguru is installed
try:
    import loguru
except ImportError:
    print("Installing missing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Launch Streamlit
import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(project_root / "app.py")]
    stcli.main()

