#!/usr/bin/env python3
"""
Launch script for the Traffic Flow Prediction Dashboard
Run this script to start the Streamlit dashboard
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import plotly
        import sklearn
        import torch
        import psutil
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def main():
    """Main function to launch the dashboard"""
    print("🚗 Starting Traffic Flow Prediction Dashboard...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('docs/results.json'):
        print("❌ Error: docs/results.json not found in current directory")
        print("Please run this script from the project directory")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Launch Streamlit
    try:
        print("🌐 Launching dashboard in your browser...")
        print("📊 Dashboard will be available at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the dashboard")
        print("=" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
