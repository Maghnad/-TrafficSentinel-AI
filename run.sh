#!/bin/bash

echo "🚦 Starting TrafficSentinel AI Setup..."

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 could not be found. Please install Python 3.9+ to continue."
    exit
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing required packages..."
pip install -r requirements.txt

# Start the application
echo "🚀 Launching TrafficSentinel AI Dashboard..."
streamlit run app.py
