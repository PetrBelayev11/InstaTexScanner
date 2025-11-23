#!/bin/bash

echo "========================================"
echo "  InstaTexScanner - Local Launch"
echo "========================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3.9 or higher."
    exit 1
fi

echo "✅ Python found"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
if ! python3 -m pip show fastapi &> /dev/null; then
    echo "⚠️  Dependencies not installed. Installing..."
    echo ""
    python3 -m pip install -r code/deployment/api/requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error installing dependencies!"
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "========================================"
echo "  Starting servers..."
echo "========================================"
echo ""
echo "📝 Note: Two terminal windows will open:"
echo "   1. API server (port 8000)"
echo "   2. Frontend server (port 3000)"
echo ""

# Start API server in background
python3 run_api.py &
API_PID=$!

# Wait a bit before starting frontend
sleep 2

# Start Frontend server in background
python3 run_frontend.py &
FRONTEND_PID=$!

echo ""
echo "✅ Servers started!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "PID API: $API_PID"
echo "PID Frontend: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for interrupt signal
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait

