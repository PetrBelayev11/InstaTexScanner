# Deployment Guide

This guide covers how to deploy and run the InstaTexScanner application.

## Prerequisites
- Python 3.8+

## Local Development

### 1. Start the Backend Server

```bash
# Navigate to backend directory
cd api

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

The backend will start at: `http://localhost:8000`

**Verify backend is running:**
```bash
curl http://localhost:8000/health
```

### 2. Start the Frontend Server

```bash
# Navigate to frontend directory (in a new terminal)
cd app

# Start a simple HTTP server
python -m http.server 3000
```

The frontend will be available at: `http://localhost:3000`

### 3. Access the Application
- Open your browser and go to: `http://localhost:3000`
- Backend API documentation: `http://localhost:8000/docs`