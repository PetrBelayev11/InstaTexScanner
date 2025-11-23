@echo off
chcp 65001 >nul
echo ========================================
echo   InstaTexScanner - Local Launch
echo ========================================
echo.

REM Check for Python
where py >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found! Please install Python 3.9 or higher.
    echo Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check dependencies
echo 📦 Checking dependencies...
py -m pip show fastapi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Dependencies not installed. Installing...
    echo.
    py -m pip install -r code/deployment/api/requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Error installing dependencies!
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
) else (
    echo ✅ Dependencies already installed
)

echo.
echo ========================================
echo   Starting servers...
echo ========================================
echo.
echo 📝 Note: Two terminal windows will open:
echo    1. API server (port 8000)
echo    2. Frontend server (port 3000)
echo.
echo Press any key to start...
pause >nul

REM Start API server in new window
start "InstaTexScanner API" cmd /k "py run_api.py"

REM Wait a bit before starting frontend
timeout /t 2 /nobreak >nul

REM Start Frontend server in new window
start "InstaTexScanner Frontend" cmd /k "py run_frontend.py"

echo.
echo ✅ Servers started!
echo.
echo 🌐 Frontend: http://localhost:3000
echo 🔌 API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo To stop, close the terminal windows or press Ctrl+C in each window.
echo.
pause

