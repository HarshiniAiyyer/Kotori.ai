@echo off
setlocal

echo Starting Kotori.ai Development Environment
echo.

:: Go to project root
cd /d "%~dp0"

:: Step 1: Activating virtual environment
echo Step 1: Activating virtual environment...
call go\Scripts\activate.bat
echo.

:: Step 2: Starting API server
echo Step 2: Starting API server...
pushd backend
start "Kotori.ai Backend" cmd /k "python api.py"
popd
echo.

:: Step 3: Starting React frontend
echo Step 3: Starting React frontend...
pushd frontend
start "Kotori.ai Frontend" cmd /k "npm start"
popd
echo.

echo Development environment started!
echo API server running at http://localhost:8000
echo React frontend running at http://localhost:3000
echo.
echo Press any key to close all development servers...
pause > nul

:: Stop development servers
taskkill /f /im node.exe > nul 2>&1
taskkill /f /im python.exe > nul 2>&1
call go\Scripts\deactivate.bat

echo.
echo Development servers stopped.
pause
endlocal