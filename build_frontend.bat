@echo off
echo Building Kotori.ai Frontend for production...

cd /d %~dp0\frontend

echo.
echo Step 1: Installing dependencies...
call npm install

echo.
echo Step 2: Building production bundle...
call npm run build

echo.
echo Build completed!
echo Production files are available in frontend\build directory

echo.
echo You can serve these files with a static file server or copy them to your web server.
echo.

pause