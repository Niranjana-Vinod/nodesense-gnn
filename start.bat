@echo off
echo ======================================================================
echo  NodeSense — Explainable and Robust GNNs for Citation Networks
echo ======================================================================
echo.

REM Try py -3.12 first, fallback to python
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYCMD=py -3.12
) else (
    set PYCMD=python
)

echo [*] Using Python: %PYCMD%
echo [*] Starting NodeSense Web Application...
echo [*] Opening browser at http://localhost:5000
echo.

start "" http://localhost:5000
%PYCMD% backend\app.py

pause
