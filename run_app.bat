@echo off
title Kolam Inverse - Universal Mathematical Studio
cls
echo ======================================================================
echo           KOLAM INVERSE: Universal Mathematical Studio
echo ======================================================================
echo.
echo Launching the Artsy Cultural & Mathematical Studio on Port 8502...
echo.
cd /d "%~dp0"
"C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app_artsy.py --server.port 8502 --server.headless false
pause
