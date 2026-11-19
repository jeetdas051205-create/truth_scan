@echo off
title TruthScan AI - Fake News Detection System
echo ==============================================================================
echo             TRUTHSCAN AI - AI-POWERED FAKE NEWS DETECTION SYSTEM
echo                 B.Tech CSE (AI & ML) Capstone Demonstration
echo ==============================================================================
echo.
echo [*] Launching TruthScan AI Server...
echo [*] Opening browser at http://127.0.0.1:5000 ...
start "" http://127.0.0.1:5000
"..\venv\Scripts\python.exe" app.py
pause

