@echo off
title Dog Skin AI - Diagnostic Web Application
echo ==================================================================
echo   Dog Skin Disease Detection Platform
echo   Serving Full-Stack AI Web Application on http://localhost:5000/
echo ==================================================================
echo Opening web interface in your default browser...
start "" "http://localhost:5000/"
echo Starting Python Flask AI server...
python app.py
pause
