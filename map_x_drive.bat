@echo off
rem Maps X: to this project folder without storing a personal source path.
subst X: /D >nul 2>&1
subst X: "%~dp0."
