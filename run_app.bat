@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" /min cmd /c "timeout /t 4 >nul && start http://localhost:7860"
python src\app.py
pause
