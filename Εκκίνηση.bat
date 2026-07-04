@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   Επενδυτικό Dashboard - εκκίνηση...
echo   Θα ανοίξει στον browser. Κλείσε αυτό το
echo   παράθυρο για να το σταματήσεις.
echo ==========================================
py -m streamlit run app.py
pause
