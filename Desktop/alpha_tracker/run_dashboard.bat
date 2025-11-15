@echo off
cd /d "C:\Users\cocac\Desktop\alpha_tracker"
call "C:\Users\cocac\finance-env\Scripts\activate.bat"
streamlit run dashboard.py
pause
