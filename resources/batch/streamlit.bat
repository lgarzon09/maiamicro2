@echo off
cd..
cd..
setlocal
set PROJECTPATH=%cd%
set PYTHONPATH=%PROJECTPATH%
set MAINPATH=%PROJECTPATH%\streamlit_app.py
echo "My project path is: '%PROJECTPATH%'"
echo "Run command is: 'python -m streamlit run "%MAINPATH%"'"
python -m streamlit run "%MAINPATH%"
endlocal
