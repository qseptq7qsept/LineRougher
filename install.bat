REM Create a Python virtual environment called "env"
python -m venv env
if errorlevel 1 goto error

REM Activate the virtual environment
call env\Scripts\activate
if errorlevel 1 goto error

REM Install additional dependencies needed for the code
pip install opencv-python numpy PySide6
if errorlevel 1 goto error

echo Installation complete!
pause
goto end

:error
echo An error occurred during installation.
pause

:end
