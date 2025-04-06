REM Create a Python virtual environment called "env"
python -m venv env
if errorlevel 1 goto error

REM Activate the virtual environment
call env\Scripts\activate
if errorlevel 1 goto error

REM Install additional dependencies needed for the code
pip install opencv-python numpy PySide6
if errorlevel 1 goto error

REM Create the ffmpeg folder if it doesn't exist
if not exist ffmpeg (
    mkdir ffmpeg
)

REM Download ffmpeg.zip from a prebuilt release (adjust URL if needed)
echo Downloading ffmpeg...
curl -L "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -o ffmpeg\ffmpeg.zip
if errorlevel 1 goto error

REM Extract ffmpeg.zip into a temporary folder using PowerShell
echo Extracting ffmpeg...
powershell -Command "Expand-Archive -Path 'ffmpeg\ffmpeg.zip' -DestinationPath 'ffmpeg\temp'" 
if errorlevel 1 goto error

REM Move ffmpeg.exe (from the bin folder inside the extracted archive) to the ffmpeg folder
for /d %%i in (ffmpeg\temp\*) do (
    if exist "%%i\bin\ffmpeg.exe" (
        move "%%i\bin\ffmpeg.exe" ffmpeg\
    )
)
if errorlevel 1 goto error

REM Clean up temporary extraction folder and downloaded zip file
rd /s /q ffmpeg\temp
del ffmpeg\ffmpeg.zip

echo Installation complete! -q7
pause
goto end

:error
echo An error occurred during installation.
pause

:end
