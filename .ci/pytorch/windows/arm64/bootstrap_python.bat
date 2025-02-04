@echo off

echo Dependency Python installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

if "%PYTHON_VERSION%"=="Python312" (
    echo Python version is set to Python312
    set DOWNLOAD_URL="https://www.python.org/ftp/python/3.12.7/python-3.12.7-arm64.exe"
) else if "%PYTHON_VERSION%"=="Python311" (
    echo Python version is set to Python311
    set DOWNLOAD_URL="https://www.python.org/ftp/python/3.11.9/python-3.11.9-arm64.exe"
) else (
    echo PYTHON_VERSION not defined, Python version is set to Python312
    set DOWNLOAD_URL="https://www.python.org/ftp/python/3.12.7/python-3.12.7-arm64.exe"
)

set INSTALLER_FILE=%DOWNLOADS_DIR%\python-installer.exe

:: Download installer
echo Downloading Python...
curl -L -o "%INSTALLER_FILE%" %DOWNLOAD_URL%

:: Install Python
echo Installing Python...
"%INSTALLER_FILE%" /quiet Include_debug=1 TargetDir="%DEPENDENCIES_DIR%\Python"

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install Python. (exitcode = %errorlevel%)"
    exit /b 1
)

:: Add to PATH
echo %DEPENDENCIES_DIR%\Python\>> %GITHUB_PATH%
echo %DEPENDENCIES_DIR%\Python\scripts\>> %GITHUB_PATH%
echo %DEPENDENCIES_DIR%\Python\libs\>> %GITHUB_PATH%

echo Dependency Python installation finished.