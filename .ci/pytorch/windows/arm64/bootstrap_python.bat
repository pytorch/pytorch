@echo off

echo Dependency Python installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

if not defined DESIRED_PYTHON set DESIRED_PYTHON=3.12
set PYTHON_BASE=%DESIRED_PYTHON%
set PYTHON_EXE_NAME=python.exe
set ADDITIONAL_OPTIONS=
if "%DESIRED_PYTHON:~-1%" == "t" (
    set PYTHON_BASE=%DESIRED_PYTHON:~0,-1%
    set PYTHON_EXE_NAME=python%DESIRED_PYTHON%.exe
    set ADDITIONAL_OPTIONS=Include_freethreaded=1
)

:: Keep new Python pins in sync with .ci/wheel/windows/set_desired_python.sh.
if "%PYTHON_BASE%" == "3.15" (
    set PYTHON_FULL=3.15.2
) else if "%PYTHON_BASE%" == "3.14" (
    set PYTHON_FULL=3.14.6
) else if "%PYTHON_BASE%" == "3.13" (
    set PYTHON_FULL=3.13.2
) else if "%PYTHON_BASE%" == "3.12" (
    set PYTHON_FULL=3.12.7
) else if "%PYTHON_BASE%" == "3.11" (
    set PYTHON_FULL=3.11.9
) else (
    echo Unsupported DESIRED_PYTHON=%DESIRED_PYTHON% >&2
    exit /b 1
)
echo Python version is set to %DESIRED_PYTHON%
set DOWNLOAD_URL=https://www.python.org/ftp/python/%PYTHON_FULL%/python-%PYTHON_FULL%-arm64.exe

set INSTALLER_FILE=%DOWNLOADS_DIR%\python-installer.exe

:: Download installer
echo Downloading Python...
curl -fL -o "%INSTALLER_FILE%" "%DOWNLOAD_URL%"
if errorlevel 1 exit /b 1

:: Install Python
echo Installing Python...
start /wait "" "%INSTALLER_FILE%" /quiet Include_debug=1 %ADDITIONAL_OPTIONS% TargetDir="%DEPENDENCIES_DIR%\Python"

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install Python. (exitcode = %errorlevel%)"
    exit /b 1
)

set "DESIRED_PYTHON_EXE=%DEPENDENCIES_DIR%\Python\%PYTHON_EXE_NAME%"
if not exist "%DESIRED_PYTHON_EXE%" (
    echo Python installer did not provide %DESIRED_PYTHON_EXE% >&2
    exit /b 1
)
echo DESIRED_PYTHON_EXE=%DESIRED_PYTHON_EXE%>> "%GITHUB_ENV%"

:: Add to PATH
echo %DEPENDENCIES_DIR%\Python\>> %GITHUB_PATH%
echo %DEPENDENCIES_DIR%\Python\scripts\>> %GITHUB_PATH%
echo %DEPENDENCIES_DIR%\Python\libs\>> %GITHUB_PATH%

echo Dependency Python installation finished.