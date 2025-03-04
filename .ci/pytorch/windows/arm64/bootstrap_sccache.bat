@echo off

echo Dependency sccache installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

:: Set download URL for the sccache
set DOWNLOAD_URL="https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-pc-windows-msvc.zip"
set INSTALLER_FILE=%DOWNLOADS_DIR%\sccache.zip

:: Download installer
echo Downloading sccache.zip...
curl -L -o "%INSTALLER_FILE%" %DOWNLOAD_URL%

:: Install sccache
echo Extracting sccache.zip...
tar -xf "%INSTALLER_FILE%" -C %DEPENDENCIES_DIR%
cd %DEPENDENCIES_DIR%
ren sccache-v0.8.1-x86_64-pc-windows-msvc sccache
cd ..

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install sccache. (exitcode = %errorlevel%)"
    exit /b 1
)

:: Add to PATH
echo %DEPENDENCIES_DIR%\sccache\>> %GITHUB_PATH%

echo Dependency sccache installation finished.