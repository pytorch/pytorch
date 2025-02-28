@echo off

echo Dependency ARM Performance Libraries (APL) installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

:: Set download URL for the ARM Performance Libraries (APL)
set DOWNLOAD_URL="https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_24.10/arm-performance-libraries_24.10_Windows.msi"
set INSTALLER_FILE=%DOWNLOADS_DIR%\arm-performance-libraries.msi

:: Download installer
echo Downloading ARM Performance Libraries (APL)...
curl -L -o "%INSTALLER_FILE%" %DOWNLOAD_URL%

:: Install ARM Performance Libraries (APL)
echo Installing ARM Performance Libraries (APL)...
msiexec /i "%INSTALLER_FILE%" /qn /norestart ACCEPT_EULA=1 INSTALLFOLDER="%DEPENDENCIES_DIR%"

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install ARM Performance Libraries (APL) components. (exitcode = %errorlevel%)"
    exit /b 1
)

:: Add to environment
echo ARMPL_DIR=%DEPENDENCIES_DIR%\armpl_24.10\>> %GITHUB_ENV%
echo %DEPENDENCIES_DIR%\armpl_24.10\bin\>> %GITHUB_PATH%

echo Dependency ARM Performance Libraries (APL) installation finished.