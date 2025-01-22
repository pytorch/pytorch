@echo off

echo Dependency MSVC Build Tools with C++ with ARM64/ARM64EC components installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir "%DOWNLOADS_DIR%"
if not exist "%DEPENDENCIES_DIR%" mkdir "%DEPENDENCIES_DIR%"

:: Set download URL for the Visual Studio Installer
set DOWNLOAD_URL=https://aka.ms/vs/17/release/vs_BuildTools.exe
set INSTALLER_FILE=%DOWNLOADS_DIR%\vs_BuildTools.exe

:: Download installer
echo Downloading Visual Studio Build Tools with C++ installer...
curl -L -o "%INSTALLER_FILE%" %DOWNLOAD_URL%

:: Install the Visual Studio Build Tools with C++ components
echo Installing Visual Studio Build Tools with C++ components...
echo Installing MSVC %MSVC_VERSION%
if "%MSVC_VERSION%" == "latest" (
    "%INSTALLER_FILE%" --norestart --nocache --quiet --wait --installPath "%DEPENDENCIES_DIR%\VSBuildTools" ^
        --add Microsoft.VisualStudio.Component.Windows11SDK.22621 ^
        --add Microsoft.VisualStudio.Component.VC.ASAN ^
        --add Microsoft.VisualStudio.Component.VC.CMake.Project ^
        --add Microsoft.VisualStudio.Component.VC.Tools.ARM64 ^
        --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64
) else if "%MSVC_VERSION%" == "14.40" (
    "%INSTALLER_FILE%" --norestart --nocache --quiet --wait --installPath "%DEPENDENCIES_DIR%\VSBuildTools" ^
        --add Microsoft.VisualStudio.Component.Windows11SDK.22621 ^
        --add Microsoft.VisualStudio.Component.VC.ASAN ^
        --add Microsoft.VisualStudio.Component.VC.CMake.Project ^
        --add Microsoft.VisualStudio.Component.VC.14.40.17.10.ARM64 ^
        --add Microsoft.VisualStudio.Component.VC.14.40.17.10.x86.x64
) else if "%MSVC_VERSION%" == "14.36" (
    "%INSTALLER_FILE%" --norestart --nocache --quiet --wait --installPath "%DEPENDENCIES_DIR%\VSBuildTools" ^
        --add Microsoft.VisualStudio.Component.Windows11SDK.22621 ^
        --add Microsoft.VisualStudio.Component.VC.ASAN ^
        --add Microsoft.VisualStudio.Component.VC.CMake.Project ^
        --add Microsoft.VisualStudio.Component.VC.14.36.17.6.ARM64 ^
        --add Microsoft.VisualStudio.Component.VC.14.36.17.6.x86.x64
)

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install Visual Studio Build Tools with C++ components. (exitcode = %errorlevel%)"
    exit /b 1
)

echo Dependency Visual Studio Build Tools with C++ installation finished.