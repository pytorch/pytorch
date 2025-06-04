@echo off

echo Dependency Rust installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

set DOWNLOAD_URL="https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"
set INSTALLER_FILE=%DOWNLOADS_DIR%\rustup-init.exe
set RUSTUP_HOME=%DEPENDENCIES_DIR%\rust
set CARGO_HOME=%DEPENDENCIES_DIR%\cargo

:: Download installer
echo Downloading Rust...
curl -L -o "%INSTALLER_FILE%" %DOWNLOAD_URL%

:: Install APL
echo Installing Rust...
"%INSTALLER_FILE%" -q -y --default-host aarch64-pc-windows-msvc --default-toolchain stable --profile default

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install Rust. (exitcode = %errorlevel%)"
    exit /b 1
)

:: Add to PATH
echo %DEPENDENCIES_DIR%\cargo\bin\>> %GITHUB_PATH%
echo RUSTUP_HOME=%DEPENDENCIES_DIR%\rust>> %GITHUB_ENV%
echo CARGO_HOME=%DEPENDENCIES_DIR%\cargo>> %GITHUB_ENV%

echo Dependency Rust installation finished.