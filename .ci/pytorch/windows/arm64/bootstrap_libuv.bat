@echo off

echo Dependency libuv installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

:: activate visual studio
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -find VC\Auxiliary\Build\vcvarsall.bat`) do call "%%i" arm64
where cl.exe

cd %DEPENDENCIES_DIR%
:: v1.47.0
git init -q libuv || exit /b 1
git -C libuv fetch -q --depth 1 https://github.com/libuv/libuv.git be6b81a352d17513c95be153afcb3148f1a451cd || exit /b 1
git -C libuv checkout -q FETCH_HEAD || exit /b 1

echo Configuring libuv...
mkdir libuv\build
cd libuv\build
cmake .. -DBUILD_TESTING=OFF

echo Building libuv...
cmake --build . --config Release

echo Installing libuv...
cmake --install . --prefix ../install

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install libuv. (exitcode = %errorlevel%)"
    exit /b 1
)

echo Dependency libuv installation finished.