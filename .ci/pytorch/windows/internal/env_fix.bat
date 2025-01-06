@echo off

:: Caution: Please don't use this script locally
:: It may destroy your build environment.

setlocal

if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    echo Visual Studio %VC_YEAR% C++ BuildTools is required to compile PyTorch on Windows
    exit /b 1
)

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18
if "%VC_YEAR%" == "2019" (
    set VC_VERSION_LOWER=16
    set VC_VERSION_UPPER=17
)

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere

if "%VS15VCVARSALL%"=="" (
    echo Visual Studio %VC_YEAR% C++ BuildTools is required to compile PyTorch on Windows
    exit /b 1
)

call "%VS15VCVARSALL%" x86_amd64
for /f "usebackq tokens=*" %%i in (`where link.exe`) do move "%%i" "%%i.bak"

endlocal
