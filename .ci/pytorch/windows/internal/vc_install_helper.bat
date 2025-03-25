if "%VC_YEAR%" == "2022" powershell windows/internal/vs2022_install.ps1

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18


for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"  -products Microsoft.VisualStudio.Product.BuildTools -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        goto vswhere
    )
)

:vswhere
echo "Setting VSINSTALLDIR to %VS15INSTALLDIR%"

if errorlevel 1 exit /b 1
