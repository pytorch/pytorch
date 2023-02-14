if "%DEBUG%" == "1" (
  set BUILD_TYPE=debug
) ELSE (
  set BUILD_TYPE=release
)

set PATH=C:\Program Files\CMake\bin;C:\Program Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;C:\Program Files\Amazon\AWSCLI;C:\Program Files\Amazon\AWSCLI\bin;%PATH%

:: This inflates our log size slightly, but it is REALLY useful to be
:: able to see what our cl.exe commands are (since you can actually
:: just copy-paste them into a local Windows setup to just rebuild a
:: single file.)
:: log sizes are too long, but leaving this here incase someone wants to use it locally
:: set CMAKE_VERBOSE_MAKEFILE=1


set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers


call %INSTALLER_DIR%\install_mkl.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

call %INSTALLER_DIR%\install_magma.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

call %INSTALLER_DIR%\install_sccache.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

:: Miniconda has been installed as part of the Windows AMI with all the dependencies.
:: We just need to activate it here
call %INSTALLER_DIR%\activate_miniconda3.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

:: Override VS env here
pushd .
if "%VC_VERSION%" == "" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%VC_VERSION%
)
if errorlevel 1 exit /b
if not errorlevel 0 exit /b
@echo on
popd
