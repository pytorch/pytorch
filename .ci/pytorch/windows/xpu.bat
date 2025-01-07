@echo on

set MODULE_NAME=pytorch

IF NOT EXIST "setup.py" IF NOT EXIST "%MODULE_NAME%" (
    call internal\clone.bat
    cd %~dp0
) ELSE (
    call internal\clean.bat
)
IF ERRORLEVEL 1 goto :eof

call internal\check_deps.bat
IF ERRORLEVEL 1 goto :eof

REM Check for optional components

echo Disabling CUDA
set USE_CUDA=0

call internal\check_opts.bat
IF ERRORLEVEL 1 goto :eof

echo Activate XPU Bundle env
set VS2022INSTALLDIR=%VS15INSTALLDIR%
set XPU_BUNDLE_ROOT=%ProgramFiles(x86)%\Intel\oneAPI
call "%XPU_BUNDLE_ROOT%\compiler\latest\env\vars.bat"
call "%XPU_BUNDLE_ROOT%\ocloc\latest\env\vars.bat"
IF ERRORLEVEL 1 goto :eof

:: Workaround for https://github.com/pytorch/pytorch/issues/134989
set CMAKE_SHARED_LINKER_FLAGS=/FORCE:MULTIPLE
set CMAKE_MODULE_LINKER_FLAGS=/FORCE:MULTIPLE
set CMAKE_EXE_LINKER_FLAGS=/FORCE:MULTIPLE

if exist "%NIGHTLIES_PYTORCH_ROOT%" cd %NIGHTLIES_PYTORCH_ROOT%\..
call %~dp0\internal\copy_cpu.bat
IF ERRORLEVEL 1 goto :eof

call %~dp0\internal\setup.bat
IF ERRORLEVEL 1 goto :eof
