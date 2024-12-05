@echo off

set MODULE_NAME=pytorch

IF NOT EXIST "setup.py" IF NOT EXIST "%MODULE_NAME%" (
    call internal\clone.bat
    cd ..
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

call internal\copy_cpu.bat
IF ERRORLEVEL 1 goto :eof

call internal\setup.bat
IF ERRORLEVEL 1 goto :eof
