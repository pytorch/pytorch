@echo off

call internal\init.bat
IF ERRORLEVEL 1 goto :eof

echo Disabling CUDA
set USE_CUDA=0

call internal\check_opts.bat
IF ERRORLEVEL 1 goto :eof

if exist "%NIGHTLIES_PYTORCH_ROOT%" cd %NIGHTLIES_PYTORCH_ROOT%\..
call %~dp0\internal\copy_cpu.bat
IF ERRORLEVEL 1 goto :eof

call %~dp0\internal\setup.bat
IF ERRORLEVEL 1 goto :eof
