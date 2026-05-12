@echo off
:: Unified CUDA build entry point
:: Usage: cuda.bat <version>  e.g., cuda.bat 126

if "%~1"=="" (
    echo CUDA version required. Usage: cuda.bat ^<version^>
    echo Example: cuda.bat 126
    exit /b 1
)

call internal\init.bat
IF ERRORLEVEL 1 goto :eof

call internal\check_nvtx.bat
IF ERRORLEVEL 1 goto :eof

call internal\cuda_config.bat %~1
IF ERRORLEVEL 1 goto :eof

call internal\check_opts.bat
IF ERRORLEVEL 1 goto :eof

if exist "%NIGHTLIES_PYTORCH_ROOT%" cd %NIGHTLIES_PYTORCH_ROOT%\..
call %~dp0\internal\copy.bat
IF ERRORLEVEL 1 goto :eof

call %~dp0\internal\setup.bat
IF ERRORLEVEL 1 goto :eof
