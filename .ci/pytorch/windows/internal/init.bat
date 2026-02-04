@echo off
:: Common initialization script for all build entry points
:: Sets up repository and checks dependencies

set MODULE_NAME=pytorch

IF NOT EXIST "setup.py" IF NOT EXIST "%MODULE_NAME%" (
    call internal\clone.bat
    cd %~dp0\..
) ELSE (
    call internal\clean.bat
)
IF ERRORLEVEL 1 goto :eof

call internal\check_deps.bat
IF ERRORLEVEL 1 goto :eof
