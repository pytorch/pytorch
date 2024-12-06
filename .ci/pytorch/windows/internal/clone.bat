@echo off

cd "%NIGHTLIES_PYTORCH_ROOT%"
git submodule update --init --recursive
IF ERRORLEVEL 1 exit /b 1
