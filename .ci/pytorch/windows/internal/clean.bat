@echo off

cd %MODULE_NAME%
if exist build rmdir /s /q build
cd ..
