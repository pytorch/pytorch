@echo off

curl -k https://www.7-zip.org/a/7z1805-x64.exe -O
if errorlevel 1 exit /b 1

start /wait 7z1805-x64.exe /S
if errorlevel 1 exit /b 1

set "PATH=%ProgramFiles%\7-Zip;%PATH%"
