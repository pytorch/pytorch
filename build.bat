@echo on

call .circleci\scripts\cuda_install.bat
if errorlevel 1 exit /b 1

mkdir magma_build
cd magma_build
call ..\.circleci\scripts\magma_build.bat
if errorlevel 1 exit /b 1