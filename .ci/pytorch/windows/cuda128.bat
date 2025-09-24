@echo off

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

set USE_CUDA=
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64

IF "%NVTOOLSEXT_PATH%"=="" (
    IF EXIST "C:\Program Files\NVIDIA Corporation\NvToolsExt\lib\x64\nvToolsExt64_1.lib"  (
        set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
    ) ELSE (
        echo NVTX ^(Visual Studio Extension ^for CUDA^) ^not installed, failing
        exit /b 1
    )
)

IF "%CUDA_PATH_V128%"=="" (
    IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" (
        set "CUDA_PATH_V128=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    ) ELSE (
        echo CUDA 12.8 not found, failing
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('python .github\scripts\get_ci_variable.py --cuda-arch-list 12.8') do set TORCH_CUDA_ARCH_LIST=%%i
set TORCH_NVCC_FLAGS=-Xfatbin -compress-all

set "CUDA_PATH=%CUDA_PATH_V128%"
set "PATH=%CUDA_PATH_V128%\bin;%PATH%"

:optcheck

call internal\check_opts.bat
IF ERRORLEVEL 1 goto :eof

if exist "%NIGHTLIES_PYTORCH_ROOT%" cd %NIGHTLIES_PYTORCH_ROOT%\..
call  %~dp0\internal\copy.bat
IF ERRORLEVEL 1 goto :eof

call  %~dp0\internal\setup.bat
IF ERRORLEVEL 1 goto :eof
