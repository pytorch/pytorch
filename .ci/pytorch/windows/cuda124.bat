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

IF "%CUDA_PATH_V124%"=="" (
    IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" (
        set "CUDA_PATH_V124=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    ) ELSE (
        echo CUDA 12.4 not found, failing
        exit /b 1
    )
)

IF "%BUILD_VISION%" == "" (
    set TORCH_CUDA_ARCH_LIST=5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0
    set TORCH_NVCC_FLAGS=-Xfatbin -compress-all
) ELSE (
    set NVCC_FLAGS=-D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_90,code=compute_90
)

set "CUDA_PATH=%CUDA_PATH_V124%"
set "PATH=%CUDA_PATH_V124%\bin;%PATH%"

:optcheck

call internal\check_opts.bat
IF ERRORLEVEL 1 goto :eof

if exist "%NIGHTLIES_PYTORCH_ROOT%" cd %NIGHTLIES_PYTORCH_ROOT%\..
call  %~dp0\internal\copy.bat
IF ERRORLEVEL 1 goto :eof

call  %~dp0\internal\setup.bat
IF ERRORLEVEL 1 goto :eof
