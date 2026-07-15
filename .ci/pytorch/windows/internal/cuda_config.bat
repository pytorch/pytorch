@echo off
:: CUDA version configuration
:: Usage: call internal\cuda_config.bat <CUDA_VERSION>
:: e.g., call internal\cuda_config.bat 126

set "CUDA_VER=%~1"

:: Map version code to dotted version and set paths
if "%CUDA_VER%"=="126" (
    set "CUDA_DOTTED_VERSION=12.6"
    set "CUDA_ARCH_LIST=5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0"
    set "VISION_GENCODE=-gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_90,code=compute_90"
) else if "%CUDA_VER%"=="128" (
    set "CUDA_DOTTED_VERSION=12.8"
    set "CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;10.0;12.0"
    set "VISION_GENCODE=-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_90,code=compute_90 -gencode=arch=compute_100,code=compute_100 -gencode=arch=compute_120,code=compute_120"
) else if "%CUDA_VER%"=="129" (
    set "CUDA_DOTTED_VERSION=12.9"
    set "CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;10.0;12.0"
    set "VISION_GENCODE=-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_90,code=compute_90 -gencode=arch=compute_100,code=compute_100 -gencode=arch=compute_120,code=compute_120"
) else if "%CUDA_VER%"=="130" (
    set "CUDA_DOTTED_VERSION=13.0"
    set "CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;10.0;12.0"
    set "VISION_GENCODE=-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_90,code=compute_90 -gencode=arch=compute_100,code=compute_100 -gencode=arch=compute_120,code=compute_120"
) else (
    echo Unknown CUDA version: %CUDA_VER%
    exit /b 1
)

:: Set CUDA path variable name and check for CUDA installation
set "CUDA_PATH_VAR=CUDA_PATH_V%CUDA_VER%"
call set "CUDA_PATH_VAL=%%CUDA_PATH_V%CUDA_VER%%%"

IF "%CUDA_PATH_VAL%"=="" (
    IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_DOTTED_VERSION%\bin\nvcc.exe" (
        set "CUDA_PATH_V%CUDA_VER%=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_DOTTED_VERSION%"
        set "CUDA_PATH_VAL=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_DOTTED_VERSION%"
    ) ELSE (
        echo CUDA %CUDA_DOTTED_VERSION% not found, failing
        exit /b 1
    )
)

:: Set environment variables for the build
set USE_CUDA=
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64

IF "%BUILD_VISION%" == "" (
    set "TORCH_CUDA_ARCH_LIST=%CUDA_ARCH_LIST%"
    set TORCH_NVCC_FLAGS=-Xfatbin -compress-all
) ELSE (
    set "NVCC_FLAGS=-D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr %VISION_GENCODE%"
)

set "CUDA_PATH=%CUDA_PATH_VAL%"
set "PATH=%CUDA_PATH_VAL%\bin;%PATH%"
