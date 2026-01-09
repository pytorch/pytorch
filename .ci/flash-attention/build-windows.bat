
@echo on
setlocal enabledelayedexpansion

:: build FA3 wheel for Windows

if "%FA_FINAL_PACKAGE_DIR%"=="" (
    echo ERROR: FA_FINAL_PACKAGE_DIR must be set
    exit /b 1
)

if "%PYTORCH_ROOT%"=="" (
    echo ERROR: PYTORCH_ROOT must be set
    exit /b 1
)

:: Set up CUDA environment
set /a CUDA_VER=%CUDA_VERSION%
set CUDA_VER_MAJOR=%CUDA_VERSION:~0,-1%
set CUDA_VER_MINOR=%CUDA_VERSION:~-1,1%
set CUDA_VERSION_STR=%CUDA_VER_MAJOR%.%CUDA_VER_MINOR%

set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

echo CUDA_PATH=%CUDA_PATH%
echo CUDA_VERSION_STR=%CUDA_VERSION_STR%

if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ERROR: CUDA %CUDA_VERSION_STR% not found at %CUDA_PATH%
    exit /b 1
)

pushd "%PYTORCH_ROOT%\.ci\pytorch"
call windows\internal\vc_install_helper.bat
if errorlevel 1 (
    echo ERROR: Failed to discover Visual Studio installation
    popd
    exit /b 1
)
popd

set "VS15VCVARSALL=%VS15INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat"
echo Found Visual Studio at: %VS15INSTALLDIR%

call "%VS15VCVARSALL%" x64

set "FLASH_ATTENTION_DIR=%PYTORCH_ROOT%\third_party\flash-attention"
set "FLASH_ATTENTION_HOPPER_DIR=%FLASH_ATTENTION_DIR%\hopper"

if not exist "%FLASH_ATTENTION_HOPPER_DIR%" (
    echo ERROR: flash attn directory not found %FLASH_ATTENTION_HOPPER_DIR%
    exit /b 1
)

echo Installing dependencies...
python -m pip install einops packaging ninja numpy wheel setuptools
if errorlevel 1 exit /b 1

set FLASH_ATTENTION_FORCE_BUILD=TRUE
set FLASH_ATTENTION_DISABLE_SPLIT=FALSE
set FLASH_ATTENTION_DISABLE_PAGEDKV=FALSE
set FLASH_ATTENTION_DISABLE_APPENDKV=FALSE
set FLASH_ATTENTION_DISABLE_LOCAL=FALSE
set FLASH_ATTENTION_DISABLE_SOFTCAP=FALSE
set FLASH_ATTENTION_DISABLE_PACKGQA=FALSE
set FLASH_ATTENTION_DISABLE_FP16=FALSE
set FLASH_ATTENTION_DISABLE_FP8=FALSE
set FLASH_ATTENTION_DISABLE_VARLEN=FALSE
set FLASH_ATTENTION_DISABLE_CLUSTER=FALSE
set FLASH_ATTENTION_DISABLE_HDIM64=FALSE
set FLASH_ATTENTION_DISABLE_HDIM96=FALSE
set FLASH_ATTENTION_DISABLE_HDIM128=FALSE
set FLASH_ATTENTION_DISABLE_HDIM192=FALSE
set FLASH_ATTENTION_DISABLE_HDIM256=FALSE
set FLASH_ATTENTION_DISABLE_SM80=FALSE
set FLASH_ATTENTION_ENABLE_VCOLMAJOR=FALSE
set FLASH_ATTENTION_DISABLE_HDIMDIFF64=FALSE
set FLASH_ATTENTION_DISABLE_HDIMDIFF192=FALSE

if "%NVCC_THREADS%"=="" set NVCC_THREADS=4
if "%MAX_JOBS%"=="" set MAX_JOBS=%NUMBER_OF_PROCESSORS%

echo NVCC_THREADS=%NVCC_THREADS%
echo MAX_JOBS=%MAX_JOBS%
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

pushd "%FLASH_ATTENTION_HOPPER_DIR%"
if errorlevel 1 exit /b 1

git submodule update --init ../csrc/cutlass
if errorlevel 1 exit /b 1

:: build the wheel
python setup.py bdist_wheel -d "%FA_FINAL_PACKAGE_DIR%" -k --plat-name win_amd64
if errorlevel 1 (
    echo ERROR: Failed to build wheel
    popd
    exit /b 1
)

echo Wheel built successfully:
dir "%FA_FINAL_PACKAGE_DIR%\*.whl"

popd
exit /b 0
