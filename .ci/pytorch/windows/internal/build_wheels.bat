@echo off
:: Debug-libtorch-only entry point. The wheel + release-libtorch Windows
:: CD path lives in the sibling Python pipeline (build_env_setup.py /
:: build_install_deps.py / build_wheel.py) invoked by binary_windows_build.sh.
::
:: This file (and the build_pytorch.bat / setup_build.bat / cpu.bat /
:: cuda.bat / xpu.bat / internal\setup.bat chain it drives) is retained
:: solely so the debug-libtorch nightly workflow keeps producing artifacts
:: until that path is ported in a follow-up. Do not extend it.

if not "%LIBTORCH_CONFIG%" == "debug" (
    echo build_wheels.bat is debug-libtorch-only; non-debug builds must
    echo go through binary_windows_build.sh's Python pipeline path.
    exit /b 1
)

call windows/internal/vc_install_helper.bat
if errorlevel 1 exit /b 1

call windows/internal/cuda_install.bat
if errorlevel 1 exit /b 1

call windows/internal/xpu_install.bat
if errorlevel 1 exit /b 1

call windows/build_pytorch.bat %CUDA_VERSION% %PYTORCH_BUILD_VERSION% %PYTORCH_BUILD_NUMBER%
if errorlevel 1 exit /b 1
