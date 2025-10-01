set PATH=C:\Program Files\CMake\bin;C:\Program Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;C:\Program Files\Amazon\AWSCLI;C:\Program Files\Amazon\AWSCLI\bin;%PATH%

:: Install Miniconda3
set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers

pushd .
if "%VC_VERSION%" == "" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%VC_VERSION%
)
if errorlevel 1 exit /b
if not errorlevel 0 exit /b
@echo on
popd

set DISTUTILS_USE_SDK=1

if not "%USE_CUDA%"=="1" goto cuda_build_end

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%

rem version transformer, for example 10.1 to 10_1.
set VERSION_SUFFIX=%CUDA_VERSION:.=_%
set CUDA_PATH_V%VERSION_SUFFIX%=%CUDA_PATH%

set CUDNN_LIB_DIR=%CUDA_PATH%\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
set CUDNN_ROOT_DIR=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%
set NUMBAPRO_CUDALIB=%CUDA_PATH%\bin
set NUMBAPRO_LIBDEVICE=%CUDA_PATH%\nvvm\libdevice
set NUMBAPRO_NVVM=%CUDA_PATH%\nvvm\bin\nvvm64_32_0.dll

:cuda_build_end

set PYTHONPATH=%TMP_DIR_WIN%\build;%PYTHONPATH%

:: Print all existing environment variable for debugging
set
