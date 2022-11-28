if exist "%TMP_DIR%/ci_scripts/pytorch_env_restore.bat" (
    call %TMP_DIR%/ci_scripts/pytorch_env_restore.bat
    exit /b 0
)

set PATH=C:\Program Files\CMake\bin;C:\Program Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;C:\Program Files\Amazon\AWSCLI;C:\Program Files\Amazon\AWSCLI\bin;%PATH%

:: Install Miniconda3
set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers

:: Miniconda has been installed as part of the Windows AMI with all the dependencies.
:: We just need to activate it here
call %INSTALLER_DIR%\activate_miniconda3.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

:: extra conda dependencies for testing purposes
if NOT "%BUILD_ENVIRONMENT%"=="" (
    call conda install -y -q mkl protobuf numba scipy=1.6.2 typing_extensions dataclasses
    if errorlevel 1 exit /b
    if not errorlevel 0 exit /b
)

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

:: The version is fixed to avoid flakiness: https://github.com/pytorch/pytorch/issues/31136
=======
:: Pin unittest-xml-reporting to freeze printing test summary logic, related: https://github.com/pytorch/pytorch/issues/69014

pip install "ninja==1.10.0.post1" future "hypothesis==5.35.1" "expecttest==0.1.3" "librosa>=0.6.2" "scipy==1.6.3" psutil pillow "unittest-xml-reporting<=3.2.0,>=2.0.0" pytest pytest-xdist pytest-shard pytest-rerunfailures sympy "xdoctest==1.0.2" "pygments==2.12.0" "opt-einsum>=3.3"
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

set DISTUTILS_USE_SDK=1

if not "%USE_CUDA%"=="1" goto cuda_build_end

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%

rem version transformer, for example 10.1 to 10_1.
set VERSION_SUFFIX=%CUDA_VERSION:.=_%
set CUDA_PATH_V%VERSION_SUFFIX%=%CUDA_PATH%

set CUDNN_LIB_DIR=%CUDA_PATH%\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
set CUDNN_ROOT_DIR=%CUDA_PATH%
set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%
set NUMBAPRO_CUDALIB=%CUDA_PATH%\bin
set NUMBAPRO_LIBDEVICE=%CUDA_PATH%\nvvm\libdevice
set NUMBAPRO_NVVM=%CUDA_PATH%\nvvm\bin\nvvm64_32_0.dll

:cuda_build_end

set PYTHONPATH=%TMP_DIR_WIN%\build;%PYTHONPATH%

if NOT "%BUILD_ENVIRONMENT%"=="" (
    pushd %TMP_DIR_WIN%\build
    copy /Y %PYTORCH_FINAL_PACKAGE_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z %TMP_DIR_WIN%\
    :: 7z: -aos skips if exists because this .bat can be called multiple times
    7z x %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z -aos
    popd
) else (
    xcopy /s %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch %TMP_DIR_WIN%\build\torch\
)

@echo off
echo @echo off >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore.bat
for /f "usebackq tokens=*" %%i in (`set`) do echo set "%%i" >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore.bat
@echo on

if NOT "%BUILD_ENVIRONMENT%" == "" (
  :: Create a shortcut to restore pytorch environment
  echo @echo off >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore_helper.bat
  echo call "%TMP_DIR_WIN%/ci_scripts/pytorch_env_restore.bat" >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore_helper.bat
  echo cd /D "%CD%" >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore_helper.bat

  aws s3 cp "s3://ossci-windows/Restore PyTorch Environment.lnk" "C:\Users\circleci\Desktop\Restore PyTorch Environment.lnk"
)
