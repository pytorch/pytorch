if "%DEBUG%" == "1" (
  set BUILD_TYPE=debug
) ELSE (
  set BUILD_TYPE=release
)

set PATH=C:\Program Files\CMake\bin;C:\Program Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;C:\Program Files\Amazon\AWSCLI;C:\Program Files\Amazon\AWSCLI\bin;%PATH%

:: This inflates our log size slightly, but it is REALLY useful to be
:: able to see what our cl.exe commands are (since you can actually
:: just copy-paste them into a local Windows setup to just rebuild a
:: single file.)
set CMAKE_VERBOSE_MAKEFILE=1


set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers

call %INSTALLER_DIR%\install_mkl.bat
call %INSTALLER_DIR%\install_magma.bat
call %INSTALLER_DIR%\install_sccache.bat
call %INSTALLER_DIR%\install_miniconda3.bat


:: Install ninja and other deps
if "%REBUILD%"=="" ( pip install -q "ninja==1.9.0" dataclasses )

git submodule sync --recursive
git submodule update --init --recursive

:: Override VS env here
pushd .
if "%VC_VERSION%" == "" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%VC_VERSION%
)
@echo on
popd

if "%CUDA_VERSION%" == "9" goto cuda_build_9
if "%CUDA_VERSION%" == "10" goto cuda_build_10
if "%CUDA_VERSION%" == "11" goto cuda_build_11
goto cuda_build_end

:cuda_build_9

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2
set CUDA_PATH_V9_2=%CUDA_PATH%

goto cuda_build_common

:cuda_build_10

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
set CUDA_PATH_V10_1=%CUDA_PATH%

goto cuda_build_common

:cuda_build_11

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0
set CUDA_PATH_V11_0=%CUDA_PATH%

goto cuda_build_common

:cuda_build_common

set CUDNN_LIB_DIR=%CUDA_PATH%\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
set CUDNN_ROOT_DIR=%CUDA_PATH%
set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

:cuda_build_end

set DISTUTILS_USE_SDK=1
set PATH=%TMP_DIR_WIN%\bin;%PATH%

:: Target only our CI GPU machine's CUDA arch to speed up the build, we can overwrite with env var
:: default on circleci is Tesla T4 which has capability of 7.5, ref: https://developer.nvidia.com/cuda-gpus
:: jenkins has M40, which is 5.2
if "%TORCH_CUDA_ARCH_LIST%" == "" set TORCH_CUDA_ARCH_LIST=5.2

:: The default sccache idle timeout is 600, which is too short and leads to intermittent build errors.
set SCCACHE_IDLE_TIMEOUT=0
sccache --stop-server
sccache --start-server
sccache --zero-stats
set CC=sccache-cl
set CXX=sccache-cl

set CMAKE_GENERATOR=Ninja

if "%USE_CUDA%"=="1" (
  copy %TMP_DIR_WIN%\bin\sccache.exe %TMP_DIR_WIN%\bin\nvcc.exe

  :: randomtemp is used to resolve the intermittent build error related to CUDA.
  :: code: https://github.com/peterjc123/randomtemp
  :: issue: https://github.com/pytorch/pytorch/issues/25393
  ::
  :: Previously, CMake uses CUDA_NVCC_EXECUTABLE for finding nvcc and then
  :: the calls are redirected to sccache. sccache looks for the actual nvcc
  :: in PATH, and then pass the arguments to it.
  :: Currently, randomtemp is placed before sccache (%TMP_DIR_WIN%\bin\nvcc)
  :: so we are actually pretending sccache instead of nvcc itself.
  curl -kL https://github.com/peterjc123/randomtemp/releases/download/v0.3/randomtemp.exe --output %TMP_DIR_WIN%\bin\randomtemp.exe
  set RANDOMTEMP_EXECUTABLE=%TMP_DIR_WIN%\bin\nvcc.exe
  set CUDA_NVCC_EXECUTABLE=%TMP_DIR_WIN%\bin\randomtemp.exe
  set RANDOMTEMP_BASEDIR=%TMP_DIR_WIN%\bin
)

@echo off
echo @echo off >> %TMP_DIR_WIN%\ci_scripts\pytorch_env_restore.bat
for /f "usebackq tokens=*" %%i in (`set`) do echo set "%%i" >> %TMP_DIR_WIN%\ci_scripts\pytorch_env_restore.bat
@echo on

if "%REBUILD%" == "" (
  if NOT "%BUILD_ENVIRONMENT%" == "" (
    :: Create a shortcut to restore pytorch environment
    echo @echo off >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore_helper.bat
    echo call "%TMP_DIR_WIN%/ci_scripts/pytorch_env_restore.bat" >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore_helper.bat
    echo cd /D "%CD%" >> %TMP_DIR_WIN%/ci_scripts/pytorch_env_restore_helper.bat

    aws s3 cp "s3://ossci-windows/Restore PyTorch Environment.lnk" "C:\Users\circleci\Desktop\Restore PyTorch Environment.lnk"
  )
)

python setup.py install --cmake && sccache --show-stats && (
  if "%BUILD_ENVIRONMENT%"=="" (
    echo NOTE: To run `import torch`, please make sure to activate the conda environment by running `call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3` in Command Prompt before running Git Bash.
  ) else (
    7z a %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\caffe2 && copy /Y "%TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z" "%PYTORCH_FINAL_PACKAGE_DIR%\"
  )
)

