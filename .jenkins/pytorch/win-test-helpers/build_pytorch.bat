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


:: Install ninja
if "%REBUILD%"=="" ( pip install -q ninja )

git submodule sync --recursive
git submodule update --init --recursive

if "%CUDA_VERSION%" == "9" goto cuda_build_9
if "%CUDA_VERSION%" == "10" goto cuda_build_10
goto cuda_build_end

:cuda_build_9

:: Override VS env here
pushd .
if "%VC_VERSION%" == "" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%VC_VERSION%
)
@echo on
popd

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2
set CUDA_PATH_V9_2=%CUDA_PATH%

goto cuda_build_common

:cuda_build_10
pushd .
if "%VC_VERSION%" == "" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    call "C:\Program Files (x86)\Microsoft Visual Studio\%VC_YEAR%\%VC_PRODUCT%\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%VC_VERSION%
)
@echo on
popd

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
set CUDA_PATH_V10_1=%CUDA_PATH%

goto cuda_build_common

:cuda_build_common

set DISTUTILS_USE_SDK=1

set CUDNN_LIB_DIR=%CUDA_PATH%\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
set CUDNN_ROOT_DIR=%CUDA_PATH%
set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

:cuda_build_end

set PATH=%TMP_DIR_WIN%\bin;%PATH%

:: Target only our CI GPU machine's CUDA arch to speed up the build, we can overwrite with env var
:: default on circleci is Tesla T4 which has capability of 7.5, ref: https://developer.nvidia.com/cuda-gpus
:: jenkins has M40, which is 5.2
if "%TORCH_CUDA_ARCH_LIST%" == "" set TORCH_CUDA_ARCH_LIST=5.2

sccache --stop-server
sccache --start-server
sccache --zero-stats
set CC=sccache-cl
set CXX=sccache-cl

set CMAKE_GENERATOR=Ninja

:: The following code will try to build PyTorch twice if USE_CUDA is neither 0
:: nor 1. It is intended so that both builds can be folded into 1 CI run.

if not "%USE_CUDA%"=="1" (
  if "%REBUILD%"=="" (
    :: Must save and restore the original value of USE_CUDA, otherwise the
    :: `if not "%USE_CUDA%"=="0"` line can be messed up.
    set OLD_USE_CUDA=%USE_CUDA%
    set USE_CUDA=0
    python setup.py install
    set USE_CUDA=%OLD_USE_CUDA%
  )
  if errorlevel 1 exit /b 1
  if not errorlevel 0 exit /b 1
)

if not "%USE_CUDA%"=="0" (
  :: sccache will fail for CUDA builds if all cores are used for compiling
  if not defined MAX_JOBS set /A MAX_JOBS=%NUMBER_OF_PROCESSORS%-1

  if "%REBUILD%"=="" (
    sccache --show-stats
    sccache --zero-stats
    rd /s /q %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch
    for /f "delims=" %%i in ('where /R caffe2\proto *.py') do (
      IF NOT "%%i" == "%CD%\caffe2\proto\__init__.py" (
        del /S /Q %%i
      )
    )
    copy %TMP_DIR_WIN%\bin\sccache.exe %TMP_DIR_WIN%\bin\nvcc.exe
  )

  :: randomtemp is used to resolve the intermittent build error related to CUDA.
  :: code: https://github.com/peterjc123/randomtemp
  :: issue: https://github.com/pytorch/pytorch/issues/25393
  ::
  :: Previously, CMake uses CUDA_NVCC_EXECUTABLE for finding nvcc and then
  :: the calls are redirected to sccache. sccache looks for the actual nvcc
  :: in PATH, and then pass the arguments to it.
  :: Currently, randomtemp is placed before sccache (%TMP_DIR_WIN%\bin\nvcc)
  :: so we are actually pretending sccache instead of nvcc itself.
  curl -kL https://github.com/peterjc123/randomtemp/releases/download/v0.2/randomtemp.exe --output %TMP_DIR_WIN%\bin\randomtemp.exe
  set RANDOMTEMP_EXECUTABLE=%TMP_DIR_WIN%\bin\nvcc.exe
  set CUDA_NVCC_EXECUTABLE=%TMP_DIR_WIN%\bin\randomtemp.exe
  set RANDOMTEMP_BASEDIR=%TMP_DIR_WIN%\bin

  if "%REBUILD%"=="" set USE_CUDA=1

  python setup.py install --cmake && sccache --show-stats && (
    if "%BUILD_ENVIRONMENT%"=="" (
      echo NOTE: To run `import torch`, please make sure to activate the conda environment by running `call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3` in Command Prompt before running Git Bash.
    ) else (
      7z a %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\caffe2 && python %SCRIPT_HELPERS_DIR%\upload_image.py %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z
    )
  )
)

