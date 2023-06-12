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
:: log sizes are too long, but leaving this here incase someone wants to use it locally
:: set CMAKE_VERBOSE_MAKEFILE=1


set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers


call %INSTALLER_DIR%\install_mkl.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

call %INSTALLER_DIR%\install_magma.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

call %INSTALLER_DIR%\install_sccache.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

:: Miniconda has been installed as part of the Windows AMI with all the dependencies.
:: We just need to activate it here
call %INSTALLER_DIR%\activate_miniconda3.bat
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

:: Override VS env here
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

if not "%USE_CUDA%"=="1" goto cuda_build_end

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%

if x%CUDA_VERSION:.=%==x%CUDA_VERSION% (
    echo CUDA version %CUDA_VERSION% format isn't correct, which doesn't contain '.'
    exit /b 1
)
rem version transformer, for example 10.1 to 10_1.
if x%CUDA_VERSION:.=%==x%CUDA_VERSION% (
    echo CUDA version %CUDA_VERSION% format isn't correct, which doesn't contain '.'
    exit /b 1
)
set VERSION_SUFFIX=%CUDA_VERSION:.=_%
set CUDA_PATH_V%VERSION_SUFFIX%=%CUDA_PATH%

set CUDNN_LIB_DIR=%CUDA_PATH%\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
set CUDNN_ROOT_DIR=%CUDA_PATH%
set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

set CUDNN_LIB_DIR=%CUDA_PATH%\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
set CUDNN_ROOT_DIR=%CUDA_PATH%
set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

:cuda_build_end

set DISTUTILS_USE_SDK=1
set PATH=%TMP_DIR_WIN%\bin;%PATH%

:: The latest Windows CUDA test is running on AWS G5 runner with A10G GPU
if "%TORCH_CUDA_ARCH_LIST%" == "" set TORCH_CUDA_ARCH_LIST=8.6

:: The default sccache idle timeout is 600, which is too short and leads to intermittent build errors.
set SCCACHE_IDLE_TIMEOUT=0
set SCCACHE_IGNORE_SERVER_IO_ERROR=1
sccache --stop-server
sccache --start-server
sccache --zero-stats
set CC=sccache-cl
set CXX=sccache-cl

set CMAKE_GENERATOR=Ninja

if "%USE_CUDA%"=="1" (
  :: randomtemp is used to resolve the intermittent build error related to CUDA.
  :: code: https://github.com/peterjc123/randomtemp-rust
  :: issue: https://github.com/pytorch/pytorch/issues/25393
  ::
  :: CMake requires a single command as CUDA_NVCC_EXECUTABLE, so we push the wrappers
  :: randomtemp.exe and sccache.exe into a batch file which CMake invokes.
  curl -kL https://github.com/peterjc123/randomtemp-rust/releases/download/v0.4/randomtemp.exe --output %TMP_DIR_WIN%\bin\randomtemp.exe
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
  echo @"%TMP_DIR_WIN%\bin\randomtemp.exe" "%TMP_DIR_WIN%\bin\sccache.exe" "%CUDA_PATH%\bin\nvcc.exe" %%* > "%TMP_DIR%/bin/nvcc.bat"
  cat %TMP_DIR%/bin/nvcc.bat
  set CUDA_NVCC_EXECUTABLE=%TMP_DIR%/bin/nvcc.bat
  for /F "usebackq delims=" %%n in (`cygpath -m "%CUDA_PATH%\bin\nvcc.exe"`) do set CMAKE_CUDA_COMPILER=%%n
  set CMAKE_CUDA_COMPILER_LAUNCHER=%TMP_DIR%/bin/randomtemp.exe;%TMP_DIR%\bin\sccache.exe
)

:: Print all existing environment variable for debugging
set

python setup.py bdist_wheel
if errorlevel 1 exit /b
if not errorlevel 0 exit /b
sccache --show-stats
python -c "import os, glob; os.system('python -mpip install --no-index --no-deps ' + glob.glob('dist/*.whl')[0])"
(
  if "%BUILD_ENVIRONMENT%"=="" (
    echo NOTE: To run `import torch`, please make sure to activate the conda environment by running `call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3` in Command Prompt before running Git Bash.
  ) else (
    copy /Y "dist\*.whl" "%PYTORCH_FINAL_PACKAGE_DIR%"

    :: export test times so that potential sharded tests that'll branch off this build will use consistent data
    python tools/stats/export_test_times.py
    copy /Y ".pytorch-test-times.json" "%PYTORCH_FINAL_PACKAGE_DIR%"

    :: Also save build/.ninja_log as an artifact
    copy /Y "build\.ninja_log" "%PYTORCH_FINAL_PACKAGE_DIR%\"
  )
)

sccache --show-stats --stats-format json | jq .stats > sccache-stats-%BUILD_ENVIRONMENT%-%OUR_GITHUB_JOB_ID%.json
sccache --stop-server
