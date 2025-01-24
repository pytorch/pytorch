@echo off

:: This script parses args, installs required libraries (miniconda, MKL,
:: Magma), and then delegates to cpu.bat, cuda80.bat, etc.

if not "%CUDA_VERSION%" == "" if not "%PYTORCH_BUILD_VERSION%" == "" if not "%PYTORCH_BUILD_NUMBER%" == "" goto env_end
if "%~1"=="" goto arg_error
if "%~2"=="" goto arg_error
if "%~3"=="" goto arg_error
if not "%~4"=="" goto arg_error
goto arg_end

:arg_error

echo Illegal number of parameters. Pass cuda version, pytorch version, build number
echo CUDA version should be Mm with no dot, e.g. '80'
echo DESIRED_PYTHON should be M.m, e.g. '2.7'
exit /b 1

:arg_end

set CUDA_VERSION=%~1
set PYTORCH_BUILD_VERSION=%~2
set PYTORCH_BUILD_NUMBER=%~3

:env_end

set CUDA_PREFIX=cuda%CUDA_VERSION%
if "%CUDA_VERSION%" == "cpu" set CUDA_PREFIX=cpu
if "%CUDA_VERSION%" == "xpu" set CUDA_PREFIX=xpu

if "%DESIRED_PYTHON%" == "" set DESIRED_PYTHON=3.5;3.6;3.7
set DESIRED_PYTHON_PREFIX=%DESIRED_PYTHON:.=%
set DESIRED_PYTHON_PREFIX=py%DESIRED_PYTHON_PREFIX:;=;py%

set SRC_DIR=%~dp0
pushd %SRC_DIR%

:: Install Miniconda3
set "CONDA_HOME=%CD%\conda"
set "tmp_conda=%CONDA_HOME%"
set "miniconda_exe=%CD%\miniconda.exe"
rmdir /s /q conda
del miniconda.exe
curl --retry 3 -k https://repo.anaconda.com/miniconda/Miniconda3-py311_23.9.0-0-Windows-x86_64.exe -o "%miniconda_exe%"
start /wait "" "%miniconda_exe%" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D=%tmp_conda%
if ERRORLEVEL 1 exit /b 1
set "ORIG_PATH=%PATH%"
set "PATH=%CONDA_HOME%;%CONDA_HOME%\scripts;%CONDA_HOME%\Library\bin;%PATH%"

:: create a new conda environment and install packages
:try
SET /A tries=3
:loop
IF %tries% LEQ 0 GOTO :exception
call condaenv.bat
IF %ERRORLEVEL% EQU 0 GOTO :done
SET /A "tries=%tries%-1"
:exception
echo "Failed to create conda env"
exit /B 1
:done

:: Download MAGMA Files on CUDA builds
set MAGMA_VERSION=2.5.4

if "%DEBUG%" == "1" (
    set BUILD_TYPE=debug
) else (
    set BUILD_TYPE=release
)

if not "%CUDA_VERSION%" == "cpu" if not "%CUDA_VERSION%" == "xpu" (
    rmdir /s /q magma_%CUDA_PREFIX%_%BUILD_TYPE%
    del magma_%CUDA_PREFIX%_%BUILD_TYPE%.7z
    curl -k https://s3.amazonaws.com/ossci-windows/magma_%MAGMA_VERSION%_%CUDA_PREFIX%_%BUILD_TYPE%.7z -o magma_%CUDA_PREFIX%_%BUILD_TYPE%.7z
    7z x -aoa magma_%CUDA_PREFIX%_%BUILD_TYPE%.7z -omagma_%CUDA_PREFIX%_%BUILD_TYPE%
)

:: Install sccache
if "%USE_SCCACHE%" == "1" (
    mkdir %CD%\tmp_bin
    curl -k https://s3.amazonaws.com/ossci-windows/sccache.exe --output %CD%\tmp_bin\sccache.exe
    curl -k https://s3.amazonaws.com/ossci-windows/sccache-cl.exe --output %CD%\tmp_bin\sccache-cl.exe
    if not "%CUDA_VERSION%" == "" (
        set ADDITIONAL_PATH=%CD%\tmp_bin
        set SCCACHE_IDLE_TIMEOUT=1500

        :: randomtemp is used to resolve the intermittent build error related to CUDA.
        :: code: https://github.com/peterjc123/randomtemp-rust
        :: issue: https://github.com/pytorch/pytorch/issues/25393
        ::
        :: CMake requires a single command as CUDA_NVCC_EXECUTABLE, so we push the wrappers
        :: randomtemp.exe and sccache.exe into a batch file which CMake invokes.
        curl -kL https://github.com/peterjc123/randomtemp-rust/releases/download/v0.4/randomtemp.exe --output %SRC_DIR%\tmp_bin\randomtemp.exe
        echo @"%SRC_DIR%\tmp_bin\randomtemp.exe" "%SRC_DIR%\tmp_bin\sccache.exe" "%CUDA_PATH%\bin\nvcc.exe" %%* > "%SRC_DIR%/tmp_bin/nvcc.bat"
        cat %SRC_DIR%/tmp_bin/nvcc.bat
        set CUDA_NVCC_EXECUTABLE=%SRC_DIR%/tmp_bin/nvcc.bat
        :: CMake doesn't accept back-slashes in the path
        for /F "usebackq delims=" %%n in (`cygpath -m "%CUDA_PATH%\bin\nvcc.exe"`) do set CMAKE_CUDA_COMPILER=%%n
        set CMAKE_CUDA_COMPILER_LAUNCHER=%SRC_DIR%\tmp_bin\randomtemp.exe;%SRC_DIR%\tmp_bin\sccache.exe
    )
)

set PYTORCH_BINARY_BUILD=1
set TH_BINARY_BUILD=1
set INSTALL_TEST=0

for %%v in (%DESIRED_PYTHON_PREFIX%) do (
    :: Activate Python Environment
    set PYTHON_PREFIX=%%v
    set "CONDA_LIB_PATH=%CONDA_HOME%\envs\%%v\Library\bin"
    if not "%ADDITIONAL_PATH%" == "" (
        set "PATH=%ADDITIONAL_PATH%;%CONDA_HOME%\envs\%%v;%CONDA_HOME%\envs\%%v\scripts;%CONDA_HOME%\envs\%%v\Library\bin;%ORIG_PATH%"
    ) else (
        set "PATH=%CONDA_HOME%\envs\%%v;%CONDA_HOME%\envs\%%v\scripts;%CONDA_HOME%\envs\%%v\Library\bin;%ORIG_PATH%"
    )
    pip install ninja
    @setlocal
    :: Set Flags
    if not "%CUDA_VERSION%"=="cpu" if not "%CUDA_VERSION%" == "xpu" (
        set MAGMA_HOME=%cd%\magma_%CUDA_PREFIX%_%BUILD_TYPE%
    )
    echo "Calling arch build script"
    call %CUDA_PREFIX%.bat
    if ERRORLEVEL 1 exit /b 1
    @endlocal
)

set "PATH=%ORIG_PATH%"
popd

if ERRORLEVEL 1 exit /b 1
