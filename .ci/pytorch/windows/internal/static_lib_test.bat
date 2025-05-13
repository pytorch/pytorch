set SRC_DIR=%~dp0

pushd %SRC_DIR%\..

if "%CUDA_VERSION%" == "cpu" call internal\driver_update.bat
if errorlevel 1 exit /b 1

call internal\cuda_install.bat
set LIB=%CUDA_PATH%\lib\x64;%LIB%
if errorlevel 1 exit /b 1
set "ORIG_PATH=%PATH%"

setlocal EnableDelayedExpansion
set NVIDIA_GPU_EXISTS=0
for /F "delims=" %%i in ('wmic path win32_VideoController get name') do (
    set GPUS=%%i
    if not "x!GPUS:NVIDIA=!" == "x!GPUS!" (
        SET NVIDIA_GPU_EXISTS=1
        goto gpu_check_end
    )
)
:gpu_check_end
endlocal & set NVIDIA_GPU_EXISTS=%NVIDIA_GPU_EXISTS%

:: Download MAGMA Files on CUDA builds
set MAGMA_VERSION=2.5.4
set CUDA_PREFIX=cuda%CUDA_VERSION%
if "%CUDA_VERSION%" == "92" set MAGMA_VERSION=2.5.2
if "%CUDA_VERSION%" == "100" set MAGMA_VERSION=2.5.2

if "%DEBUG%" == "1" (
    set BUILD_TYPE=debug
) else (
    set BUILD_TYPE=release
)

if not "%CUDA_VERSION%" == "cpu" (
    rmdir /s /q magma_%CUDA_PREFIX%_%BUILD_TYPE%
    del magma_%CUDA_PREFIX%_%BUILD_TYPE%.7z
    curl -k https://s3.amazonaws.com/ossci-windows/magma_%MAGMA_VERSION%_%CUDA_PREFIX%_%BUILD_TYPE%.7z -o magma_%CUDA_PREFIX%_%BUILD_TYPE%.7z
    7z x -aoa magma_%CUDA_PREFIX%_%BUILD_TYPE%.7z -omagma_%CUDA_PREFIX%_%BUILD_TYPE%
    set LIB=%CD%\magma_%CUDA_PREFIX%_%BUILD_TYPE%\lib;%LIB%
)

echo "install conda package"

:: Install Miniconda3
set "CONDA_HOME=%CD%\conda"
set "tmp_conda=%CONDA_HOME%"
set "miniconda_exe=%CD%\miniconda.exe"

rmdir /s /q conda
del miniconda.exe
curl -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o "%miniconda_exe%"
start /wait "" "%miniconda_exe%" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D=%tmp_conda%
if ERRORLEVEL 1 exit /b 1

set "PATH=%CONDA_HOME%;%CONDA_HOME%\scripts;%CONDA_HOME%\Library\bin;%PATH%"

conda create -qyn testenv python=%DESIRED_PYTHON%
if errorlevel 1 exit /b 1

call %CONDA_HOME%\condabin\activate.bat testenv
if errorlevel 1 exit /b 1

call conda install  -y -q -c conda-forge libuv=1.39
call conda install -y -q intel-openmp

echo "install and test libtorch"
pip install cmake
echo "installing cmake"

if "%VC_YEAR%" == "2019" powershell internal\vs2019_install.ps1
if "%VC_YEAR%" == "2022" powershell internal\vs2022_install.ps1

if ERRORLEVEL 1 exit /b 1

for /F "delims=" %%i in ('where /R "%PYTORCH_FINAL_PACKAGE_DIR:/=\%" *-latest.zip') do 7z x "%%i" -otmp
if ERRORLEVEL 1 exit /b 1


pushd tmp\libtorch

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18
IF "%VC_YEAR%" == "2019" (
    set VC_VERSION_LOWER=16
    set VC_VERSION_UPPER=17
)

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
IF "%VS15VCVARSALL%"=="" (
    echo Visual Studio %VC_YEAR% C++ BuildTools is required to compile PyTorch test on Windows
    exit /b 1
)
call "%VS15VCVARSALL%" x64

set install_root=%CD%
set INCLUDE=%INCLUDE%;%install_root%\include;%install_root%\include\torch\csrc\api\include
set LIB=%LIB%;%install_root%\lib\x64
set PATH=%PATH%;%install_root%\lib


cd %PYTORCH_ROOT%\.ci\pytorch\test_example_code\
mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=%install_root% ..

if ERRORLEVEL 1 exit /b 1

cmake --build . --config Release

.\Release\simple-torch-test.exe
if ERRORLEVEL 1 exit /b 1

popd

echo Cleaning temp files
rd /s /q "tmp" || ver > nul

:end
set "PATH=%ORIG_PATH%"
popd
