if exist "%TMP_DIR%/ci_scripts/pytorch_env_restore.bat" (
    call %TMP_DIR%/ci_scripts/pytorch_env_restore.bat
    exit /b 0
)

set PATH=C:\Program Files\CMake\bin;C:\Program Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;C:\Program Files\Amazon\AWSCLI;%PATH%

:: Install Miniconda3
if "%BUILD_ENVIRONMENT%"=="" (
    set CONDA_PARENT_DIR=%CD%
) else (
    set CONDA_PARENT_DIR=C:\Jenkins
)
if NOT "%BUILD_ENVIRONMENT%"=="" (
    IF EXIST %CONDA_PARENT_DIR%\Miniconda3 ( rd /s /q %CONDA_PARENT_DIR%\Miniconda3 )
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe --output %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe
    %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_PARENT_DIR%\Miniconda3
)
call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3
if NOT "%BUILD_ENVIRONMENT%"=="" (
    :: We have to pin Python version to 3.6.7, until mkl supports Python 3.7
    call conda install -y -q python=3.6.7 numpy mkl cffi pyyaml boto3 protobuf numba
)
pip install -q ninja future hypothesis "librosa>=0.6.2" psutil
:: No need to install faulthandler since we only test Python >= 3.6 on Windows
:: faulthandler is builtin since Python 3.3

pushd .
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x86_amd64
popd

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp;%PATH%
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set CUDA_PATH_V9_0=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt
set CUDNN_LIB_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set CUDNN_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set PYTHONPATH=%TMP_DIR_WIN%\build;%PYTHONPATH%
set NUMBAPRO_CUDALIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
set NUMBAPRO_LIBDEVICE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\nvvm\libdevice
set NUMBAPRO_NVVM=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\nvvm\bin\nvvm64_32_0.dll

if NOT "%BUILD_ENVIRONMENT%"=="" (
    pushd %TMP_DIR_WIN%\build
    python %SCRIPT_HELPERS_DIR%\download_image.py %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z
    :: 7z: -aos skips if exists because this .bat can be called multiple times
    7z x %TMP_DIR_WIN%\%IMAGE_COMMIT_TAG%.7z -aos
    popd
) else (
    xcopy /s %CONDA_PARENT_DIR%\Miniconda3\Lib\site-packages\torch %TMP_DIR_WIN%\build\torch\
)

for /f "usebackq tokens=*" %%i in (`set`) do echo set "%%i" >> %TMP_DIR%/ci_scripts/pytorch_env_restore.bat
