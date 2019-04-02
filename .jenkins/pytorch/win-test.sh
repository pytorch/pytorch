#!/bin/bash -e

COMPACT_JOB_NAME=pytorch-win-ws2016-cuda9-cudnn7-py3-test

SCRIPT_PARENT_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$SCRIPT_PARENT_DIR/common.sh"

export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

export TMP_DIR="${PWD}/build/win_tmp"
export TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
mkdir -p $TMP_DIR/ci_scripts/
mkdir -p $TMP_DIR/build/torch

if [ ! -z "$(ls $TMP_DIR/ci_scripts/*)" ]; then
    rm $TMP_DIR/ci_scripts/*
fi


cat >$TMP_DIR/ci_scripts/setup_pytorch_env.bat <<EOL

if exist "%TMP_DIR%/ci_scripts/pytorch_env_restore.bat" (
    call %TMP_DIR%/ci_scripts/pytorch_env_restore.bat
    exit /b 0
)

set PATH=C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Amazon\\AWSCLI;%PATH%

:: Install Miniconda3
if "%BUILD_ENVIRONMENT%"=="" (
    set CONDA_PARENT_DIR=%CD%
) else (
    set CONDA_PARENT_DIR=C:\\Jenkins
)
if NOT "%BUILD_ENVIRONMENT%"=="" (
    IF EXIST %CONDA_PARENT_DIR%\\Miniconda3 ( rd /s /q %CONDA_PARENT_DIR%\\Miniconda3 )
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe --output %TMP_DIR_WIN%\\Miniconda3-latest-Windows-x86_64.exe
    %TMP_DIR_WIN%\\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_PARENT_DIR%\\Miniconda3
)
call %CONDA_PARENT_DIR%\\Miniconda3\\Scripts\\activate.bat %CONDA_PARENT_DIR%\\Miniconda3
if NOT "%BUILD_ENVIRONMENT%"=="" (
    :: We have to pin Python version to 3.6.7, until mkl supports Python 3.7
    call conda install -y -q python=3.6.7 numpy mkl cffi pyyaml boto3 protobuf numba
)
pip install -q ninja future hypothesis "librosa>=0.6.2" psutil

set WORKING_DIR=%CD%
call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x86_amd64
cd %WORKING_DIR%

set PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp;%PATH%
set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDA_PATH_V9_0=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set NVTOOLSEXT_PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt
set CUDNN_LIB_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDNN_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set PYTHONPATH=%TMP_DIR_WIN%\\build;%PYTHONPATH%
set NUMBAPRO_CUDALIB=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin
set NUMBAPRO_LIBDEVICE=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\nvvm\\libdevice
set NUMBAPRO_NVVM=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\nvvm\\bin\\nvvm64_32_0.dll

if NOT "%BUILD_ENVIRONMENT%"=="" (
    cd %TMP_DIR_WIN%\\build
    python %TMP_DIR_WIN%\\ci_scripts\\download_image.py %TMP_DIR_WIN%\\%IMAGE_COMMIT_TAG%.7z
    :: 7z: -aos skips if exists because this .bat can be called multiple times
    7z x %TMP_DIR_WIN%\\%IMAGE_COMMIT_TAG%.7z -aos
    cd %WORKING_DIR%
) else (
    xcopy /s %CONDA_PARENT_DIR%\\Miniconda3\\Lib\\site-packages\\torch %TMP_DIR_WIN%\\build\\torch\\
)

for /f "usebackq tokens=*" %%i in (\`set\`) do echo set "%%i" >> %TMP_DIR%/ci_scripts/pytorch_env_restore.bat

EOL

cat >$TMP_DIR/ci_scripts/test_python_nn.bat <<EOL
call %TMP_DIR%/ci_scripts/setup_pytorch_env.bat
:: Some smoke tests
cd test/
:: Checking that caffe2.python is available
python -c "from caffe2.python import core"
if ERRORLEVEL 1 exit /b 1
:: Checking that MKL is available
python -c "import torch; exit(0 if torch.backends.mkl.is_available() else 1)"
if ERRORLEVEL 1 exit /b 1
:: Checking that CUDA archs are setup correctly
python -c "import torch; torch.randn([3,5]).cuda()"
if ERRORLEVEL 1 exit /b 1
:: Checking that magma is available
python -c "import torch; torch.rand(1).cuda(); exit(0 if torch.cuda.has_magma else 1)"
if ERRORLEVEL 1 exit /b 1
:: Checking that CuDNN is available
python -c "import torch; exit(0 if torch.backends.cudnn.is_available() else 1)"
if ERRORLEVEL 1 exit /b 1
cd ..

:: Run nn tests
cd test/ && python run_test.py --include nn --verbose && cd ..
EOL

cat >$TMP_DIR/ci_scripts/test_python_all_except_nn.bat <<EOL
call %TMP_DIR%/ci_scripts/setup_pytorch_env.bat
cd test/ && python run_test.py --exclude nn --verbose && cd ..
EOL

cat >$TMP_DIR/ci_scripts/test_custom_script_ops.bat <<EOL
call %TMP_DIR%/ci_scripts/setup_pytorch_env.bat

cd test/custom_operator

:: Build the custom operator library.
mkdir build
cd build
:: Note: Caffe2 does not support MSVC + CUDA + Debug mode (has to be Release mode)
cmake -DCMAKE_PREFIX_PATH=%TMP_DIR_WIN%\\build\\torch -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
cd ..

:: Run tests Python-side and export a script module.
python test_custom_ops.py -v
python model.py --export-script-module="build/model.pt"
:: Run tests C++-side and load the exported script module.
cd build
set PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt/bin/x64;%TMP_DIR_WIN%\\build\\torch\\lib;%PATH%
test_custom_ops.exe model.pt
EOL

cat >$TMP_DIR/ci_scripts/test_libtorch.bat <<EOL
call %TMP_DIR%/ci_scripts/setup_pytorch_env.bat
dir
dir %TMP_DIR_WIN%\\build
dir %TMP_DIR_WIN%\\build\\torch
dir %TMP_DIR_WIN%\\build\\torch\\lib
cd %TMP_DIR_WIN%\\build\\torch\\lib
set PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt/bin/x64;%TMP_DIR_WIN%\\build\\torch\\lib;%PATH%
test_api.exe --gtest_filter="-IntegrationTest.MNIST*"
EOL



SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers
CI_SCRIPTS_DIR=$TMP_DIR/ci_scripts

# Used by setup_pytorch_env.bat:
cp $SCRIPT_HELPERS_DIR/download_image.py $CI_SCRIPTS_DIR


run_tests() {
    if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
        $TMP_DIR/ci_scripts/test_python_nn.bat && \
        $TMP_DIR/ci_scripts/test_python_all_except_nn.bat && \
        $TMP_DIR/ci_scripts/test_custom_script_ops.bat && \
        $TMP_DIR/ci_scripts/test_libtorch.bat
    else
        if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
            $TMP_DIR/ci_scripts/test_python_nn.bat
        elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
            $TMP_DIR/ci_scripts/test_python_all_except_nn.bat && \
            $TMP_DIR/ci_scripts/test_custom_script_ops.bat && \
            $TMP_DIR/ci_scripts/test_libtorch.bat
        fi
    fi
}

run_tests && assert_git_not_dirty && echo "TEST PASSED"
