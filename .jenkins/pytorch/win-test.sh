#!/bin/bash

COMPACT_JOB_NAME=pytorch-win-ws2016-cuda9-cudnn7-py3-test
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

mkdir -p ci_scripts/

cat >ci_scripts/download_image.py << EOL

import os
import sys
import boto3
import botocore

IMAGE_COMMIT_TAG = os.getenv('IMAGE_COMMIT_TAG')

session = boto3.session.Session()
s3 = session.resource('s3')
BUCKET_NAME = 'ossci-windows-build'
KEY = 'pytorch/'+IMAGE_COMMIT_TAG+'.7z'
LOCAL_FILE_PATH = sys.argv[1]
try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, LOCAL_FILE_PATH)
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise

EOL

cat >ci_scripts/setup_pytorch_env.bat <<EOL

set PATH=C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Amazon\\AWSCLI;%PATH%

:: Install Miniconda3
if "%BUILD_ENVIRONMENT%"=="" (
    set CONDA_PARENT_DIR=%CD%
) else (
    set CONDA_PARENT_DIR=C:\\Jenkins
)
if NOT "%BUILD_ENVIRONMENT%"=="" (
    IF EXIST %CONDA_PARENT_DIR%\\Miniconda3 ( rd /s /q %CONDA_PARENT_DIR%\\Miniconda3 )
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -O
    .\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_PARENT_DIR%\\Miniconda3
)
call %CONDA_PARENT_DIR%\\Miniconda3\\Scripts\\activate.bat %CONDA_PARENT_DIR%\\Miniconda3
if NOT "%BUILD_ENVIRONMENT%"=="" (
    :: We have to pin Python version to 3.6.7, until mkl supports Python 3.7
    call conda install -y -q python=3.6.7 numpy mkl cffi pyyaml boto3
)
pip install ninja future hypothesis

call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x86_amd64

set PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp;%PATH%
set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDA_PATH_V9_0=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set NVTOOLSEXT_PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt
set CUDNN_LIB_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDNN_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set PYTHONPATH=%CD%\\test;%PYTHONPATH%

if NOT "%BUILD_ENVIRONMENT%"=="" (
    cd test/
    python ..\\ci_scripts\\download_image.py %IMAGE_COMMIT_TAG%.7z
    7z x %IMAGE_COMMIT_TAG%.7z
    cd ..
) else (
    xcopy /s %CONDA_PARENT_DIR%\\Miniconda3\\Lib\\site-packages\\torch .\\test\\torch\\
)

EOL

cat >ci_scripts/test_python_nn.bat <<EOL
call ci_scripts/setup_pytorch_env.bat
cd test/ && python run_test.py --include nn --verbose && cd ..
EOL

cat >ci_scripts/test_python_all_except_nn.bat <<EOL
call ci_scripts/setup_pytorch_env.bat
cd test/ && python run_test.py --exclude nn --verbose && cd ..
EOL

cat >ci_scripts/test_custom_script_ops.bat <<EOL
call ci_scripts/setup_pytorch_env.bat

cd test/custom_operator

:: Build the custom operator library.
mkdir build
cd build
:: Note: Caffe2 does not support MSVC + CUDA + Debug mode (has to be Release mode)
cmake -DCMAKE_PREFIX_PATH=%CD%\\..\\..\\torch -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
cd ..

:: Run tests Python-side and export a script module.
python test_custom_ops.py -v
python model.py --export-script-module="build/model.pt"
:: Run tests C++-side and load the exported script module.
cd build
set PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt/bin/x64;%CD%\\..\\..\\torch\\lib;%PATH%
test_custom_ops.exe model.pt
EOL

cat >ci_scripts/test_libtorch.bat <<EOL
call ci_scripts/setup_pytorch_env.bat
dir
dir %CD%\\test 
dir %CD%\\test\\torch
dir %CD%\\test\\torch\\lib
cd %CD%\\test\\torch\\lib
set PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt/bin/x64;%CD%\\..\\..\\torch\\lib;%PATH%
test_api.exe --gtest_filter="-IntegrationTest.MNIST*"
EOL

run_tests() {
    if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
        ci_scripts/test_python_nn.bat && ci_scripts/test_python_all_except_nn.bat && ci_scripts/test_custom_script_ops.bat && ci_scripts/test_libtorch.bat
    else
        if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
            ci_scripts/test_python_nn.bat
        elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
            ci_scripts/test_python_all_except_nn.bat && ci_scripts/test_custom_script_ops.bat && ci_scripts/test_libtorch.bat
        fi
    fi
}

run_tests && echo "TEST PASSED"
