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
IF EXIST C:\\Jenkins\\Miniconda3 ( rd /s /q C:\\Jenkins\\Miniconda3 )
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -O
.\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=C:\\Jenkins\\Miniconda3
call C:\\Jenkins\\Miniconda3\\Scripts\\activate.bat C:\\Jenkins\\Miniconda3
call conda install -y -q numpy mkl cffi pyyaml boto3

pip install ninja

call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x86_amd64

set PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp;%PATH%
set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDA_PATH_V9_0=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set NVTOOLSEXT_PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt
set CUDNN_LIB_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDNN_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set PYTHONPATH=%CD%\\test;%PYTHONPATH%

cd test/

python ..\\ci_scripts\\download_image.py %IMAGE_COMMIT_TAG%.7z

7z x %IMAGE_COMMIT_TAG%.7z

cd ..

EOL

cat >ci_scripts/test_python_nn.bat <<EOL
call ci_scripts/setup_pytorch_env.bat
cd test/ && python run_test.py --include nn --verbose && cd ..
EOL

cat >ci_scripts/test_python_all_except_nn.bat <<EOL
call ci_scripts/setup_pytorch_env.bat
cd test/ && python run_test.py --exclude nn --verbose && cd ..
EOL

run_tests() {
    if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
        ci_scripts/test_python_nn.bat && ci_scripts/test_python_all_except_nn.bat
    else
        if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
            ci_scripts/test_python_nn.bat
        elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
            ci_scripts/test_python_all_except_nn.bat
        fi
    fi
}

run_tests && echo "TEST PASSED"
