#!/bin/bash

set -ex

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

cat >ci_scripts/delete_image.py << EOL

import os
import boto3

IMAGE_COMMIT_TAG = os.getenv('IMAGE_COMMIT_TAG')

session = boto3.session.Session()
s3 = session.resource('s3')
BUCKET_NAME = 'ossci-windows-build'
KEY = 'pytorch/'+IMAGE_COMMIT_TAG+'.7z'
s3.Object(BUCKET_NAME, KEY).delete()

EOL

cat >ci_scripts/test_pytorch.bat <<EOL

set PATH=C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;C:\\curl-7.57.0-win64-mingw\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Amazon\\AWSCLI;%PATH%

:: Install Miniconda3
rm -rf C:\\Jenkins\\Miniconda3
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -O
.\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=C:\\Jenkins\\Miniconda3
call C:\\Jenkins\\Miniconda3\\Scripts\\activate.bat C:\\Jenkins\\Miniconda3
call conda install -y -q numpy mkl cffi pyyaml boto3

set PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp;%PATH%
set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDA_PATH_V9_0=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set NVTOOLSEXT_PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt
set CUDNN_LIB_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDNN_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0

cd test/

python ..\\ci_scripts\\download_image.py %IMAGE_COMMIT_TAG%.7z
python ..\\ci_scripts\\delete_image.py

7z x %IMAGE_COMMIT_TAG%.7z
sh run_test.sh

EOL

echo "ENTERED_USER_LAND" && ci_scripts/test_pytorch.bat && echo "EXITED_USER_LAND" && echo "TEST PASSED"
