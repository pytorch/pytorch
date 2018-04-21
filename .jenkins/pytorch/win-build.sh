#!/bin/bash

COMPACT_JOB_NAME=pytorch-win-ws2016-cuda9-cudnn7-py3-build
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

mkdir -p ci_scripts/

cat >ci_scripts/upload_image.py << EOL

import os
import sys
import boto3

IMAGE_COMMIT_TAG = os.getenv('IMAGE_COMMIT_TAG')

session = boto3.session.Session()
s3 = session.resource('s3')
data = open(sys.argv[1], 'rb')
s3.Bucket('ossci-windows-build').put_object(Key='pytorch/'+IMAGE_COMMIT_TAG+'.7z', Body=data)
object_acl = s3.ObjectAcl('ossci-windows-build','pytorch/'+IMAGE_COMMIT_TAG+'.7z')
response = object_acl.put(ACL='public-read')

EOL

cat >ci_scripts/build_pytorch.bat <<EOL

set PATH=C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;C:\\curl-7.57.0-win64-mingw\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Amazon\\AWSCLI;%PATH%

:: Install MKL
aws s3 cp s3://ossci-windows/mkl_2018.2.185.7z mkl.7z --quiet && 7z x -aoa mkl.7z -omkl
set CMAKE_INCLUDE_PATH=%cd%\\mkl\\include
set LIB=%cd%\\mkl\\lib;%LIB

:: Install MAGMA
aws s3 cp s3://ossci-windows/magma_cuda90_release_mkl_2018.2.185.7z magma_cuda90_release_mkl_2018.2.185.7z --quiet && 7z x -aoa magma_cuda90_release_mkl_2018.2.185.7z -omagma
set MAGMA_HOME=%cd%\\magma

:: Install clcache
aws s3 cp s3://ossci-windows/clcache.7z clcache.7z --quiet && 7z x -aoa clcache.7z -oclcache

:: Install Miniconda3
IF EXIST C:\\Jenkins\\Miniconda3 ( rd /s /q C:\\Jenkins\\Miniconda3 )
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -O
.\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=C:\\Jenkins\\Miniconda3
call C:\\Jenkins\\Miniconda3\\Scripts\\activate.bat C:\\Jenkins\\Miniconda3
call conda install -y -q numpy cffi pyyaml boto3

:: Install ninja
pip install ninja

call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x86_amd64

git submodule update --init --recursive

set PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp;%PATH%
set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDA_PATH_V9_0=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set NVTOOLSEXT_PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt
set CUDNN_LIB_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDNN_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0

set TORCH_CUDA_ARCH_LIST=5.2

set USE_CLCACHE=1
set CLCACHE_DIR=%cd%\\clcache_tmp
set CC=%cd%\\clcache\\clcache_main.exe
set CXX=%cd%\\clcache\\clcache_main.exe

set DISTUTILS_USE_SDK=1

set CMAKE_GENERATOR=Ninja

set NO_CUDA=1

python setup.py install

if %errorlevel% neq 0 exit /b %errorlevel%

rd /s /q C:\\Jenkins\\Miniconda3\\Lib\\site-packages\\torch

set NO_CUDA=

python setup.py install && 7z a %IMAGE_COMMIT_TAG%.7z C:\\Jenkins\\Miniconda3\\Lib\\site-packages\\torch && python ci_scripts\\upload_image.py %IMAGE_COMMIT_TAG%.7z

EOL

ci_scripts/build_pytorch.bat && echo "BUILD PASSED"
