#!/bin/bash

# If you want to rebuild, run this with REBUILD=1
# If you want to build with CUDA, run this with USE_CUDA=1
# If you want to build without CUDA, run this with USE_CUDA=0

if [ ! -f setup.py ]; then
  echo "ERROR: Please run this build script from PyTorch root directory."
  exit 1
fi

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

set PATH=C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Amazon\\AWSCLI;%PATH%

:: Install MKL
if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl -k https://s3.amazonaws.com/ossci-windows/mkl_2018.2.185.7z --output mkl.7z
  ) else (
    aws s3 cp s3://ossci-windows/mkl_2018.2.185.7z mkl.7z --quiet
  )
  7z x -aoa mkl.7z -omkl
)
set CMAKE_INCLUDE_PATH=%cd%\\mkl\\include
set LIB=%cd%\\mkl\\lib;%LIB

:: Install MAGMA
if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl -k https://s3.amazonaws.com/ossci-windows/magma_cuda90_release_mkl_2018.2.185.7z --output magma_cuda90_release_mkl_2018.2.185.7z
  ) else (
    aws s3 cp s3://ossci-windows/magma_cuda90_release_mkl_2018.2.185.7z magma_cuda90_release_mkl_2018.2.185.7z --quiet
  )
  7z x -aoa magma_cuda90_release_mkl_2018.2.185.7z -omagma
)
set MAGMA_HOME=%cd%\\magma

:: Install sccache
mkdir %CD%\\tmp_bin
if "%REBUILD%"=="" (
  :check_sccache
  %CD%\\tmp_bin\\sccache.exe --show-stats || (
    taskkill /im sccache.exe /f /t || ver > nul
    del %CD%\\tmp_bin\\sccache.exe
    if "%BUILD_ENVIRONMENT%"=="" (
      curl -k https://s3.amazonaws.com/ossci-windows/sccache.exe --output %CD%\\tmp_bin\\sccache.exe
    ) else (
      aws s3 cp s3://ossci-windows/sccache.exe %CD%\\tmp_bin\\sccache.exe
    )
    goto :check_sccache
  )
)

:: Install Miniconda3
if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\\Jenkins
)
if "%REBUILD%"=="" (
  IF EXIST %CONDA_PARENT_DIR%\\Miniconda3 ( rd /s /q %CONDA_PARENT_DIR%\\Miniconda3 )
  curl -k https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -O
  .\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_PARENT_DIR%\\Miniconda3
)
call %CONDA_PARENT_DIR%\\Miniconda3\\Scripts\\activate.bat %CONDA_PARENT_DIR%\\Miniconda3
if "%REBUILD%"=="" (
  :: We have to pin Python version to 3.6.7, until mkl supports Python 3.7
  call conda install -y -q python=3.6.7 numpy cffi pyyaml boto3
)

:: Install ninja
if "%REBUILD%"=="" ( pip install ninja )

call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64
call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x86_amd64

git submodule update --init --recursive

set PATH=%CD%\\tmp_bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp;%PATH%
set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDA_PATH_V9_0=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set NVTOOLSEXT_PATH=C:\\Program Files\\NVIDIA Corporation\\NvToolsExt
set CUDNN_LIB_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64
set CUDA_TOOLKIT_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0
set CUDNN_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0

:: Target only our CI GPU machine's CUDA arch to speed up the build
set TORCH_CUDA_ARCH_LIST=5.2

sccache --stop-server
sccache --start-server
sccache --zero-stats
set CC=sccache cl
set CXX=sccache cl

set DISTUTILS_USE_SDK=1

set CMAKE_GENERATOR=Ninja

if not "%USE_CUDA%"=="1" (
  if "%REBUILD%"=="" (
    set NO_CUDA=1
    python setup.py install
  )
  if errorlevel 1 exit /b 1
  if not errorlevel 0 exit /b 1
)

if not "%USE_CUDA%"=="0" (
  if "%REBUILD%"=="" (
    sccache --show-stats
    sccache --zero-stats
    rd /s /q %CONDA_PARENT_DIR%\\Miniconda3\\Lib\\site-packages\\torch
    copy %CD%\\tmp_bin\\sccache.exe tmp_bin\\nvcc.exe
  )

  set CUDA_NVCC_EXECUTABLE=%CD%\\tmp_bin\\nvcc

  if "%REBUILD%"=="" set NO_CUDA=0

  python setup.py install && sccache --show-stats && (
    if "%BUILD_ENVIRONMENT%"=="" (
      echo NOTE: To run \`import torch\`, please make sure to activate the conda environment by running \`call %CONDA_PARENT_DIR%\\Miniconda3\\Scripts\\activate.bat %CONDA_PARENT_DIR%\\Miniconda3\` in Command Prompt before running Git Bash.
    ) else (
      mv %CD%\\build\\bin\\test_api.exe %CONDA_PARENT_DIR%\\Miniconda3\\Lib\\site-packages\\torch\\lib
      7z a %IMAGE_COMMIT_TAG%.7z %CONDA_PARENT_DIR%\\Miniconda3\\Lib\\site-packages\\torch && python ci_scripts\\upload_image.py %IMAGE_COMMIT_TAG%.7z
    )
  )
)

EOL

ci_scripts/build_pytorch.bat
if [ ! -f $IMAGE_COMMIT_TAG.7z ] && [ ! ${BUILD_ENVIRONMENT} == "" ]; then
    exit 1
fi
echo "BUILD PASSED"
