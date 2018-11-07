:: #############################################################################
:: Example command to build on Windows.
:: #############################################################################

:: This script shows how one can build a Caffe2 binary for windows.

@echo off
setlocal

SET ORIGINAL_DIR=%cd%
SET CAFFE2_ROOT=%~dp0%..

if NOT DEFINED BUILD_BINARY (
  set BUILD_BINARY=OFF
)

if NOT DEFINED CAFFE2_STATIC_LINK_CUDA (
  set CAFFE2_STATIC_LINK_CUDA=OFF
)

if NOT DEFINED CMAKE_BUILD_TYPE (
  set CMAKE_BUILD_TYPE=Release
)

if NOT DEFINED USE_CUDA (
  set USE_CUDA=OFF
)

if NOT DEFINED USE_OBSERVERS (
  set USE_OBSERVERS=OFF
)

if NOT DEFINED CMAKE_GENERATOR (
  if DEFINED APPVEYOR_BUILD_WORKER_IMAGE (
    if "%APPVEYOR_BUILD_WORKER_IMAGE%" == "Visual Studio 2017" (
      set CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
    ) else if "%APPVEYOR_BUILD_WORKER_IMAGE%" == "Visual Studio 2015" (
      set CMAKE_GENERATOR="Visual Studio 14 2015 Win64"
    ) else (
      echo "You made a programming error: unknown APPVEYOR_BUILD_WORKER_IMAGE:"
      echo %APPVEYOR_BUILD_WORKER_IMAGE%
      exit /b
    )
  ) else (
    :: In default we use win64 VS 2015.
    :: Main reason is that currently, cuda 9 does not support VS 2017 newest
    :: version. To use cuda you will have to use 2015.
    set CMAKE_GENERATOR="Visual Studio 15 2017 Win64"
  )
)

:: Install pyyaml for Aten codegen
pip install pyyaml

echo CAFFE2_ROOT=%CAFFE2_ROOT%
echo CMAKE_GENERATOR=%CMAKE_GENERATOR%
echo CMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%

if not exist %CAFFE2_ROOT%\build mkdir %CAFFE2_ROOT%\build
cd %CAFFE2_ROOT%\build

:: Set up cmake. We will skip building the test files right now.
:: TODO: enable cuda support.
cmake .. ^
  -G%CMAKE_GENERATOR% ^
  -DBUILD_TEST=OFF ^
  -DBUILD_BINARY=%BUILD_BINARY% ^
  -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% ^
  -DCAFFE2_STATIC_LINK_CUDA=%CAFFE2_STATIC_LINK_CUDA% ^
  -DUSE_CUDA=%USE_CUDA% ^
  -DTORCH_CUDA_ARCH_LIST=5.0 ^
  -DUSE_NNPACK=OFF ^
  -DUSE_CUB=OFF ^
  -DUSE_GLOG=OFF ^
  -DUSE_GFLAGS=OFF ^
  -DUSE_LMDB=OFF ^
  -DUSE_LEVELDB=OFF ^
  -DUSE_OBSERVERS=%USE_OBSERVERS%^
  -DUSE_ROCKSDB=OFF ^
  -DUSE_OPENCV=OFF ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DBUILD_PYTHON=OFF^
  || goto :label_error

:: Actually run the build
cmake --build . --config %CMAKE_BUILD_TYPE% -- /maxcpucount:%NUMBER_OF_PROCESSORS% || goto :label_error

echo "Caffe2 built successfully"
cd %ORIGINAL_DIR%
endlocal
exit /b 0

:label_error
echo "Caffe2 building failed"
cd %ORIGINAL_DIR%
endlocal
exit /b 1
