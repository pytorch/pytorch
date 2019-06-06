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

if NOT DEFINED BUILD_SHARED_LIBS (
  set BUILD_SHARED_LIBS=OFF
)

IF NOT DEFINED BUILDING_WITH_TORCH_LIBS (
  set BUILDING_WITH_TORCH_LIBS=OFF
)

if NOT DEFINED CAFFE2_STATIC_LINK_CUDA (
  set CAFFE2_STATIC_LINK_CUDA=OFF
)

if NOT DEFINED CMAKE_BUILD_TYPE (
  set CMAKE_BUILD_TYPE=Release
)

if NOT DEFINED ONNX_NAMESPACE (
  set ONNX_NAMESPACE=onnx_c2
)

if NOT DEFINED TORCH_CUDA_ARCH_LIST (
  set TORCH_CUDA_ARCH_LIST=5.0
)

if NOT DEFINED USE_CUDA (
  set USE_CUDA=OFF
)

if NOT DEFINED USE_OBSERVERS (
  set USE_OBSERVERS=OFF
)

if NOT DEFINED MSVC_Z7_OVERRIDE (
  set MSVC_Z7_OVERRIDE=OFF
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

:: Set up cmake. We will skip building the test files right now.
pushd %CAFFE2_ROOT%
python tools\build_libtorch.py || goto :label_error
popd

echo "Caffe2 built successfully"
cd %ORIGINAL_DIR%
endlocal
exit /b 0

:label_error
echo "Caffe2 building failed"
cd %ORIGINAL_DIR%
endlocal
exit /b 1
