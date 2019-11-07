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
  :: On CI, we test with BUILD_SHARED_LIBS=OFF.
  :: By default, it will be BUILD_SHARED_LIBS=ON.
  if NOT DEFINED BUILD_ENVIRONMENT (
    set BUILD_SHARED_LIBS=OFF
  )
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
  set CMAKE_GENERATOR=Ninja
)

set CMAKE_VERBOSE_MAKEFILE=1

:: Install pyyaml for Aten codegen
pip install pyyaml ninja

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
