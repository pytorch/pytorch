:: #############################################################################
:: Example command to build on Windows.
:: #############################################################################

:: This script shows how one can build a Caffe2 binary for windows.

@echo on

SET ORIGINAL_DIR=%cd%
SET CAFFE2_ROOT=%~dp0%..
if not exist %CAFFE2_ROOT%\build_host_protoc\bin\protoc.exe call %CAFFE2_ROOT%\scripts\build_host_protoc.bat

if not exist %CAFFE2_ROOT%\build mkdir %CAFFE2_ROOT%\build
cd %CAFFE2_ROOT%\build

if NOT DEFINED USE_CUDA (
  set USE_CUDA=ON
)

if NOT DEFINED CMAKE_BUILD_TYPE (
  set CMAKE_BUILD_TYPE=Release
)

:: Set up cmake. We will skip building the test files right now.
:: TODO: enable cuda support.
cmake .. ^
  -G"Visual Studio 14 2015 Win64" ^
  -DCMAKE_VERBOSE_MAKEFILE=1 ^
  -DBUILD_TEST=OFF ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% ^
  -DUSE_CUDA=%USE_CUDA% ^
  -DPROTOBUF_PROTOC_EXECUTABLE=%CAFFE2_ROOT%\build_host_protoc\bin\protoc.exe ^
  || exit /b

:: Actually run the build
cmake --build . --config %CMAKE_BUILD_TYPE% || exit /b

cd %ORIGINAL_DIR%
