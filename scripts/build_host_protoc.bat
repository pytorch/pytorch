:: ###########################################################################
:: Build script to build the protoc compiler for the host platform.
:: ###########################################################################
:: This script builds the protoc compiler for Windows, which is needed
:: if protobuf is not installed already on windows, as we will need to convert
:: the protobuf source files to cc files.

:: After the execution of the file, one should be able to find the host protoc
:: binary at build_host_protoc/bin/protoc.exe.

@echo off
setlocal

SET ORIGINAL_DIR=%cd%
SET CAFFE2_ROOT=%~dp0%..

if NOT DEFINED CMAKE_BUILD_TYPE (
  set CMAKE_BUILD_TYPE=Release
)

if not exist %CAFFE2_ROOT%\build_host_protoc mkdir %CAFFE2_ROOT%\build_host_protoc
echo "Created %CAFFE2_ROOT%\build_host_protoc"
cd %CAFFE2_ROOT%\build_host_protoc

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
    set CMAKE_GENERATOR="Visual Studio 14 2015 Win64"
  )
)

echo CAFFE2_ROOT=%CAFFE2_ROOT%
echo CMAKE_GENERATOR=%CMAKE_GENERATOR%
echo CMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%

echo "Generating cmake"
cmake ..\third_party\protobuf\cmake ^
  -G%CMAKE_GENERATOR% ^
  -DCMAKE_INSTALL_PREFIX=. ^
  -Dprotobuf_BUILD_TESTS=OFF ^
  -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% ^
  || goto :label_error

:: Actually run the build
echo "Building protobuf"
cmake --build . --config %CMAKE_BUILD_TYPE% --target INSTALL || goto :label_error

echo "protobuf built successfully"
cd %ORIGINAL_DIR%
endlocal
exit /b 0

:label_error
echo "protobuf building failed"
cd %ORIGINAL_DIR%
endlocal
exit /b 1
