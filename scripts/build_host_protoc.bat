:: ###########################################################################
:: Build script to build the protoc compiler for the host platform.
:: ###########################################################################
:: This script builds the protoc compiler for Windows, which is needed
:: if protobuf is not installed already on windows, as we will need to convert
:: the protobuf source files to cc files.

:: After the execution of the file, one should be able to find the host protoc
:: binary at build_host_protoc/bin/protoc.exe.

@echo off

SET ORIGINAL_DIR=%cd%
SET CAFFE2_ROOT=%~dp0%..
if not exist %CAFFE2_ROOT%\build_host_protoc mkdir %CAFFE2_ROOT%\build_host_protoc
echo "Created %CAFFE2_ROOT%\build_host_protoc"

cd %CAFFE2_ROOT%\build_host_protoc

cmake ..\third_party\protobuf\cmake -DCMAKE_INSTALL_PREFIX=. -Dprotobuf_BUILD_TESTS=OFF
msbuild INSTALL.vcxproj

cd %ORIGINAL_DIR%