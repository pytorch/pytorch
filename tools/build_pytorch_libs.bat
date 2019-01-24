:: @echo off
cd "%~dp0\.."

set PATH=%INSTALL_DIR%\bin;%PATH%

: The following environment variables are used exclusively by cmake and should have forward slashes rather than backslashes
set BASE_DIR=%cd:\=/%
set TORCH_LIB_DIR=%cd:\=/%/torch/lib
set THIRD_PARTY_DIR=%cd:\=/%/third_party
set BASIC_C_FLAGS=
set BASIC_CUDA_FLAGS=

IF NOT DEFINED INSTALL_DIR (
  set "INSTALL_DIR=%cd:\=/%/torch/lib/tmp_install"
) ELSE (
  set "INSTALL_DIR=%INSTALL_DIR:\=/%"
)

set LDFLAGS=/LIBPATH:%INSTALL_DIR%/lib
:: set TORCH_CUDA_ARCH_LIST=6.1

set C_FLAGS=%BASIC_C_FLAGS% /D_WIN32 /Z7 /EHa /DNOMINMAX
set LINK_FLAGS=
: End cmake variables

if not exist torch\lib\tmp_install mkdir torch\lib\tmp_install

: Variable defaults
set /a USE_CUDA=0
set /a USE_FBGEMM=1
set /a USE_ROCM=0
set /a USE_NNPACK=0
set /a USE_QNNPACK=0
set /a USE_GLOO_IBVERBS=0
set /a USE_MKLDNN=0

set _BUILD_ARGS=

: Process command line arguments
:process_args
if "%1"=="" (
  goto :process_args_exit
)

if "%1"=="--use-cuda" (
  set /a USE_CUDA=1
  goto :process_args_processed
)

if "%1"=="--use-fbgemm" (
  set /a USE_FBGEMM=1
  goto :process_args_processed
)

if "%1"=="--use-rocm" (
  set /a USE_ROCM=1
  goto :process_args_processed
)

if "%1"=="--use-nnpack" (
  set /a USE_NNPACK=1
  goto :process_args_processed
)

if "%1"=="--use-qnnpack" (
  set /a USE_QNNPACK=1
  goto :process_args_processed
)

if "%1"=="--use-mkldnn" (
  set /a USE_MKLDNN=1
  goto :process_args_processed
)

if "%1"=="--use-gloo-ibverbs" (
  echo Warning: gloo iverbs is enabled but build is not yet implemented 1>&2
  set /a USE_GLOO_IBVERBS=1
  goto :process_args_processed
)

set _BUILD_ARGS=%_BUILD_ARGS% %1
:: Fall-through intended

:process_args_processed
shift
goto :process_args

:process_args_exit

: Set vars based on environment state
set BUILD_TYPE=Release
IF "%DEBUG%"=="1" (
  set BUILD_TYPE=Debug
  set LINK_FLAGS=%LINK_FLAGS% /DEBUG:FULL
)
IF "%REL_WITH_DEB_INFO%"=="1" (
  set BUILD_TYPE=RelWithDebInfo
  set LINK_FLAGS=%LINK_FLAGS% /DEBUG:FULL
)

:: sccache will fail if all cores are used for compiling
IF NOT DEFINED MAX_JOBS (
  set /a MAX_JOBS=%NUMBER_OF_PROCESSORS% - 1
)

IF NOT DEFINED BUILD_SHARED_LIBS (
  set BUILD_SHARED_LIBS=ON
)

IF "%CMAKE_GENERATOR%"=="" (
  set CMAKE_GENERATOR_COMMAND=
  set MAKE_COMMAND=msbuild INSTALL.vcxproj /p:Configuration=Release
) ELSE (
  set CMAKE_GENERATOR_COMMAND=-G "%CMAKE_GENERATOR%"
  IF "%CMAKE_GENERATOR%"=="Ninja" (
    IF "%CC%"== "" set CC=cl.exe
    IF "%CXX%"== "" set CXX=cl.exe
    set MAKE_COMMAND=cmake --build . --target install --config %BUILD_TYPE% -- -j%MAX_JOBS%
  ) ELSE (
    set MAKE_COMMAND=msbuild INSTALL.vcxproj /p:Configuration=%BUILD_TYPE%
  )
)

: Note [Backslash munging on Windows]
: In CMake, Windows native backslashes are not well handled.
: It will cause a warning as the following
:   CMake Warning (dev) at cmake (source_group):
:    Syntax error in cmake code at cmake
:    when parsing string
:      Header Files C:\include\cudnn.h
:    Invalid escape sequence \i
: which is said to become an error in the future.
: As an alternative, we should use forward slashes instead.
: Here those paths should be escaped before passing to CMake.
if not "%NVTOOLEXT_HOME%" == "" set NVTOOLEXT_HOME=%NVTOOLEXT_HOME:\=/%
if not "%CUDNN_INCLUDE_DIR%" == "" set CUDNN_INCLUDE_DIR=%CUDNN_INCLUDE_DIR:\=/%
if not "%CUDNN_LIB_DIR%" == "" set CUDNN_LIB_DIR=%CUDNN_LIB_DIR:\=/%
if not "%CUDNN_LIBRARY%" == "" set CUDNN_LIBRARY=%CUDNN_LIBRARY:\=/%
if not "%PYTORCH_PYTHON_LIBRARY%" == "" set PYTORCH_PYTHON_LIBRARY=%PYTORCH_PYTHON_LIBRARY:\=/%
if not "%NUMPY_INCLUDE_DIR%" == "" set NUMPY_INCLUDE_DIR=%NUMPY_INCLUDE_DIR:\=/%

IF NOT "%PREBUILD_COMMAND%"=="" call "%PREBUILD_COMMAND%" %PREBUILD_COMMAND_ARGS%
if not exist build mkdir build
pushd build
cmake .. %CMAKE_GENERATOR_COMMAND% ^
                -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
                -DTORCH_BUILD_VERSION="%PYTORCH_BUILD_VERSION%" ^
                -DPYTHON_LIBRARY="%PYTORCH_PYTHON_LIBRARY%" ^
                -DBUILD_TORCH="%BUILD_TORCH%" ^
                -DNVTOOLEXT_HOME="%NVTOOLEXT_HOME%" ^
                -DBUILD_SHARED_LIBS="%BUILD_SHARED_LIBS%" ^
                -DBUILD_PYTHON=%BUILD_PYTHON% ^
                -DBUILD_BINARY=%BUILD_BINARY% ^
                -DBUILD_TEST=%BUILD_TEST% ^
                -DINSTALL_TEST=%INSTALL_TEST% ^
                -DBUILD_CAFFE2_OPS=%BUILD_CAFFE2_OPS% ^
                -DONNX_NAMESPACE=%ONNX_NAMESPACE% ^
                -DUSE_CUDA=%USE_CUDA% ^
                -DUSE_DISTRIBUTED=%USE_DISTRIBUTED% ^
                -DUSE_FBGEMM=%USE_FBGEMM% ^
                -DUSE_NUMPY=%USE_NUMPY% ^
                -DNUMPY_INCLUDE_DIR="%NUMPY_INCLUDE_DIR%" ^
                -DUSE_NNPACK=%USE_NNPACK% ^
                -DUSE_LEVELDB=%USE_LEVELDB% ^
                -DUSE_LMDB=%USE_LMDB% ^
                -DUSE_OPENCV=%USE_OPENCV% ^
                -DUSE_QNNPACK=%USE_QNNPACK% ^
                -DUSE_FFMPEG=%USE_FFMPEG% ^
                -DUSE_GLOG=OFF ^
                -DUSE_GFLAGS=OFF ^
                -DUSE_SYSTEM_EIGEN_INSTALL=OFF ^
                -DCUDNN_INCLUDE_DIR="%CUDNN_INCLUDE_DIR%" ^
                -DCUDNN_LIB_DIR="%CUDNN_LIB_DIR%" ^
                -DCUDNN_LIBRARY="%CUDNN_LIBRARY%" ^
                -DUSE_MKLDNN=%USE_MKLDNN% ^
                -DATEN_NO_CONTRIB=1 ^
                -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
                -DCMAKE_C_FLAGS="%USER_CFLAGS%" ^
                -DCMAKE_CXX_FLAGS="/EHa %USER_CFLAGS%" ^
                -DCMAKE_EXE_LINKER_FLAGS="%USER_LDFLAGS%" ^
                -DCMAKE_SHARED_LINKER_FLAGS="%USER_LDFLAGS%" ^
                -DUSE_ROCM=%USE_ROCM% %EXTRA_CAFFE2_CMAKE_FLAGS%
IF ERRORLEVEL 1 exit 1
IF NOT ERRORLEVEL 0 exit 1

%MAKE_COMMAND%
IF ERRORLEVEL 1 exit 1
IF NOT ERRORLEVEL 0 exit 1

:: Install Python proto files
IF "%BUILD_PYTHON%" == "ON" (
  for /f "delims=" %%i in ('where /R caffe2\proto *.py') do (
    IF NOT "%%i" == "%CD%\caffe2\proto\__init__.py" (
      copy /Y %%i ..\caffe2\proto\
    )
  )
)

popd

: Copy Artifacts
cd torch\lib

copy /Y "%INSTALL_DIR%\lib\*" .
IF EXIST "%INSTALL_DIR%\bin" (
  copy /Y "%INSTALL_DIR%\bin\*" .
)
xcopy /Y /E "%INSTALL_DIR%\include\*.*" include\*.*
xcopy /Y ..\..\aten\src\THNN\generic\THNN.h  .
xcopy /Y ..\..\aten\src\THCUNN\generic\THCUNN.h .

cd ..\..
