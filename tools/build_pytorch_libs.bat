:: @echo off
cd "%~dp0/.."

set BASE_DIR=%cd:\=/%
set TORCH_LIB_DIR=%cd:\=/%/torch/lib
set INSTALL_DIR=%cd:\=/%/torch/lib/tmp_install
set THIRD_PARTY_DIR=%cd:\=/%/third_party
set PATH=%INSTALL_DIR%/bin;%PATH%
set BASIC_C_FLAGS= /I%INSTALL_DIR%/include /I%INSTALL_DIR%/include/TH /I%INSTALL_DIR%/include/THC /I%INSTALL_DIR%/include/THS /I%INSTALLDIR%/include/THCS /I%INSTALLDIR%/include/THPP /I%INSTALLDIR%/include/THNN /I%INSTALLDIR%/include/THCUNN
set BASIC_CUDA_FLAGS= -I%INSTALL_DIR%/include -I%INSTALL_DIR%/include/TH -I%INSTALL_DIR%/include/THC -I%INSTALL_DIR%/include/THS -I%INSTALLDIR%/include/THCS -I%INSTALLDIR%/include/THPP -I%INSTALLDIR%/include/THNN -I%INSTALLDIR%/include/THCUNN
set LDFLAGS=/LIBPATH:%INSTALL_DIR%/lib
:: set TORCH_CUDA_ARCH_LIST=6.1

set CWRAP_FILES=%BASE_DIR%/torch/lib/ATen/Declarations.cwrap;%BASE_DIR%/torch/lib/ATen/Local.cwrap;%BASE_DIR%/torch/lib/THNN/generic/THNN.h;%BASE_DIR%/torch/lib/THCUNN/generic/THCUNN.h;%BASE_DIR%/torch/lib/ATen/nn.yaml
set C_FLAGS=%BASIC_C_FLAGS% /D_WIN32 /Z7 /EHa /DNOMINMAX
set LINK_FLAGS=/DEBUG:FULL

mkdir torch/lib/tmp_install

IF "%~1"=="--use-cuda" (
  set /a USE_CUDA=1
  shift
) ELSE (
  set /a USE_CUDA=0
)

IF "%~1"=="--use-rocm" (
  set /a USE_ROCM=1
  shift
) ELSE (
  set /a USE_ROCM=0
)

IF "%~1"=="--use-nnpack" (
  set /a NO_NNPACK=0
  set /a USE_NNPACK=1
  shift
) ELSE (
  set /a NO_NNPACK=1
  set /a USE_NNPACK=0
)

IF "%~1"=="--use-mkldnn" (
  set /a NO_MKLDNN=0
  shift
) ELSE (
  set /a NO_MKLDNN=1
)

IF "%~1"=="--use-gloo-ibverbs" (
  set /a USE_GLOO_IBVERBS=1
  echo Warning: gloo iverbs is enabled but build is not yet implemented 1>&2
  shift
) ELSE (
  set /a USE_GLOO_IBVERBS=0
)

set BUILD_TYPE=Release
IF "%DEBUG%"=="1" (
  set BUILD_TYPE=Debug
)
IF "%REL_WITH_DEB_INFO%"=="1" (
  set BUILD_TYPE=RelWithDebInfo
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


:read_loop
if "%1"=="" goto after_loop
if "%1"=="caffe2" (
  call:build_caffe2 %~1
) ELSE (
  set "IS_OURS="
  IF "%1"=="THD" set IS_OURS=1
  IF "%1"=="libshm_windows" set IS_OURS=1
  if defined IS_OURS (
    cd torch\lib
    call:build %~1
    cd ..\..
  ) ELSE (
    cd third_party
    call:build %~1
    cd ..
  )
)
shift
goto read_loop

:after_loop

cd torch/lib

copy /Y tmp_install\lib\* .
IF EXIST ".\tmp_install\bin" (
  copy /Y tmp_install\bin\* .
)
xcopy /Y /E tmp_install\include\*.* include\*.*
xcopy /Y ..\..\aten\src\THNN\generic\THNN.h  .
xcopy /Y ..\..\aten\src\THCUNN\generic\THCUNN.h .

cd ..\..

goto:eof

:build
  @setlocal
  IF NOT "%PREBUILD_COMMAND%"=="" call "%PREBUILD_COMMAND%" %PREBUILD_COMMAND_ARGS%
  mkdir build\%~1
  cd build/%~1
  cmake ../../%~1 %CMAKE_GENERATOR_COMMAND% ^
                  -DCMAKE_MODULE_PATH=%BASE_DIR%/cmake/FindCUDA ^
                  -DTorch_FOUND="1" ^
                  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
                  -DCMAKE_C_FLAGS="%C_FLAGS%" ^
                  -DCMAKE_SHARED_LINKER_FLAGS="%LINK_FLAGS%" ^
                  -DCMAKE_CXX_FLAGS="%C_FLAGS% %CPP_FLAGS%" ^
                  -DCUDA_NVCC_FLAGS="%BASIC_CUDA_FLAGS%" ^
                  -Dcwrap_files="%CWRAP_FILES%" ^
                  -DTH_INCLUDE_PATH="%INSTALL_DIR%/include" ^
                  -DTH_LIB_PATH="%INSTALL_DIR%/lib" ^
                  -DTH_LIBRARIES="%INSTALL_DIR%/lib/caffe2.lib" ^
                  -DTHS_LIBRARIES="%INSTALL_DIR%/lib/caffe2.lib" ^
                  -DTHC_LIBRARIES="%INSTALL_DIR%/lib/caffe2_gpu.lib" ^
                  -DTHCS_LIBRARIES="%INSTALL_DIR%/lib/caffe2_gpu.lib" ^
                  -DCAFFE2_LIBRARIES="%INSTALL_DIR%/lib/caffe2.lib" ^
                  -DTHNN_LIBRARIES="%INSTALL_DIR%/lib/caffe2.lib" ^
                  -DTHCUNN_LIBRARIES="%INSTALL_DIR%/lib/caffe2_gpu.lib" ^
                  -DTH_SO_VERSION=1 ^
                  -DTHC_SO_VERSION=1 ^
                  -DTHNN_SO_VERSION=1 ^
                  -DTHCUNN_SO_VERSION=1 ^
                  -DUSE_CUDA=%USE_CUDA% ^
                  -DBUILD_EXAMPLES=OFF ^
                  -DBUILD_TEST=%BUILD_TEST% ^
                  -DNO_NNPACK=%NO_NNPACK% ^
                  -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

  %MAKE_COMMAND%
  IF ERRORLEVEL 1 exit 1
  IF NOT ERRORLEVEL 0 exit 1
  cd ../..
  @endlocal

goto:eof

:build_caffe2
  @setlocal
  IF NOT "%PREBUILD_COMMAND%"=="" call "%PREBUILD_COMMAND%" %PREBUILD_COMMAND_ARGS%
  mkdir build
  cd build
  cmake .. %CMAKE_GENERATOR_COMMAND% ^
                  -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
                  -DTORCH_BUILD_VERSION="%PYTORCH_BUILD_VERSION%" ^
                  -DBUILD_TORCH="%BUILD_TORCH%" ^
                  -DNVTOOLEXT_HOME="%NVTOOLEXT_HOME%" ^
                  -DNO_API=ON ^
                  -DBUILD_SHARED_LIBS="%BUILD_SHARED_LIBS%" ^
                  -DBUILD_PYTHON=%BUILD_PYTHON% ^
                  -DBUILD_BINARY=%BUILD_BINARY% ^
                  -DBUILD_TEST=%BUILD_TEST% ^
                  -DINSTALL_TEST=%INSTALL_TEST% ^
                  -DBUILD_CAFFE2_OPS=%BUILD_CAFFE2_OPS% ^
                  -DONNX_NAMESPACE=%ONNX_NAMESPACE% ^
                  -DUSE_CUDA=%USE_CUDA% ^
                  -DUSE_NUMPY=%USE_NUMPY% ^
                  -DUSE_CUDNN=OFF ^
                  -DUSE_NNPACK=%USE_NNPACK% ^
                  -DUSE_LEVELDB=%USE_LEVELDB% ^
                  -DUSE_LMDB=%USE_LMDB% ^
                  -DUSE_OPENCV=%USE_OPENCV% ^
                  -DUSE_GLOG=OFF ^
                  -DUSE_GFLAGS=OFF ^
                  -DUSE_SYSTEM_EIGEN_INSTALL=OFF ^
                  -DCUDNN_INCLUDE_DIR="%CUDNN_INCLUDE_DIR%" ^
                  -DCUDNN_LIB_DIR="%CUDNN_LIB_DIR%" ^
                  -DCUDNN_LIBRARY="%CUDNN_LIBRARY%" ^
                  -DNO_MKLDNN=%NO_MKLDNN% ^
                  -DMKLDNN_INCLUDE_DIR="%MKLDNN_INCLUDE_DIR%" ^
                  -DMKLDNN_LIB_DIR="%MKLDNN_LIB_DIR%" ^
                  -DMKLDNN_LIBRARY="%MKLDNN_LIBRARY%" ^
                  -DATEN_NO_CONTRIB=1 ^
                  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
                  -DCMAKE_C_FLAGS="%USER_CFLAGS%" ^
                  -DCMAKE_CXX_FLAGS="/EHa %USER_CFLAGS%" ^
                  -DCMAKE_EXE_LINKER_FLAGS="%USER_LDFLAGS%" ^
                  -DCMAKE_SHARED_LINKER_FLAGS="%USER_LDFLAGS%" ^
                  -DUSE_ROCM=%USE_ROCM%

  %MAKE_COMMAND%
  IF ERRORLEVEL 1 exit 1
  IF NOT ERRORLEVEL 0 exit 1
  cd ..
  @endlocal

goto:eof
