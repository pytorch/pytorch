

@echo off
cd "%~dp0"
cd "../.."

set BASE_DIR=%cd:\=/%
cd torch/lib

set INSTALL_DIR=%cd:\=/%/tmp_install
set PATH=%INSTALL_DIR%/bin;%PATH%
set BASIC_C_FLAGS= /DTH_INDEX_BASE=0 /I%INSTALL_DIR%/include /I%INSTALL_DIR%/include/TH /I%INSTALL_DIR%/include/THC
set BASIC_CUDA_FLAGS= -DTH_INDEX_BASE=0 -I%INSTALL_DIR%/include -I%INSTALL_DIR%/include/TH -I%INSTALL_DIR%/include/THC
set LDFLAGS=/LIBPATH:%INSTALL_DIR%/lib
:: set TORCH_CUDA_ARCH_LIST=6.1

set C_FLAGS=%BASIC_C_FLAGS% /D_WIN32 /Z7 /EHa /DNOMINMAX
set LINK_FLAGS=/DEBUG:FULL

mkdir tmp_install
call:build TH
call:build THS
call:build THNN

IF "%~1"=="--with-cuda" (
  call:build THC
  call:build THCS 
  call:build THCUNN
)

call:build THPP
call:build libshm_windows

copy /Y tmp_install\lib\* .
copy /Y tmp_install\bin\* .
xcopy /Y THNN\generic\THNN.h .
xcopy /Y THCUNN\generic\THCUNN.h .

goto:eof

:build
  mkdir build\%~1
  cd build/%~1
  cmake ../../%~1 -G "Visual Studio 14 2015 Win64" ^
                  -DCMAKE_MODULE_PATH=%BASE_DIR%/cmake/FindCUDA ^
                  -DTorch_FOUND="1" ^
                  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
                  -DCMAKE_C_FLAGS="%C_FLAGS%" ^
                  -DCMAKE_SHARED_LINKER_FLAGS="%LINK_FLAGS%" ^
                  -DCMAKE_CXX_FLAGS="%C_FLAGS% %CPP_FLAGS%" ^
                  -DCUDA_NVCC_FLAGS="%BASIC_CUDA_FLAGS%" ^
                  -DTH_INCLUDE_PATH="%INSTALL_DIR%/include" ^
                  -DTH_LIB_PATH="%INSTALL_DIR%/lib" ^
                  -DTH_LIBRARIES="%INSTALL_DIR%/lib/TH.lib" ^
                  -DTHS_LIBRARIES="%INSTALL_DIR%/lib/THS.lib" ^
                  -DTHC_LIBRARIES="%INSTALL_DIR%/lib/THC.lib" ^
                  -DTHCS_LIBRARIES="%INSTALL_DIR%/lib/THCS.lib" ^
                  -DTH_SO_VERSION=1 ^
                  -DTHC_SO_VERSION=1 ^
                  -DTHNN_SO_VERSION=1 ^
                  -DTHCUNN_SO_VERSION=1 ^
                  -DCMAKE_BUILD_TYPE=Release ^
                  -DLAPACK_LIBRARIES="%INSTALL_DIR%/lib/mkl_rt.lib" -DLAPACK_FOUND=TRUE 
                  :: debug/release

  msbuild INSTALL.vcxproj /p:Configuration=Release
  cd ../..

goto:eof


