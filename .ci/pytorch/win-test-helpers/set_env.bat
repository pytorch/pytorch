if "%DEBUG%=="1" (
  set BUILD_TYPE=debug
) else (
  set BUILD_TYPE=release
)

if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\Jenkins
)

set PATH=C:\Program Files\CMake\bin;C:\Program Files\7-Zip;C:\ProgramData\chocolatey\bin;C:\Program Files\Git\cmd;C:\Program Files\Amazon\AWSCLI;C:\Program Files\Amazon\AWSCLI\bin;%PATH%

set INSTALLER_DIR=%SCRIPT_HELPERS_DIR%\installation-helpers

set CMAKE_INCLUDE_PATH=%TMP_DIR_WIN%\mkl\include

set LIB=%TMP_DIR_WIN%\mkl\lib;%LIB%

set INSTALL_FRESH_CONDA=1

set PATH=%CONDA_PARENT_DIR%\Miniconda3\Library\bin;%CONDA_PARENT_DIR%\Miniconda3;%CONDA_PARENT_DIR%\Miniconda3\Scripts;%PATH%

set DISTUTILS_USE_SDK=1

set PATH=%TMP_DIR_WIN%\bin;%PATH%

if not "%TORCH_CUDA_ARCH_LIST%"=="" (
  set TORCH_CUDA_ARCH_LIST=5.2
)

set SCCACHE_IDLE_TIMEOUT=0

set SCCACHE_IGNORE_SERVER_IO_ERROR=1

set CC=sccache-cl

set CXX=sccache-cl

set CMAKE_GENERATOR=Ninja
