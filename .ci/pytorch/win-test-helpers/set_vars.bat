set DISTUTILS_USE_SDK=1
set PATH=%TMP_DIR_WIN%\bin;%PATH%

:: The latest Windows CUDA test is running on AWS G5 runner with A10G GPU
if "%TORCH_CUDA_ARCH_LIST%" == "" set TORCH_CUDA_ARCH_LIST=8.6

:: The default sccache idle timeout is 600, which is too short and leads to intermittent build errors.
set SCCACHE_IDLE_TIMEOUT=0
set SCCACHE_IGNORE_SERVER_IO_ERROR=1
sccache --stop-server
sccache --start-server
sccache --zero-stats
set CC=sccache-cl
set CXX=sccache-cl

set CMAKE_GENERATOR=Ninja
