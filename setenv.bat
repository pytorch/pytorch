rem set USE_NINJA=OFF
set USE_CUDA=0

set DEBUG=1
set USE_DISTRIBUTED=0

set CMAKE_VERBOSE_MAKEFILE=1
set VSCMD_DEBUG=2
%set TMP_DIR_WIN=C:\git\
%set CMAKE_INCLUDE_PATH=%TMP_DIR_WIN%\mkl\include
%set LIB=%TMP_DIR_WIN%\mkl\lib;%LIB


set USE_NINJA=OFF
set CMAKE_GENERATOR=Visual Studio 16 2019
set CMAKE_GENERATOR_TOOLSET_VERSION=14.26
set DISTUTILS_USE_SDK=1
set PATH=%TMP_DIR_WIN%\bin;%PATH%
sccache --stop-server
sccache --start-server
sccache --zero-stats
set CC=sccache-cl
set CXX=sccache-cl

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

rem set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64

