#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
-D CMAKE_CXX_FLAGS="-O3 -ftemplate-backtrace-limit=0 -gline-tables-only -save-temps=$PWD"         \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D BUILD_DEV=ON                                                                                   \
-D GPU_TARGETS="gfx908;gfx90a"                                                                    \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                 \
-D USE_BITINT_EXTENSION_INT4=OFF                                                                  \
${MY_PROJECT_SOURCE}

#-D AMDGPU_TARGETS=gfx90a;gfx908
