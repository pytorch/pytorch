#!/bin/bash
set -ex
RUNDIR="$(cd "$(dirname "$0")" ; pwd -P)"
BUILDDIR="$RUNDIR/build"
BUILDTYPE="Release"
VERB=
FILTER=
BUILD_ONLY=0
RMBUILD=1
LOGFILE=/tmp/pytorch_cpp_test.log
LTC_EXPERIMENTAL="nonzero:masked_select"

if [ "$DEBUG" == "1" ]; then
  BUILDTYPE="Debug"
fi

while getopts 'VLDKBF:X:' OPTION
do
  case $OPTION in
    V)
      VERB="VERBOSE=1"
      ;;
    L)
      LOGFILE=
      ;;
    D)
      BUILDTYPE="Debug"
      ;;
    K)
      RMBUILD=0
      ;;
    B)
      BUILD_ONLY=1
      ;;
    F)
      FILTER="--gtest_filter=$OPTARG"
      ;;
    X)
      LTC_EXPERIMENTAL="$OPTARG"
      ;;
  esac
done
shift $(($OPTIND - 1))

if [[ "$TPUVM_MODE" != "1" ]]; then
  # Dynamic shape is not supported on the tpuvm.
  export LTC_EXPERIMENTAL
fi

rm -rf "$BUILDDIR"
mkdir "$BUILDDIR" 2>/dev/null
pushd "$BUILDDIR"

export GET_PYTHON_LIB_SCRIPT="
import distutils.sysconfig as sysconfig
import pathlib
import sys

path = sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY')
if not pathlib.Path(path).exists() and path[-2:] == '.a':
  path = path[:-2] + '.so'
print(path)
"
cmake "$RUNDIR" \
  -DCMAKE_BUILD_TYPE=$BUILDTYPE \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_LIBRARY=$(python -c "$GET_PYTHON_LIB_SCRIPT")
# Ninja needs a separate build invocation for googletest.
cmake --build . --target googletest
cmake --build . --target all

if [ $BUILD_ONLY -eq 0 ]; then
  if [ "$LOGFILE" != "" ]; then
    ./test_ptltc ${FILTER:+"$FILTER"} 2> $LOGFILE
  else
    ./test_ptltc ${FILTER:+"$FILTER"}
  fi
fi
popd
if [ $RMBUILD -eq 1 -a $BUILD_ONLY -eq 0 ]; then
  rm -rf "$BUILDDIR"
fi
