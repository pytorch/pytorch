#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR=$ROOT_DIR/caffe2_tests

# Figure out which Python to use
PYTHON="python"
if [ -n "$BUILD_ENVIRONMENT" ]; then
  if [[ "$BUILD_ENVIRONMENT" == py2* ]]; then
    PYTHON="python2"
  elif [[ "$BUILD_ENVIRONMENT" == py3* ]]; then
    PYTHON="python3"
  fi
fi

# The prefix must mirror the setting from build.sh
INSTALL_PREFIX="/usr/local/caffe2"

# Anaconda builds have a special install prefix and python
if [[ "$BUILD_ENVIRONMENT" == conda* ]]; then
  # This path comes from install_anaconda.sh which installs Anaconda into the
  # docker image
  PYTHON="/opt/conda/bin/python"
  INSTALL_PREFIX="/opt/conda/"
fi

# Add the site-packages in the caffe2 install prefix to the PYTHONPATH
SITE_DIR=$($PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_lib(prefix=''))")
INSTALL_SITE_DIR="${INSTALL_PREFIX}/${SITE_DIR}"

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

# Set PYTHONPATH and LD_LIBRARY_PATH so that python can find the installed
# Caffe2. This shouldn't be done on Anaconda, as Anaconda should handle this.
if [[ "$BUILD_ENVIRONMENT" != conda* ]]; then
  export PYTHONPATH="${PYTHONPATH}:$INSTALL_SITE_DIR"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${INSTALL_PREFIX}/lib"
fi

cd "$ROOT_DIR"

if [ -d $TEST_DIR ]; then
  echo "Directory $TEST_DIR already exists; please remove it..."
  exit 1
fi

mkdir -p $TEST_DIR/{cpp,python}

cd ${INSTALL_PREFIX}

# C++ tests
echo "Running C++ tests.."
for test in $INSTALL_PREFIX/test/*; do
  # Skip tests we know are hanging or bad
  case "$(basename "$test")" in
    mkl_utils_test)
      continue
      ;;
  esac

  "$test" --gtest_output=xml:"$TEST_DIR"/cpp/$(basename "$test").xml
done

# Get the relative path to where the caffe2 python module was installed
CAFFE2_PYPATH="$INSTALL_SITE_DIR/caffe2"

# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]]; then
  EXTRA_TESTS+=("$CAFFE2_PYPATH/contrib/nccl")
fi

conda_ignore_test=()
if [[ $BUILD_ENVIRONMENT == conda* ]]; then
  # These tests both assume Caffe2 was built with leveldb, which is not the case
  conda_ignore_test+=("--ignore $CAFFE2_PYPATH/python/dataio_test.py")
  conda_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/checkpoint_test.py")
fi

# Python tests
echo "Running Python tests.."
"$PYTHON" \
  -m pytest \
  -x \
  -v \
  --junit-xml="$TEST_DIR/python/result.xml" \
  --ignore "$CAFFE2_PYPATH/python/test/executor_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/matmul_op_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/pack_ops_test.py" \
  --ignore "$CAFFE2_PYPATH/python/mkl/mkl_sbn_speed_test.py" \
  ${conda_ignore_test[@]} \
  "$CAFFE2_PYPATH/python" \
  "${EXTRA_TESTS[@]}"
