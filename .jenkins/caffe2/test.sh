#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR=$ROOT_DIR/caffe2_tests

# Figure out which Python to use
PYTHON="python"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  PYTHON="python${BASH_REMATCH[1]}"
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

cd "${WORKSPACE}"

# C++ tests
echo "Running C++ tests.."
gtest_reports_dir="${TEST_DIR}/cpp"
junit_reports_dir="${TEST_DIR}/junit_reports"
mkdir -p "$gtest_reports_dir" "$junit_reports_dir"
for test in $(find "${INSTALL_PREFIX}/test" -executable -type f); do
  case "$test" in
    # skip tests we know are hanging or bad
    */mkl_utils_test|*/aten/integer_divider_test)
      continue
      ;;
    */scalar_tensor_test|*/basic|*/native_test)
	  if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
		continue
	  else
	    "$test"
	  fi
	  ;;
	*)
      # Currently, we use a mixture of gtest (caffe2) and Catch2 (ATen). While
      # planning to migrate to gtest as the common PyTorch c++ test suite, we
      # currently do NOT use the xml test reporter, because Catch doesn't
      # support multiple reporters
      # c.f. https://github.com/catchorg/Catch2/blob/master/docs/release-notes.md#223
      # which means that enabling XML output means you lose useful stdout
      # output for Jenkins.  It's more important to have useful console
      # output than it is to have XML output for Jenkins.
      # Note: in the future, if we want to use xml test reporter once we switch
      # to all gtest, one can simply do:
      # "$test" --gtest_output=xml:"$gtest_reports_dir/$(basename $test).xml"
      "$test"
      ;;
  esac
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

rocm_ignore_test=()
if [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
  # Currently these tests are failing on ROCM platform:

  # Unknown reasons, need to debug
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/arg_ops_test.py")
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/piecewise_linear_transform_test.py")
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/unique_ops_test.py")
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/softmax_ops_test.py")
  
  # Need to go through roi ops to replace max(...) with fmaxf(...)
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/roi_align_rotated_op_test.py")
fi

# Python tests
# NB: Warnings are disabled because they make it harder to see what
# the actual erroring test is
echo "Running Python tests.."
pip install --user pytest-sugar
"$PYTHON" \
  -m pytest \
  -x \
  -v \
  --disable-warnings \
  --junit-xml="$TEST_DIR/python/result.xml" \
  --ignore "$CAFFE2_PYPATH/python/test/executor_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/matmul_op_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/pack_ops_test.py" \
  --ignore "$CAFFE2_PYPATH/python/mkl/mkl_sbn_speed_test.py" \
  ${conda_ignore_test[@]} \
  ${rocm_ignore_test[@]} \
  "$CAFFE2_PYPATH/python" \
  "${EXTRA_TESTS[@]}"

cd ${INSTALL_PREFIX}

if [[ -n "$INTEGRATED" ]]; then
  pip install --user torchvision
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
