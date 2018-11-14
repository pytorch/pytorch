#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

set -ex

echo "Testing Caffe2"

# libdc1394 (dependency of OpenCV) expects /dev/raw1394 to exist...
if [ ! -e /dev/raw1394 ]; then
  sudo ln /dev/null /dev/raw1394
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Hotfix, use hypothesis 3.44.6 on Ubuntu 14.04
# See comments on https://github.com/HypothesisWorks/hypothesis-python/commit/eadd62e467d6cee6216e71b391951ec25b4f5830
if [[ "$BUILD_ENVIRONMENT" == *ubuntu14.04* ]]; then
  sudo pip -q uninstall -y hypothesis
  # "pip install hypothesis==3.44.6" from official server is unreliable on CircleCI, so we host a copy on S3 instead
  sudo pip -q install attrs==18.1.0 -f https://s3.amazonaws.com/ossci-linux/wheels/attrs-18.1.0-py2.py3-none-any.whl
  sudo pip -q install coverage==4.5.1 -f https://s3.amazonaws.com/ossci-linux/wheels/coverage-4.5.1-cp36-cp36m-macosx_10_12_x86_64.whl
  sudo pip -q install hypothesis==3.44.6 -f https://s3.amazonaws.com/ossci-linux/wheels/hypothesis-3.44.6-py3-none-any.whl
fi

# conda must be added to the path for Anaconda builds (this location must be
# the same as that in install_anaconda.sh used to build the docker image)
if [[ "${BUILD_ENVIRONMENT}" == conda* ]]; then
  export PATH=/opt/conda/bin:$PATH
fi

# set the env var for onnx build and test
if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  export INTEGRATED=1
fi

TEST_DIR=$ROOT_DIR/caffe2_tests

# Install Caffe2 test requirements
pip -q install --user Cython

# Upgrade SSL module to avoid old SSL warnings
pip -q install --user --upgrade pyOpenSSL ndg-httpsclient pyasn1

# Install Caffe2 test requirements
pip -q install --user hypothesis
pip -q install --user mock
pip -q install --user onnx
#pip -q install --user -b /tmp/pip_install_onnx "file://${ROOT_DIR}/third_party/onnx#egg=onnx"

# Install ONNX test requirements
pip -q install --user click
pip -q install --user typing
pip -q install --user typing-extensions
pip -q install --user tabulate

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

# Set PYTHONPATH and LD_LIBRARY_PATH so that python can find the installed
# Caffe2.
export PYTHONPATH="${PYTHONPATH}:$INSTALL_SITE_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${INSTALL_LIB_DIR}"
ls "${INSTALL_LIB_DIR}"

cd "$ROOT_DIR"

if [ -d $TEST_DIR ]; then
  echo "Directory $TEST_DIR already exists; removing it."
  rm -rf $TEST_DIR
fi

mkdir -p $TEST_DIR/{cpp,python}

cd "${WORKSPACE}"

#############
# C++ tests #
#############

echo "Running C++ tests.."
gtest_reports_dir="${TEST_DIR}/cpp"
mkdir -p "$gtest_reports_dir"
for test in $(find "${INSTALL_PREFIX}/cpp_test" -executable -type f); do
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
      "$test" --gtest_output=xml:"$gtest_reports_dir/$(basename $test).xml"
      ;;
  esac
done

################
# Python tests #
################

pytest_reports_dir="${TEST_DIR}/python"
mkdir -p "$pytest_reports_dir"

# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]]; then
  EXTRA_TESTS+=("$CAFFE2_PYPATH/contrib/nccl")
fi

rocm_ignore_test=()
if [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
  # Currently these tests are failing on ROCM platform:

  # Unknown reasons, need to debug
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/arg_ops_test.py")
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/piecewise_linear_transform_test.py")
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/softmax_ops_test.py")
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/unique_ops_test.py")
fi

# NB: Warnings are disabled because they make it harder to see what
# the actual erroring test is
echo "Running Python tests.."
pip install --user pytest-sugar
"$PYTHON" \
  -m pytest \
  -x \
  -v \
  --disable-warnings \
  --junit-xml="$pytest_reports_dir/result.xml" \
  --ignore "$CAFFE2_PYPATH/python/test/executor_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/matmul_op_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/pack_ops_test.py" \
  --ignore "$CAFFE2_PYPATH/python/mkl/mkl_sbn_speed_test.py" \
  ${rocm_ignore_test[@]} \
  "$CAFFE2_PYPATH/python" \
  "${EXTRA_TESTS[@]}"

cd ${INSTALL_PREFIX}

if [[ -n "$INTEGRATED" ]]; then
  pip install --user torchvision
  "$ROOT_DIR/scripts/onnx/test.sh"
fi

# Remove benign core dumps.
# These are tests for signal handling (including SIGABRT).
rm -f ./crash/core.fatal_signal_as.*
rm -f ./crash/core.logging_test.*
