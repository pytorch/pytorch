#!/bin/bash

set -ex

ROOT_DIR="$(cd $(dirname ${BASH_SOURCE[0]})/../.. && pwd)"
source "$ROOT_DIR/.jenkins/caffe2/common.sh"

# Find where cpp tests and Caffe2 itself are installed
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  # For cmake only build we install everything into /usr/local
  cpp_test_dir="$INSTALL_PREFIX/cpp_test"
  ld_library_path="$INSTALL_PREFIX/lib"
else
  # For Python builds we install into python
  # cd to /usr first so the python import doesn't get confused by any 'caffe2'
  # directory in cwd
  python_installation="$(dirname $(dirname $(cd /usr && python -c 'import os; import caffe2; print(os.path.realpath(caffe2.__file__))')))"
  caffe2_pypath="$python_installation/caffe2"
  cpp_test_dir="$python_installation/torch/test"
  ld_library_path="$python_installation/torch/lib"
fi

# libdc1394 (dependency of OpenCV) expects /dev/raw1394 to exist...
if [ ! -e /dev/raw1394 ]; then
  sudo ln /dev/null /dev/raw1394
fi

# Make sure there is an empty test dir. This should only be needed if people
# are running this script locally
if [[ -d "$ROOT_DIR/$TEST_DIR" ]]; then
  echo "Directory "$ROOT_DIR/$TEST_DIR" already exists; removing it."
  rm -rf "$ROOT_DIR/$TEST_DIR"
fi
mkdir -p "$ROOT_DIR/$TEST_DIR/{cpp,python}"


################################################################################
# C++ tests #
################################################################################
#echo "Running C++ tests.."
#for test in $(find "$cpp_test_dir" -executable -type f); do
#  case "$test" in
#    # skip tests we know are hanging or bad
#    */mkl_utils_test|*/aten/integer_divider_test)
#      continue
#      ;;
#    */scalar_tensor_test|*/basic|*/native_test)
#      if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
#        continue
#      else
#        LD_LIBRARY_PATH="$ld_library_path" "$test"
#      fi
#      ;;
#    *)
#      # Currently, we use a mixture of gtest (caffe2) and Catch2 (ATen). While
#      # planning to migrate to gtest as the common PyTorch c++ test suite, we
#      # currently do NOT use the xml test reporter, because Catch doesn't
#      # support multiple reporters
#      # c.f. https://github.com/catchorg/Catch2/blob/master/docs/release-notes.md#223
#      # which means that enabling XML output means you lose useful stdout
#      # output for Jenkins.  It's more important to have useful console
#      # output than it is to have XML output for Jenkins.
#      # Note: in the future, if we want to use xml test reporter once we switch
#      # to all gtest, one can simply do:
#      LD_LIBRARY_PATH="$ld_library_path" \
#          "$test" --gtest_output=xml:"$gtest_reports_dir/$(basename $test).xml"
#      ;;
#  esac
#done

################################################################################
# Python tests #
################################################################################
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  exit 0
fi

if [[ "$($PYTHON --version 2>&1)" == *3.* ]]; then
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8
fi

# Upgrade SSL module to avoid old SSL warnings
"$PIP" install -q --user --upgrade pyOpenSSL ndg-httpsclient pyasn1

# Install Caffe2 test requirements
"$PIP" install -q --user -r "$ROOT_DIR/requirements.txt"
if [[ "$BUILD_ENVIRONMENT" == *ubuntu14.04* ]]; then
  # Hotfix, use hypothesis 3.44.6 on Ubuntu 14.04
  # See comments on
  # https://github.com/HypothesisWorks/hypothesis-python/commit/eadd62e467d6cee6216e71b391951ec25b4f5830
  "$PIP" uninstall -q --no-deps -y hypothesis
  # "pip install hypothesis==3.44.6" from official server is unreliable on
  # CircleCI, so we host a copy on S3 instead
  "$PIP" install -q --user --no-deps attrs==18.1.0 -f https://s3.amazonaws.com/ossci-linux/wheels/attrs-18.1.0-py2.py3-none-any.whl
  "$PIP" install -q --user --no-deps coverage==4.5.1 -f https://s3.amazonaws.com/ossci-linux/wheels/coverage-4.5.1-cp36-cp36m-macosx_10_12_x86_64.whl
  "$PIP" install -q --user --no-deps hypothesis==3.44.6 -f https://s3.amazonaws.com/ossci-linux/wheels/hypothesis-3.44.6-py3-none-any.whl
else
  "$PIP" install -q --user --no-deps --no-cache-dir attrs coverage hypothesis==3.59.0
fi

# We need networkx==2.0 , but for some reason if we install that with pip then
# conda thinks its actually 2.2, so we install with conda if that's what the
# python is from
if [[ "$(which python)" == *conda* ]]; then
  # TODO maybe create another env here
  conda install -q -y \
    click \
    future \
    mock \
    networkx==2.0 \
    numpy \
    protobuf \
    pytest \
    pyyaml \
    scipy==0.19.1 \
    scikit-image \
    tabulate \
    typing
    "$PIP" install -q --user --no-deps typing-extensions
else
  "$PIP" install -q --user --no-cache-dir \
    click \
    future \
    mock \
    networkx==2.0 \
    numpy \
    protobuf \
    pytest \
    pyyaml \
    scipy==0.19.1 \
    scikit-image \
    tabulate \
    typing \
    typing-extensions
fi

# Install ONNX into a local directory
if [[ ! "$(python -c 'import onnx' 2>/dev/null)" ]]; then
  # This recursive submodule update is needed to checkout pybind11 for the ONNX
  # install to work. The current Pytorch build jobs do not do this while the
  # Caffe2 ones do. When all Caffe2 test jobs are moved on top of Pytorch build
  # jobs, this recursive flag should be added to the base Pytorch build jobs
  pushd "$ROOT_DIR"
  git submodule sync
  git submodule update -q --init --recursive
  popd
  "$PIP" install --user --no-deps -b /tmp/pip_install_onnx "file://${ROOT_DIR}/third_party/onnx#egg=onnx" 
fi

# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]]; then
  EXTRA_TESTS+=("$caffe2_pypath/contrib/nccl")
fi

rocm_ignore_test=()
if [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
  # Currently these tests are failing on ROCM platform:

  # Unknown reasons, need to debug
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/operator_test/arg_ops_test.py")
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/operator_test/piecewise_linear_transform_test.py")
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/operator_test/softmax_ops_test.py")
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/operator_test/unique_ops_test.py")

  # On ROCm, RCCL (distributed) development isn't complete.
  # https://github.com/ROCmSoftwarePlatform/rccl
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/data_parallel_model_test.py")
fi

echo "Running Python tests with these packages"
"$PIP" freeze
conda list || true

# NB: Warnings are disabled because they make it harder to see what
# the actual erroring test is
echo "Running Python tests.."
"$PYTHON" \
  -m pytest \
  -x \
  -v \
  --disable-warnings \
  --junit-xml="$pytest_reports_dir/result.xml" \
  --ignore "$caffe2_pypath/python/test/executor_test.py" \
  --ignore "$caffe2_pypath/python/operator_test/matmul_op_test.py" \
  --ignore "$caffe2_pypath/python/operator_test/pack_ops_test.py" \
  --ignore "$caffe2_pypath/python/mkl/mkl_sbn_speed_test.py" \
  ${rocm_ignore_test[@]} \
  "$caffe2_pypath/python" \
  "${EXTRA_TESTS[@]}"


#####################
# torchvision tests #
#####################
if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  "$PIP" install --user click typing typing-extensions tabulate torchvision
  "$ROOT_DIR/scripts/onnx/test.sh"
fi

# Remove benign core dumps.
# These are tests for signal handling (including SIGABRT).
rm -f $INSTALL_PREFIX/crash/core.fatal_signal_as.* || true
rm -f $INSTALL_PREFIX/crash/core.logging_test.* || true
