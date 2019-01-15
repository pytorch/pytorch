#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

rm -rf "$TEST_DIR" && mkdir -p "$TEST_DIR"

cd "${WORKSPACE}"

#############
# C++ tests #
#############

echo "Running C++ tests.."
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${INSTALL_PREFIX}/lib"
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

################################################################################
# Python tests #
################################################################################
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  exit 0
fi

# Ideally this would be where the Python bits get installed to from using
# setup.py. However on our dockers this is not correct for two reasons.
# 1. This lies in /usr/local/lib/pythonM.m, but the dockers don't have the
#    right permissions setup so the build doesn't have write access to this
#    dir.  For this reason we use --user flag in all pip install instructions
#    which install into the $HOME/.local directory instead.
# 2. This returns lib/pythonM.m/dist-packages, but we install in site-packages.
# We use this same way of getting the install directory in other places in our
# build, so not really sure why it is not correct here
INSTALL_SITE_DIR=$($PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_lib())")
if [[ -n "$(find $INSTALL_SITE_DIR -name caffe2 2>/dev/null)" ]]; then
  # Caffe2 will probably be found here if using a Python from a virtualenv or
  # from conda
  CAFFE2_PYPATH="$INSTALL_SITE_DIR/caffe2"
elif [[ -n "$(find $HOME/.local/lib -name caffe2 2>/dev/null)" ]]; then
  # Caffe2 will be found here in the case of using no env and adding --user to
  # the setup.py call
  pyver=($(python --version 2>&1))
  pyver=${pyver[1]}
  pyver=${pyver:0:3}
  CAFFE2_PYPATH="$HOME/.local/lib/python$pyver/site-packages/caffe2"
else
  echo "I do not know where Caffe2 is installed"
  find / -name caffe2 2>/dev/null
  exit 1
fi
if [[ ! -d "$CAFFE2_PYPATH" ]]; then
  echo "Failed to find where Caffe2 Python bits are installed"
  find / -name caffe2 2>/dev/null
  exit 1
fi


if [[ "$BUILD_ENVIRONMENT" == *ubuntu14.04* ]]; then
  # Hotfix, use hypothesis 3.44.6 on Ubuntu 14.04
  # See comments on
  # https://github.com/HypothesisWorks/hypothesis-python/commit/eadd62e467d6cee6216e71b391951ec25b4f5830
  sudo pip -q uninstall -y hypothesis
  # "pip install hypothesis==3.44.6" from official server is unreliable on
  # CircleCI, so we host a copy on S3 instead
  sudo pip -q install attrs==18.1.0 -f https://s3.amazonaws.com/ossci-linux/wheels/attrs-18.1.0-py2.py3-none-any.whl
  sudo pip -q install coverage==4.5.1 -f https://s3.amazonaws.com/ossci-linux/wheels/coverage-4.5.1-cp36-cp36m-macosx_10_12_x86_64.whl
  sudo pip -q install hypothesis==3.44.6 -f https://s3.amazonaws.com/ossci-linux/wheels/hypothesis-3.44.6-py3-none-any.whl
else
  pip install --user --no-cache-dir hypothesis==3.59.0
fi

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

  # On ROCm, RCCL (distributed) development isn't complete.
  # https://github.com/ROCmSoftwarePlatform/rccl
  rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/data_parallel_model_test.py")
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

#####################
# torchvision tests #
#####################

cd ${INSTALL_PREFIX}

if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  pip install --user torchvision
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
