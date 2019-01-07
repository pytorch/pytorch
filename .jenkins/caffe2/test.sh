#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

cd "$ROOT_DIR"

TEST_DIR="$ROOT_DIR/caffe2_tests"
rm -rf "$TEST_DIR" && mkdir -p "$TEST_DIR"

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

cd /var/lib/jenkins

if [[ -n "$INTEGRATED" ]]; then
  pip install --user torchvision
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
