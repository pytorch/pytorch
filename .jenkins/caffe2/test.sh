#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

# Find where cpp tests and Caffe2 itself are installed
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  # For cmake only build we install everything into /usr/local
  cpp_test_dir="$INSTALL_PREFIX/cpp_test"
  ld_library_path="$INSTALL_PREFIX/lib"
else
  # For Python builds we install into python
  # cd to /usr first so the python import doesn't get confused by any 'caffe2'
  # directory in cwd
  python_installation="$(dirname $(dirname $(cd /usr && $PYTHON -c 'import os; import caffe2; print(os.path.realpath(caffe2.__file__))')))"
  caffe2_pypath="$python_installation/caffe2"
  cpp_test_dir="$python_installation/torch/test"
  ld_library_path="$python_installation/torch/lib"
fi

################################################################################
# C++ tests #
################################################################################
echo "Running C++ tests.."
for test in $(find "$cpp_test_dir" -executable -type f); do
  case "$test" in
    # skip tests we know are hanging or bad
    */mkl_utils_test|*/aten/integer_divider_test)
      continue
      ;;
    */scalar_tensor_test|*/basic|*/native_test)
      if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
        continue
      else
        LD_LIBRARY_PATH="$ld_library_path" "$test"
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
      LD_LIBRARY_PATH="$ld_library_path" \
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

# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]]; then
  EXTRA_TESTS+=("$caffe2_pypath/contrib/nccl")
fi

rocm_ignore_test=()
if [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
  # Currently these tests are failing on ROCM platform:

  # On ROCm, RCCL (distributed) development isn't complete.
  # https://github.com/ROCmSoftwarePlatform/rccl
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/data_parallel_model_test.py")
fi

# NB: Warnings are disabled because they make it harder to see what
# the actual erroring test is
echo "Running Python tests.."
if [[ "$BUILD_ENVIRONMENT" == *py3* ]]; then
  # locale setting is required by click package with py3
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8
fi

pip install --user pytest-sugar
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
  pip install -q --user git+https://github.com/pytorch/vision.git
  pip install -q --user ninja
  # JIT C++ extensions require ninja, so put it into PATH.
  export PATH="/var/lib/jenkins/.local/bin:$PATH"
  if [[ "$BUILD_ENVIRONMENT" == *py3* ]]; then
    # default pip version is too old(9.0.2), unable to support tag `manylinux2010`.
    # Fix the pip error: Couldn't find a version that satisfies the requirement
    sudo pip install --upgrade pip
    pip install -q --user -i https://test.pypi.org/simple/ ort-nightly==0.5.0.dev905
  fi
  "$ROOT_DIR/scripts/onnx/test.sh"
fi

