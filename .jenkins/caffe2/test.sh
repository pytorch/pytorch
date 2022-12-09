#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ${BUILD_ENVIRONMENT} == *onnx* ]]; then
  pip install click mock tabulate networkx==2.0
  pip -q install --user "file:///var/lib/jenkins/workspace/third_party/onnx#egg=onnx"
fi

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi
if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
  # temporary to locate some kernel issues on the CI nodes
  export HSAKMT_DEBUG_LEVEL=4
fi
# These additional packages are needed for circleci ROCm builds.
if [[ $BUILD_ENVIRONMENT == *rocm* ]]; then
    # Need networkx 2.0 because bellmand_ford was moved in 2.1 . Scikit-image by
    # defaults installs the most recent networkx version, so we install this lower
    # version explicitly before scikit-image pulls it in as a dependency
    pip install networkx==2.0
    # click - onnx
    pip install --progress-bar off click protobuf tabulate virtualenv mock typing-extensions
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
# Only run cpp tests in the first shard, don't run cpp tests a second time in the second shard
if [[ "${SHARD_NUMBER:-1}" == "1" ]]; then
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
      */*_benchmark)
        LD_LIBRARY_PATH="$ld_library_path" "$test" --benchmark_color=false
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
fi

################################################################################
# Python tests #
################################################################################
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  exit 0
fi

# If pip is installed as root, we must use sudo.
# CircleCI docker images could install conda as jenkins user, or use the OS's python package.
PIP=$(which pip)
PIP_USER=$(stat --format '%U' $PIP)
CURRENT_USER=$(id -u -n)
if [[ "$PIP_USER" = root && "$CURRENT_USER" != root ]]; then
  MAYBE_SUDO=sudo
fi

# Uninstall pre-installed hypothesis and coverage to use an older version as newer
# versions remove the timeout parameter from settings which ideep/conv_transpose_test.py uses
$MAYBE_SUDO pip -q uninstall -y hypothesis
$MAYBE_SUDO pip -q uninstall -y coverage

# "pip install hypothesis==3.44.6" from official server is unreliable on
# CircleCI, so we host a copy on S3 instead
$MAYBE_SUDO pip -q install attrs==18.1.0 -f https://s3.amazonaws.com/ossci-linux/wheels/attrs-18.1.0-py2.py3-none-any.whl
$MAYBE_SUDO pip -q install coverage==4.5.1 -f https://s3.amazonaws.com/ossci-linux/wheels/coverage-4.5.1-cp36-cp36m-macosx_10_12_x86_64.whl
$MAYBE_SUDO pip -q install hypothesis==3.44.6 -f https://s3.amazonaws.com/ossci-linux/wheels/hypothesis-3.44.6-py3-none-any.whl

# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]] || [[ "$BUILD_ENVIRONMENT" == *-rocm* ]]; then
  EXTRA_TESTS+=("$caffe2_pypath/contrib/nccl")
fi

rocm_ignore_test=()
if [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
  # Currently these tests are failing on ROCM platform:

  # On ROCm, RCCL (distributed) development isn't complete.
  # https://github.com/ROCmSoftwarePlatform/rccl
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/data_parallel_model_test.py")

  # This test has been flaky in ROCm CI (but note the tests are
  # cpu-only so should be unrelated to ROCm)
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/operator_test/blobs_queue_db_test.py")
  # This test is skipped on Jenkins(compiled without MKL) and otherwise known flaky
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/ideep/convfusion_op_test.py")
  # This test is skipped on Jenkins(compiled without MKL) and causing segfault on Circle
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/ideep/pool_op_test.py")
fi

echo "Running Python tests.."
# locale setting is required by click package
for loc in "en_US.utf8" "C.UTF-8"; do
  if locale -a | grep "$loc" >/dev/null 2>&1; then
    export LC_ALL="$loc"
    export LANG="$loc"
    break;
  fi
done

# Some Caffe2 tests fail when run using AVX512 ISA, see https://github.com/pytorch/pytorch/issues/66111
export DNNL_MAX_CPU_ISA=AVX2

# Should still run even in the absence of SHARD_NUMBER
if [[ "${SHARD_NUMBER:-1}" == "1" ]]; then
  # TODO(sdym@meta.com) remove this when the linked issue resolved.
  # py is temporary until https://github.com/Teemu/pytest-sugar/issues/241 is fixed
  pip install --user py==1.11.0
  pip install --user pytest-sugar
  # NB: Warnings are disabled because they make it harder to see what
  # the actual erroring test is
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
    --ignore "$caffe2_pypath/python/trt/test_pt_onnx_trt.py" \
    ${rocm_ignore_test[@]} \
    "$caffe2_pypath/python" \
    "${EXTRA_TESTS[@]}"
fi
