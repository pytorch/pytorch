#!/bin/bash

set -e

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/.. && pwd)

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

export PYTHONPATH="${PYTHONPATH}:/usr/local/caffe2"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/caffe2/lib"

exit_code=0

cd "$ROOT_DIR"/caffe2/python/tutorials
python tutorials_to_script_converter.py
git status
if git diff --quiet HEAD; then
  echo "Source tree is clean."
else
  echo "After running a tutorial -> script sync there are changes. This probably means you edited an ipython notebook without a proper sync to a script. Please see caffe2/python/tutorials/README.md for more information"
  if [ "$exit_code" -eq 0 ]; then
    exit_code=1
  fi
fi

cd "$ROOT_DIR"

if [ -d ./test ]; then
  echo "Directory ./test already exists; please remove it..."
  exit 1
fi

mkdir -p ./test/{cpp,python}
TEST_DIR="$PWD/test"


cd /usr/local/caffe2

# Commands below may exit with non-zero status
set +e

# C++ tests
echo "Running C++ tests.."
for test in ./test/*; do
  # Skip tests we know are hanging or bad
  case "$(basename "$test")" in
    mkl_utils_test)
      continue
      ;;
  esac

  "$test" --gtest_output=xml:"$TEST_DIR"/cpp/$(basename "$test").xml
  tmp_exit_code="$?"
  if [ "$exit_code" -eq 0 ]; then
    exit_code="$tmp_exit_code"
  fi
done

# Figure out which Python to use
PYTHON="python"
if [ -n "$BUILD_ENVIRONMENT" ]; then
  if [[ "$BUILD_ENVIRONMENT" == py2* ]]; then
    PYTHON="python2"
  elif [[ "$BUILD_ENVIRONMENT" == py3* ]]; then
    PYTHON="python3"
  fi
fi

# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]]; then
  EXTRA_TESTS+=(caffe2/contrib/nccl)
fi

# Python tests
echo "Running Python tests.."
"$PYTHON" \
  -m pytest \
  -v \
  --junit-xml="$TEST_DIR"/python/result.xml \
  --ignore caffe2/python/test/executor_test.py \
  --ignore caffe2/python/operator_test/matmul_op_test.py \
  --ignore caffe2/python/operator_test/pack_ops_test.py \
  --ignore caffe2/python/mkl/mkl_sbn_speed_test.py \
  caffe2/python/ \
  ${EXTRA_TESTS[@]}

tmp_exit_code="$?"
if [ "$exit_code" -eq 0 ]; then
  exit_code="$tmp_exit_code"
fi

# Exit with the first non-zero status we got
exit "$exit_code"
