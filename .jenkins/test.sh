#!/bin/bash

set -e

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$LOCAL_DIR")

cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH}:/usr/local/caffe2"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/caffe2/lib"

rm -rf test
mkdir -p test/{cpp,python}
TEST_DIR="$ROOT_DIR/test"

cd /usr/local/caffe2

# Commands below may exit with non-zero status
set +e
exit_code=0

# C++ tests
echo "Running C++ tests.."
for test in ./test/*; do
  # Skip tests we know are hanging or bad
  case "$(basename "$test")" in
    net_test)
      continue
      ;;
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

# Python tests
echo "Running Python tests.."
python \
  -m pytest \
  -v \
  --junit-xml="$TEST_DIR"/python/result.xml \
  --ignore caffe2/python/test/executor_test.py \
  --ignore caffe2/python/operator_test/matmul_op_test.py \
  --ignore caffe2/python/operator_test/rnn_cell_test.py \
  --ignore caffe2/python/mkl/mkl_sbn_op_test.py \
  --ignore caffe2/python/mkl/mkl_sbn_speed_test.py \
  caffe2/python/
tmp_exit_code="$?"
if [ "$exit_code" -eq 0 ]; then
  exit_code="$tmp_exit_code"
fi

# Exit with the first non-zero status we got
exit "$exit_code"
