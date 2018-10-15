#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../../../ && pwd)

cd "$ROOT_DIR"

# Get the relative path to where the caffe2 python module was installed
CAFFE2_PYPATH="$ROOT_DIR/build_caffe2/caffe2"

rocm_ignore_test=()
# need to debug
rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/arg_ops_test.py")
rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/piecewise_linear_transform_test.py")
rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/unique_ops_test.py")
rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/model_device_test.py")
rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/data_parallel_model_test.py")

# Need to go through roi ops to replace max(...) with fmaxf(...)
rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/roi_align_rotated_op_test.py")

# cuda top_k op has some asm code, the hipified version doesn't
# compile yet, so we don't have top_k operator for now
rocm_ignore_test+=("--ignore $CAFFE2_PYPATH/python/operator_test/top_k_test.py")


# Python tests
echo "Running Python tests.."
python \
  -m pytest \
  -v \
  --ignore "$CAFFE2_PYPATH/python/test/executor_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/matmul_op_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/pack_ops_test.py" \
  --ignore "$CAFFE2_PYPATH/python/mkl/mkl_sbn_speed_test.py" \
  ${rocm_ignore_test[@]} \
  "$CAFFE2_PYPATH/python" 