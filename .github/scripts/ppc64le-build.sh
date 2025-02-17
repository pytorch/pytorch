#!/usr/bin/env bash

# Environment variables
PACKAGE_NAME=pytorch

cd /workspace/$PACKAGE_NAME

# Clean up old artifacts
rm -rf build/ dist/ torch.egg-info/

# Build and install PyTorch wheel
if ! (MAX_JOBS=$(nproc) python setup.py bdist_wheel && pip install dist/*.whl); then
    echo "------------------$PACKAGE_NAME:install_fails-------------------------------------"
    exit 1
fi

# register PrivateUse1HooksInterface
python test/test_utils.py TestDeviceUtilsCPU.test_device_mode_ops_sparse_mm_reduce_cpu_bfloat16
python test/test_utils.py TestDeviceUtilsCPU.test_device_mode_ops_sparse_mm_reduce_cpu_float16
python test/test_utils.py TestDeviceUtilsCPU.test_device_mode_ops_sparse_mm_reduce_cpu_float32
python test/test_utils.py TestDeviceUtilsCPU.test_device_mode_ops_sparse_mm_reduce_cpu_float64

cd ..
pip install pytest pytest-xdist

if ! pytest "$PACKAGE_NAME/test/test_utils.py"; then
    echo "------------------$PACKAGE_NAME:install_success_but_test_fails---------------------"
    exit 2
    
else
    echo "------------------$PACKAGE_NAME:install_and_test_both_success-------------------------"
    exit 0
fi