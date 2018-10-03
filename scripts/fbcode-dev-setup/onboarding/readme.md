# Onboarding Example

Through this onboarding lab, you can get a sense of how to write
an operator for caffe2.

## Step 0: set up your development envirnoment

Probably try `https://github.com/pytorch/pytorch/blob/master/scripts/fbcode-dev-setup/onnx_c2_setup.sh`

## Step 1: starting with the templates

Copy `add5_op.h`, `add5_op.cc` to folder `caffe2/operator/`

## Step 2: fill the missing the logic for CPU part and build

Find the TODO mark in `add5_op.cc`, and fill the code you think it will work
After that, build PyTorch using `python setup.py build_deps develop`

You can find some sample implementation in `sample/sample.md`

## Step 3: run `Add5` and `Add5Gradient` operator and test it

You can run the `Add5` operator using `python scripts/fbcode-dev-setup/onboarding/run_add5_op.py`

You can follow examples in `caffe2/python/operator_test/` to write test
code for `Add5` and `Add5Gradient` ops. There are three important functions to use:
-- assertDeviceChecks
-- assertGradientChecks
-- assertReferenceChecks

Find more details in the [source code](https://github.com/pytorch/pytorch/blob/master/caffe2/python/hypothesis_test_util.py)

## Step 4: fill the missing logic for CUDA part and build

Find the TODO mark in `add5_op.cu`, and fill the code you think it will work
After that, build PyTorch using `python setup.py build_deps develop`

You can find some sample implementation in `solution/solution.md`

## Step 5: run `Add5` and `Add5Gradient` CUDA version and test it

For hypothesis test, use `hu.gcs` instead of `hu.gcs_cpu_only`

For `CreateOperator`, pass `caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)` as
named parameter `device_option`


## Further Reading
https://caffe2.ai/docs/custom-operators.html
