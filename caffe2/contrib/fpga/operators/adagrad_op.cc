#include "caffe2/sgd/adagrad_op.h"
#include "../common_fpga.h"
#include "../context.h"
#include "../operator.h"

namespace caffe2 {

template <>
bool AdagradOp<float, OpenCLContext>::RunOnDevice() {
  return false;
}

REGISTER_OPENCL_OPERATOR_WITH_ENGINE(
    Adagrad,
    FPGA,
    AdagradOp<float, OpenCLContext>);
} // namespace caffe2
