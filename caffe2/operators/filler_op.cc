#include "caffe2/operators/filler_op.h"

namespace caffe2 {

template <>
bool RangeFillOp<float, CPUContext>::Fill(
    Tensor<float, CPUContext>* output) {
  float* data = output->mutable_data();
  for (int i = 0; i < output->size(); ++i) {
    data[i] = i;
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(UniformFill, UniformFillOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(ConstantFill, ConstantFillOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(GaussianFill, GaussianFillOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(XavierFill, XavierFillOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(RangeFill, RangeFillOp<float, CPUContext>)

}  // namespace
}  // namespace caffe2
