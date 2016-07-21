#include "caffe2/operators/filler_op.h"

namespace caffe2 {

template <>
bool RangeFillOp<float, CPUContext>::Fill(
    TensorCPU* output) {
  float* data = output->mutable_data<float>();
  for (int i = 0; i < output->size(); ++i) {
    data[i] = i;
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(UniformFill, UniformFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(UniformIntFill, UniformFillOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(ConstantFill, ConstantFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ConstantIntFill, ConstantFillOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(GaussianFill, GaussianFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(XavierFill, XavierFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MSRAFill, MSRAFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(RangeFill, RangeFillOp<float, CPUContext>);


OPERATOR_SCHEMA(UniformFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(UniformIntFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(ConstantFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(ConstantIntFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(GivenTensorFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(GivenTensorIntFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(GaussianFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(XavierFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(MSRAFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(RangeFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});


NO_GRADIENT(UniformFill);
NO_GRADIENT(UniformIntFill);
NO_GRADIENT(ConstantFill);
NO_GRADIENT(ConstantIntFill);
NO_GRADIENT(GivenTensorFill);
NO_GRADIENT(GivenTensorIntFill);
NO_GRADIENT(GaussianFill);
NO_GRADIENT(XavierFill);
NO_GRADIENT(MSRAFill);
NO_GRADIENT(RangeFill);

}  // namespace
}  // namespace caffe2
