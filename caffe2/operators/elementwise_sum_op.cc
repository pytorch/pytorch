#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

namespace {
OpSchema::Cost CostInferenceForSum(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<1>(def, in);
  cost.flops *= (in.size() - 1);
  cost.params_bytes = 0;
  return cost;
}
} // namespace

REGISTER_CPU_OPERATOR(Sum, SumOp<CPUContext>);

OPERATOR_SCHEMA(Sum)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForSum)
    .InputsCanCrossDevices()
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the sum will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "sum", "Output tensor. Same dimension as inputs.")
    .InheritOnnxSchema("Sum");
}
