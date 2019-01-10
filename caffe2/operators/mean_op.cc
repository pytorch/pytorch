#include "caffe2/operators/mean_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Mean, MeanOp<CPUContext>);
REGISTER_CPU_OPERATOR(MeanGradient, MeanGradientOp<CPUContext>);

OPERATOR_SCHEMA(Mean)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise mean of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the mean will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "mean", "Output tensor. Same dimension as inputs.");

class GetMeanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto outputs = std::vector<string>();
    for (int i = 0; i < def_.input_size(); i++) {
      outputs.push_back(GI(i));
    }
    return SingleGradientDef(
        "MeanGradient", "", std::vector<string>{GO(0)}, outputs);
  }
};

REGISTER_GRADIENT(Mean, GetMeanGradient);

OPERATOR_SCHEMA(MeanGradient)
    .NumInputs(1)
    .NumOutputs(1, INT_MAX)
    .AllowInplace({{0, 0}});

} // namespace caffe2
