#include "caffe2/operators/loss_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AveragedLoss, AveragedLoss<float, CPUContext>);
REGISTER_CPU_OPERATOR(AveragedLossGradient,
                      AveragedLossGradient<float, CPUContext>);

OPERATOR_SCHEMA(AveragedLoss)
  .NumInputs(1)
  .NumOutputs(1)
  .ScalarType(TensorProto::FLOAT)
  .SetDoc(R"DOC(
AveragedLoss takes in a 1-D tensor as input and returns a single output float
value which represents the average of input data (average of the losses).
)DOC")
  .Input(0, "input", "The input data as Tensor")
  .Output(0, "output", "The output tensor of size 1 containing the averaged "
          "value.");

OPERATOR_SCHEMA(AveragedLossGradient).NumInputs(2).NumOutputs(1);

class GetAveragedLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AveragedLossGradient", "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(AveragedLoss, GetAveragedLossGradient);

}  // namespace caffe2
