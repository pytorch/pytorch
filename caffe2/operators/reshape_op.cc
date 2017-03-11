#include "caffe2/operators/reshape_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Reshape, ReshapeOp<float, CPUContext>);

OPERATOR_SCHEMA(Reshape)
    .NumInputs(1, 2)
    .NumOutputs(2)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Reshape the input tensor similar to numpy.reshape.

It takes a tensor as input and an optional tensor specifying the new shape.
When the second input is absent, an extra argument `shape` must be specified.
It outputs the reshaped tensor as well as the original shape.

At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is going to be copied
from the input tensor.
)DOC")
    .Arg("shape", "New shape")
    .Input(0, "data", "An input tensor.")
    .Input(1, "new_shape", "New shape.")
    .Output(0, "reshaped", "Reshaped data.")
    .Output(1, "old_shape", "Original shape.");

class GetReshapeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Reshape",
        "",
        vector<string>{GO(0), O(1)},
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }

  // Argument `shape` is no longer needed in backprop.
  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(Reshape, GetReshapeGradient);

} // namespace

} // namespace caffe2
