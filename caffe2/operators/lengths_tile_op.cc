#include "caffe2/operators/lengths_tile_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(LengthsTile, LengthsTileOp<CPUContext>);

OPERATOR_SCHEMA(LengthsTile)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given DATA tensor of rank r >= 1, and LENGTHS tensor of rank 1, duplicate each
entry of the outer-most dimension of DATA according to LENGTHS, and concatenate
them in an output tensor of rank r.

Example:
  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
      [6.8, 7.9],
  ]
  LENGTHS = [0, 1, 3, 2]
  OUTPUT = [
      [2.3, 3.4],
      [4.5, 5.7],
      [4.5, 5.7],
      [4.5, 5.7],
      [6.8, 7.9],
      [6.8, 7.9],
  ]
)DOC")
    .Input(
        0,
        "DATA",
        "Tensor of rank r >= 1. First dimension must be equal to the size of "
        "lengths")
    .Input(1, "LENGTHS", "Tensor of int32 lengths of rank 1")
    .Output(0, "OUTPUT", "Tensor of rank r");

class GetLengthsTileGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);
    return SingleGradientDef(
        "LengthsSum",
        "",
        // input 1 is the lengths used to repeat
        // DATA in the forward pass
        vector<string>{GO(0), I(1)},
        // only concerned with the gradient on "DATA"
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LengthsTile, GetLengthsTileGradient);
} // namespace caffe2
