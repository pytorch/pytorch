#include "caffe2/operators/square_root_divide_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SquareRootDivide, SquareRootDivideOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SquareRootDivide)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Given DATA tensor with first dimension N and SCALE vector of the same size N
produces an output tensor with same dimensions as DATA. Which consists of DATA
slices. i-th slice is divided by sqrt(SCALE[i]) elementwise. If SCALE[i] == 0
output slice is identical to the input one (no scaling)

Example:

  Data = [
    [2.0, 4.0],
    [9.0, 12.0]
  ]

  SCALE = [4, 9]

  OUTPUT = [
    [1.0, 2.0],
    [3.0, 4.0]
  ]

)DOC");

class GetSquareRootDivideGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SquareRootDivide",
        "",
        vector<string>{GO(0), I(1)},
        vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(SquareRootDivide, GetSquareRootDivideGradient);
} // namespace caffe2
