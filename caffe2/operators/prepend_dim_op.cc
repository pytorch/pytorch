#include "caffe2/operators/prepend_dim_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(PrependDim, PrependDimOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(MergeDim, MergeDimOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(PrependDim)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Reshape the tensor by prepending a dimension of fixed size and dividing the
size of the next dimension by that amount.
)DOC")
    .Arg("dim_size", "Size of the dimension to prepend.")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reshaped", "Reshaped tensor.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MergeDim)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Merge first two dimensions in a single dimension with size dim(0) * dim(1).
)DOC")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reshaped", "Reshaped tensor.")
    .InheritOnnxSchema("Reshape");

class GetPrependDimGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MergeDim", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }

  // Arguments are no longer needed in backprop.
  bool CopyArguments() const override {
    return false;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(PrependDim, GetPrependDimGradient);

} // namespace caffe2
