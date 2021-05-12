#include "caffe2/operators/expand_op.h"

#include <algorithm>
#include <functional>
#include <vector>

#include <caffe2/utils/math.h>

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Expand,
    ExpandOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    ExpandGradient,
    ExpandGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Expand)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
        Broadcast the input tensor to a materialized new tensor using given shape.
        Broadcast rule is similar to "numpy.array(input) * numpy.ones(shape)":
        Dimensions are right alignment;
        Two corresponding dimensions must have the same value, or one of them
        equals to 1.
        In order to align with PyTorch's `expand`, `shape` is allowed to have entries
        equal to -1, which means to preserve the size of the corresponding dimension
        in `X` (so it's actually equivalent to equal to 1).
)DOC")
    .Input(0, "X", "(*Tensor`<NumericType>`*): input tensor")
    .Input(1, "shape", "(*Tensor`<int>`*): expand shape")
    .Output(0, "Y", "(*Tensor`<NumericType>`*): expanded tensor");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ExpandGradient).NumInputs(2).NumOutputs(1);

namespace {

class GetExpandGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ExpandGradient",
        "",
        std::vector<string>{GO(0), I(0)},
        std::vector<string>{GI(0)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Expand, GetExpandGradient);
} // namespace caffe2
