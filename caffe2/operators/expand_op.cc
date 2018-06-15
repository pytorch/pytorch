#include "caffe2/operators/expand_op.h"

#include <algorithm>
#include <functional>
#include <vector>

#include <caffe2/utils/math.h>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Expand,
    ExpandOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext>);

REGISTER_CPU_OPERATOR(
    ExpandGradient,
    ExpandGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext>);

OPERATOR_SCHEMA(Expand)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
	Write Me
)DOC")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Input(1, "shape", "(*Tensor`<int>`*): expand shape")
    .Output(0, "Y", "(*Tensor`<float>`*): expanded tensor");

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

REGISTER_GRADIENT(Expand, GetExpandGradient);
} // namespace caffe2
