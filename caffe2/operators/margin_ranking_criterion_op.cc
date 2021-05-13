#include "caffe2/operators/margin_ranking_criterion_op.h"

#include <algorithm>

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool MarginRankingCriterionOp<CPUContext>::RunOnDevice() {
  auto& X1 = Input(0);
  auto& X2 = Input(1);
  auto& Y = Input(2);

  CAFFE_ENFORCE_EQ(
      X1.numel(),
      X2.numel(),
      "The two inputs for computing ranking loss should have the same size.");
  CAFFE_ENFORCE_EQ(
      X1.numel(), Y.numel(), "The input and label should have the same size.");
  auto* loss = Output(0, X1.sizes(), at::dtype<float>());

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  float* output = loss->template mutable_data<float>();
  for (int i = 0; i < X1.numel(); ++i) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    output[i] = std::max(-Ydata[i] * (X1data[i] - X2data[i]) + margin_, 0.f);
  }
  return true;
}

template <>
bool MarginRankingCriterionGradientOp<CPUContext>::RunOnDevice() {
  auto& X1 = Input(0);
  auto& X2 = Input(1);
  auto& Y = Input(2);
  auto& dLoss = Input(3);

  auto* dX1 = Output(0, X1.sizes(), at::dtype<float>());
  auto* dX2 = Output(1, X2.sizes(), at::dtype<float>());

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  const float* dLoss_data = dLoss.data<float>();

  float* dX1_data = dX1->template mutable_data<float>();
  float* dX2_data = dX2->template mutable_data<float>();
  for (int i = 0; i < X1.numel(); ++i) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    auto dist = -Ydata[i] * (X1data[i] - X2data[i]) + margin_;
    if (dist < 0.f) {
      dX1_data[i] = dX2_data[i] = 0.f;
    } else {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      dX1_data[i] = -Ydata[i] * dLoss_data[i];
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      dX2_data[i] = Ydata[i] * dLoss_data[i];
    }
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MarginRankingCriterion,
    MarginRankingCriterionOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MarginRankingCriterionGradient,
    MarginRankingCriterionGradientOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MarginRankingCriterion)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
MarginRankingCriterion takes two input data X1 (Tensor),
X2 (Tensor), and label Y (Tensor) to produce the
loss (Tensor) where the loss function,
loss(X1, X2, Y) = max(0, -Y * (X1 - X2) + margin), is applied to
the tensor elementwise.

If y == 1 then it assumed the first input should be ranked higher
(have a larger value) than the second input, and vice-versa for
y == -1.
)DOC")
    .Arg("margin", "The margin value as a float. Default is 1.0.")
    .Input(0, "X1", "The left input vector as a 1-dim TensorCPU.")
    .Input(1, "X2", "The right input vector as a 1-dim TensorCPU.")
    .Input(2, "Y", "The label as a 1-dim TensorCPU with int value of 1 or -1.")
    .Output(0, "loss", "The output loss with the same dimensionality as X1.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MarginRankingCriterionGradient)
    .NumInputs(4)
    .NumOutputs(2)
    .SetDoc(R"DOC(
MarginRankingCriterionGradient takes both X1, X2, Y and dY and
uses them to update dX1, and dX2 according to the chain rule
and derivatives of the loss function.
)DOC");

class GetMarginRankingCriterionGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MarginRankingCriterionGradient",
        "",
        vector<string>{I(0), I(1), I(2), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(MarginRankingCriterion, GetMarginRankingCriterionGradient);

} // namespace caffe2
