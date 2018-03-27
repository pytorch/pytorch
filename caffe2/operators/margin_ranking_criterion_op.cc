#include "caffe2/operators/margin_ranking_criterion_op.h"

#include <algorithm>

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool MarginRankingCriterionOp<CPUContext>::RunOnDevice() {
  auto& X1 = Input(0);
  auto& X2 = Input(1);
  auto& Y = Input(2);
  auto* loss = Output(0);
  CAFFE_ENFORCE(
      X1.size() == X2.size(),
      "The two inputs for computing ranking loss should have the same size.");
  CAFFE_ENFORCE(
      X1.size() == Y.size(), "The input and label should have the same size.");
  loss->ResizeLike(X1);

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  float* output = loss->mutable_data<float>();
  for (int i = 0; i < X1.size(); ++i) {
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
  auto* dX1 = Output(0);
  auto* dX2 = Output(1);

  dX1->ResizeLike(X1);
  dX2->ResizeLike(X2);

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  const float* dLoss_data = dLoss.data<float>();

  float* dX1_data = dX1->mutable_data<float>();
  float* dX2_data = dX2->mutable_data<float>();
  for (int i = 0; i < X1.size(); ++i) {
    auto dist = -Ydata[i] * (X1data[i] - X2data[i]) + margin_;
    if (dist < 0.f) {
      dX1_data[i] = dX2_data[i] = 0.f;
    } else {
      dX1_data[i] = -Ydata[i] * dLoss_data[i];
      dX2_data[i] = Ydata[i] * dLoss_data[i];
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    MarginRankingCriterion,
    MarginRankingCriterionOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    MarginRankingCriterionGradient,
    MarginRankingCriterionGradientOp<CPUContext>);

OPERATOR_SCHEMA(MarginRankingCriterion)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
MarginRankingCriterion takes two input data X1 (Tensor<float>),
X2 (Tensor<float>), and label Y (Tensor<int>) to produce the
loss (Tensor<float>) where the loss function,
loss(X1, X2, Y) = max(0, -Y * (X1 - X2) + margin), is applied to
the tensor elementwise.

If y == 1 then it assumed the first input should be ranked higher
(have a larger value) than the second input, and vice-versa for
y == -1.
)DOC")
    .Input(0, "X1", "The left input vector as a 1-dim TensorCPU.")
    .Input(1, "X2", "The right input vector as a 1-dim TensorCPU.")
    .Input(2, "Y", "The label as a 1-dim TensorCPU with int value of 1 or -1.")
    .Output(0, "loss", "The output loss with the same dimensionality as X1.");

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
REGISTER_GRADIENT(MarginRankingCriterion, GetMarginRankingCriterionGradient);

} // namespace caffe2
