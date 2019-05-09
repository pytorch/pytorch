#include "caffe2/operators/cosine_embedding_criterion_op.h"

#include <algorithm>

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool CosineEmbeddingCriterionOp<CPUContext>::RunOnDevice() {
  auto& S = Input(0);
  auto& Y = Input(1);

  CAFFE_ENFORCE(
      S.numel() == Y.numel(),
      "The embedding and label should have the same size.");
  auto* output = Output(0, S.sizes(), at::dtype<float>());

  const float* Sdata = S.data<float>();
  const int* Ydata = Y.data<int>();
  float* output_data = output->template mutable_data<float>();
  for (int i = 0; i < S.numel(); ++i) {
    output_data[i] =
        Ydata[i] == 1 ? (1.f - Sdata[i]) : std::max(0.f, Sdata[i] - margin_);
  }
  return true;
}

template <>
bool CosineEmbeddingCriterionGradientOp<CPUContext>::RunOnDevice() {
  auto& S = Input(0);
  auto& Y = Input(1);
  auto& dOutput = Input(2);

  auto* dS = Output(0, S.sizes(), at::dtype<float>());

  const float* Sdata = S.data<float>();
  const int* Ydata = Y.data<int>();
  const float* dOutput_data = dOutput.data<float>();
  float* dSdata = dS->template mutable_data<float>();
  for (int i = 0; i < S.numel(); ++i) {
    dSdata[i] = dOutput_data[i] *
        (Ydata[i] == 1 ? -1.f : static_cast<float>(Sdata[i] >= margin_));
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    CosineEmbeddingCriterion,
    CosineEmbeddingCriterionOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    CosineEmbeddingCriterionGradient,
    CosineEmbeddingCriterionGradientOp<CPUContext>);

OPERATOR_SCHEMA(CosineEmbeddingCriterion)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
CosineEmbeddingCriterion takes two inputs: the similarity value and
the label, and computes the elementwise criterion output as

  output = 1 - s,               if y == 1
           max(0, s - margin),  if y == -1
)DOC")
    .Input(0, "S", "The cosine similarity as a 1-dim TensorCPU.")
    .Input(1, "Y", "The label as a 1-dim TensorCPU with int value of 1 or -1.")
    .Output(0, "loss", "The output loss with the same dimensionality as S.");

OPERATOR_SCHEMA(CosineEmbeddingCriterionGradient).NumInputs(3).NumOutputs(1);

class GetCosineEmbeddingCriterionGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CosineEmbeddingCriterionGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(
    CosineEmbeddingCriterion,
    GetCosineEmbeddingCriterionGradient);

} // namespace caffe2
