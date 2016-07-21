#include "caffe2/operators/cross_entropy_op.h"

namespace caffe2 {

template <>
bool LabelCrossEntropyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  DCHECK((label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
  DCHECK_EQ(label.dim32(0), N);
  Y->Resize(vector<TIndex>{N});
  const auto* Xdata = X.data<float>();
  const auto* labeldata = label.data<int>();
  auto* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    CAFFE_ENFORCE(
        labeldata[i] < D,
        "Label seems incorrect: label value larger than number of classes: ",
        labeldata[i], " vs ", D);
    Ydata[i] = -log(std::max(Xdata[i * D + labeldata[i]], kLOG_THRESHOLD()));
  }
  return true;
}

template <>
bool LabelCrossEntropyGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  DCHECK((label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
  DCHECK_EQ(label.dim32(0), N);
  DCHECK_EQ(dY.ndim(), 1);
  DCHECK_EQ(dY.dim32(0), N);
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(dX->size(), 0.f, dX->mutable_data<float>(),
                               &context_);
  const float* Xdata = X.data<float>();
  const float* dYdata = dY.data<float>();
  const int* labeldata = label.data<int>();
  float* dXdata = dX->mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    dXdata[i * D + labeldata[i]] =
        - dYdata[i] / std::max(Xdata[i * D + labeldata[i]], kLOG_THRESHOLD());
  }
  return true;
}

template <>
bool MakeTwoClassOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto shape = X.dims();
  shape.push_back(2);
  TIndex N = X.size();
  Y->Resize(shape);
  const auto* Xdata = X.data<float>();
  auto* Ydata = Y->mutable_data<float>();
  for (TIndex i = 0; i < N; ++i) {
    DCHECK_GE(Xdata[i], 0.0);
    DCHECK_LE(Xdata[i], 1.0);
    Ydata[i * 2] = 1.0 - Xdata[i];
    Ydata[i * 2 + 1] = Xdata[i];
  }
  return true;
}

template <>
bool MakeTwoClassGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto* dX = Output(0);
  auto shape = dY.dims();
  CHECK_GE(shape.size(), 1);
  CHECK_EQ(shape.back(), 2);
  shape.pop_back();
  dX->Resize(shape);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  TIndex N = dX->size();
  // use eigen?
  for (TIndex i = 0; i < N; ++i) {
    dXdata[i] = dYdata[i * 2 + 1] - dYdata[i * 2];
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(LabelCrossEntropy,
                      LabelCrossEntropyOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LabelCrossEntropyGradient,
                      LabelCrossEntropyGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LabelCrossEntropy)
  .NumInputs(2)
  .NumOutputs(1)
  .SetDoc(R"DOC(
Operator computes the cross entropy between the input and the label set. In
practice, it is most commonly used at the end of models, after the SoftMax
operator and before the AveragedLoss operator.
  )DOC")
  .Input(0, "X", "Input blob from the previous layer, which is almost always "
  "the result of a softmax operation.")
  .Input(1, "label", "Blob containing the labels used to compare the input")
  .Output(0, "Y", "Output blob after the cross entropy computation");
OPERATOR_SCHEMA(LabelCrossEntropyGradient)
  .NumInputs(3)
  .NumOutputs(1);

class GetLabelCrossEntropyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LabelCrossEntropyGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LabelCrossEntropy, GetLabelCrossEntropyGradient);

REGISTER_CPU_OPERATOR(MakeTwoClass,
                      MakeTwoClassOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MakeTwoClassGradient,
                      MakeTwoClassGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(MakeTwoClass)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
Given a vector of probabilities, this operator transforms this into a 2-column
matrix with complimentary probabilities for binary classification. In explicit
terms, given the vector X, the output Y is vstack(1 - X, X).
  )DOC")
  .Input(0, "X", "Input vector of probabilities")
  .Output(0, "Y", "2-column matrix with complimentary probabilities of X for "
  "binary classification");

OPERATOR_SCHEMA(MakeTwoClassGradient)
  .NumInputs(1)
  .NumOutputs(1);

struct GetMakeTwoClassGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MakeTwoClassGradient",
        "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(MakeTwoClass, GetMakeTwoClassGradient);
}  // namespace
}  // namespace caffe2
