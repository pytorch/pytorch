#include "caffe2/operators/rank_loss_op.h"

namespace caffe2 {

namespace {

// Computes log(1 + exp(y)) in a way that avoids early over-/under-flow
template <class T>
inline T logLogit(T x) {
  static const auto kMinLogDiff = std::log(std::numeric_limits<T>::epsilon());

  if (x < kMinLogDiff) {
    return 0;
  }
  if (x > -kMinLogDiff) {
    return x;
  }
  return std::log(std::exp(x) + 1);
}
}

template <typename T, class Context>
bool PairWiseLossOp<T, Context>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto* Y = Output(0);

  Y->Resize(1); // assumes all the rows represent documents which belongs to the
  // same session
  auto* Ydata = Y->template mutable_data<T>();
  Ydata[0] = 0;

  int N = X.ndim() > 0 ? X.dim32(0) : 0;
  if (N == 0) {
    return true;
  }
  int D = X.size() / N;
  CAFFE_ENFORCE(
      (label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(1, D); // only support one class at the moment

  const auto* Xdata = X.template data<T>();
  const auto* labelData = label.template data<T>();
  int numPairs = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      if (std::abs(labelData[i] - labelData[j]) <
          std::numeric_limits<T>::epsilon()) {
        continue;
      }
      ++numPairs;
      // only use sigmoid loss function at the moment
      auto sign = labelData[i] > labelData[j] ? 1 : -1;
      Ydata[0] += logLogit(sign * (Xdata[j] - Xdata[i]));
    }
  }
  if (numPairs > 0) {
    Ydata[0] /= numPairs;
  }
  return true;
}

template <class T, class Context>
bool PairWiseLossGradientOp<T, Context>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  int N = X.ndim() > 0 ? X.dim32(0) : 0;
  CAFFE_ENFORCE(
      (label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(dY.ndim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), 1);
  CAFFE_ENFORCE_EQ(N, X.size());
  dX->ResizeLike(X);
  math::Set<T, CPUContext>(
      dX->size(), 0.f, dX->template mutable_data<T>(), &context_);
  if (N == 0) {
    return true;
  }
  const T* Xdata = X.template data<T>();
  const T* dYdata = dY.template data<T>();
  const T* labelData = label.template data<T>();
  T* dXdata = dX->template mutable_data<T>();
  int numPairs = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      if (std::abs(labelData[i] - labelData[j]) <
          std::numeric_limits<T>::epsilon()) {
        continue;
      }
      ++numPairs;
      // only use sigmoid loss function at the moment
      auto sign = labelData[i] > labelData[j] ? 1 : -1;
      auto grad = sign * dYdata[0] / (1 + exp(-sign * (Xdata[j] - Xdata[i])));
      dXdata[i] -= grad;
      dXdata[j] += grad;
    }
  }
  if (numPairs > 0) {
    for (int i = 0; i < N; ++i) {
      dXdata[i] /= numPairs;
    }
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(PairWiseLoss, PairWiseLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    PairWiseLossGradient,
    PairWiseLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(PairWiseLoss)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Operator computes the pair wise loss between all pairs within a batch
 using the logit loss function on the difference in scores between pairs
)DOC")
    .Input(
        0,
        "X",
        "Input blob from the previous layer, which is almost always "
        "the result of a softmax operation; X is a 2D array of size N x 1"
        "where N is the batch size. For more info: "
        "D. Sculley, Large Scale Learning to Rank. "
        "https://www.eecs.tufts.edu/~dsculley/papers/large-scale-rank.pdf")
    .Input(1, "label", "Blob containing the labels used to compare the input")
    .Output(0, "Y", "Output blob after the cross entropy computation");
OPERATOR_SCHEMA(PairWiseLossGradient).NumInputs(3).NumOutputs(1);

class GetPairWiseLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PairWiseLossGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(PairWiseLoss, GetPairWiseLossGradient);

} // namespace
} // namespace caffe2
