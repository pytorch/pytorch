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
  auto& X = Input(XVALUE);
  auto& label = Input(LABEL);

  int N = X.dim() > 0 ? X.dim32(0) : 0;
  if (N == 0) {
    // Set correct data type for output
    Output(YVALUE, {0}, at::dtype<T>());
    return true;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const int32_t* lengths_vec;
  int len_size = 1;
  if (InputSize() > LENGTHS) {
    auto& lengths = Input(LENGTHS);
    CAFFE_ENFORCE_EQ(lengths.dim(), 1);
    len_size = lengths.numel();
    lengths_vec = lengths.template data<int32_t>();
    int len_sum = 0;
    if (len_size > 0) {
      math::Sum<int, Context>(len_size, lengths_vec, &len_sum, &context_);
    }
    CAFFE_ENFORCE_EQ(len_sum, N);
  } else {
    lengths_vec = &N;
  }

  // a total of len_size sessions
  auto* Y = Output(YVALUE, {len_size}, at::dtype<T>());
  auto* Ydata = Y->template mutable_data<T>();

  int D = X.numel() / N;
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(1, D); // only support one class at the moment

  const auto* Xdata = X.template data<T>();
  const auto* labelData = label.template data<T>();
  int offset = 0;
  for (int idx = 0; idx < len_size; ++idx) {
    Ydata[idx] = 0;
    int numPairs = 0;
    for (int i = offset; i < offset + lengths_vec[idx]; ++i) {
      for (int j = offset; j < i; ++j) {
        if (std::abs(labelData[i] - labelData[j]) <
            std::numeric_limits<T>::epsilon()) {
          continue;
        }
        ++numPairs;
        // only use sigmoid loss function at the moment
        auto sign = labelData[i] > labelData[j] ? 1 : -1;
        Ydata[idx] += logLogit(sign * (Xdata[j] - Xdata[i]));
      }
    }
    if (numPairs > 0) {
      Ydata[idx] /= numPairs;
    }
    offset += lengths_vec[idx];
  }
  return true;
}

template <class T, class Context>
bool PairWiseLossGradientOp<T, Context>::RunOnDevice() {
  auto& X = Input(XVALUE);
  auto& label = Input(LABEL);
  auto& dY = Input(DYVALUE);

  int N = X.dim() > 0 ? X.dim32(0) : 0;
  CAFFE_ENFORCE_EQ(N, X.numel());
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  auto* dX = Output(DXVALUE, X.sizes(), at::dtype<T>());
  math::Set<T, CPUContext>(
      dX->numel(), 0.f, dX->template mutable_data<T>(), &context_);

  if (N == 0) {
    return true;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const int32_t* lengths_vec;
  int len_size = 1;
  if (InputSize() > LENGTHS) {
    auto& lengths = Input(LENGTHS);
    CAFFE_ENFORCE_EQ(lengths.dim(), 1);
    len_size = lengths.numel();
    lengths_vec = lengths.template data<int32_t>();
    int len_sum = 0;
    if (len_size > 0) {
      math::Sum<int, Context>(len_size, lengths_vec, &len_sum, &context_);
    }
    CAFFE_ENFORCE_EQ(len_sum, N);
  } else {
    lengths_vec = &N;
  }

  CAFFE_ENFORCE_EQ(dY.dim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), len_size);

  const T* Xdata = X.template data<T>();
  const T* dYdata = dY.template data<T>();
  const T* labelData = label.template data<T>();
  T* dXdata = dX->template mutable_data<T>();
  int offset = 0;
  for (int idx = 0; idx < len_size; ++idx) {
    int numPairs = 0;
    for (int i = offset; i < offset + lengths_vec[idx]; ++i) {
      for (int j = offset; j < i; ++j) {
        if (std::abs(labelData[i] - labelData[j]) <
            std::numeric_limits<T>::epsilon()) {
          continue;
        }
        ++numPairs;
        // only use sigmoid loss function at the moment
        auto sign = labelData[i] > labelData[j] ? 1 : -1;
        auto grad =
            sign * dYdata[idx] / (1 + exp(-sign * (Xdata[j] - Xdata[i])));
        dXdata[i] -= grad;
        dXdata[j] += grad;
      }
    }
    if (numPairs > 0) {
      for (int i = offset; i < offset + lengths_vec[idx]; ++i) {
        dXdata[i] /= numPairs;
      }
    }
    offset += lengths_vec[idx];
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(PairWiseLoss, PairWiseLossOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    PairWiseLossGradient,
    PairWiseLossGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(PairWiseLoss)
    .NumInputs(2, 3)
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
    .Input(
        2,
        "lengths",
        "Optional input blob that contains the lengths"
        "of multiple sessions. The summation of this blob must be equal"
        "to the size of blob X. If lengths blob is provided, the output"
        "blob has the same size as lengths blob, and the cross entropy"
        "is computed within each session.")
    .Output(0, "Y", "Output blob after the cross entropy computation");
OPERATOR_SCHEMA(PairWiseLossGradient).NumInputs(3, 4).NumOutputs(1);

class GetPairWiseLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> blob_names{I(0), I(1), GO(0)};

    // Add lengths blob if given
    if (def_.input_size() == 3) {
      blob_names.push_back(I(2));
    }
    return SingleGradientDef(
        "PairWiseLossGradient", "", blob_names, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(PairWiseLoss, GetPairWiseLossGradient);

} // namespace
} // namespace caffe2
