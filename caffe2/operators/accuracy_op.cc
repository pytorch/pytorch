#include "caffe2/operators/accuracy_op.h"

namespace caffe2 {

template <>
bool AccuracyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  CAFFE_ENFORCE_EQ(label.dim(), 1);
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  auto* Y = Output(0, vector<int64_t>(), at::dtype<float>());
  const auto* Xdata = X.data<float>();
  const auto* labelData = label.data<int>();
  const int top_k = top_k_;
  int correct = 0;

  // it's equivalent to using a stable sorting algorithm to sort the
  // classes (with their predictions as key) and then check whether
  // the label is within the first top_k slots.
  for (int i = 0; i < N; ++i) {
    auto label_i = labelData[i];
    auto label_pred = Xdata[i * D + label_i];
    int ngt = 1;
    for (int j = 0; j < D; ++j) {
      auto pred = Xdata[i * D + j];
      if ((pred > label_pred) || (pred == label_pred && j < label_i)) {
        if (++ngt > top_k) {
          break;
        }
      }
    }
    if (ngt <= top_k) {
      ++correct;
    }
  }
  CAFFE_ENFORCE_LE(correct, N);
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  *(Y->template mutable_data<float>()) = static_cast<float>(correct) / N;

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Accuracy, AccuracyOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Accuracy)
    .NumInputs(2)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc(R"DOC(
Accuracy takes two inputs- predictions and labels, and returns a float
accuracy value for the batch. Predictions are expected in the form of 2-D tensor
containing a batch of scores for various classes, and labels are expected in the
 form of 1-D tensor containing true label indices of samples in the batch. If
the score for the label index in the predictions is the highest among all
classes, it is considered a correct prediction.
)DOC")
    .Arg(
        "top_k",
        "Count as correct by comparing the true label to the top k scoring "
        "classes (default 1: only compare to the top scoring class i.e. argmax)")
    .Input(
        0,
        "predictions",
        "2-D tensor (Tensor<float>) of size "
        "(num_batches x num_classes) containing scores")
    .Input(
        1,
        "labels",
        "1-D tensor (Tensor<float>) of size (num_batches) having "
        "the indices of true labels")
    .Output(
        0,
        "accuracy",
        "1-D tensor (Tensor<float>) of size 1 containing "
        "accuracy");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Accuracy);
}  // namespace caffe2
