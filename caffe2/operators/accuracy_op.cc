#include "caffe2/operators/accuracy_op.h"

namespace caffe2 {

template <>
bool AccuracyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  Y->Resize(vector<TIndex>());
  const auto* Xdata = X.data<float>();
  const auto* labeldata = label.data<int>();
  int correct = 0;
  for (int i = 0; i < N; ++i) {
    float maxval = std::numeric_limits<float>::lowest();
    int maxid = 0;
    for (int j = 0; j < D; ++j) {
      if (Xdata[i * D + j] > maxval) {
        maxval = Xdata[i * D + j];
        maxid = j;
      }
    }
    if (maxid == labeldata[i]) {
      ++correct;
    }
  }
  DCHECK_LE(correct, N);
  *(Y->mutable_data<float>()) = static_cast<float>(correct) / N;
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Accuracy, AccuracyOp<float, CPUContext>);

OPERATOR_SCHEMA(Accuracy)
  .NumInputs(2)
  .NumOutputs(1)
  .SetDoc(R"DOC(
Accuracy takes two inputs- predictions and labels, and returns a float
accuracy value for the batch. Predictions are expected in the form of 2-D tensor
containing a batch of scores for various classes, and labels are expected in the
 form of 1-D tensor containing true label indices of samples in the batch. If
the score for the label index in the predictions is the highest among all
classes, it is considered a correct prediction.
)DOC")
  .Input(0, "predictions", "2-D tensor (Tensor<float>) of size "
         "(num_batches x num_classes) containing scores")
  .Input(1, "labels", "1-D tensor (Tensor<int>) of size (num_batches) having "
        "the indices of true labels")
  .Output(0, "accuracy", "1-D tensor (Tensor<float>) of size 1 containing "
          "accuracy");

SHOULD_NOT_DO_GRADIENT(Accuracy);
}  // namespace
}  // namespace caffe2
