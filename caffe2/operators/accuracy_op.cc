#include "caffe2/operators/accuracy_op.h"

namespace caffe2 {

struct PredictionCmp {
  bool operator()(
      const std::pair<float, int>& lhs,
      const std::pair<float, int>& rhs) {
    return (
        lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

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
  const auto* labelData = label.data<int>();
  const int top_k = top_k_;
  int correct = 0;

  if (top_k == 1) {
    // Specially handling the case when top_k equals to 1
    // to achieve a better performance

    for (int i = 0; i < N; ++i) {
      // Find the corresponding index of the max prediction
      float max_pred = Xdata[i * D];
      int max_idx = 0;
      for (int j = 1; j < D; j++) {
        float pred = Xdata[i * D + j];
        if (pred > max_pred) {
          max_pred = pred;
          max_idx = j;
        }
      }
      // Increment accurary if the max predictions equal to the expected label
      if (max_idx == labelData[i]) {
        ++correct;
      }
    }
  } else {
    for (int i = 0; i < N; ++i) {
      // Build a min-heap, the heap element is paire of (prediction, idx)
      // the top of the heap is the smallest prediction
      std::priority_queue<
          std::pair<float, int>,
          std::vector<std::pair<float, int>>,
          PredictionCmp>
          PQ;

      // Maintain the size of heap to be less or equal to top_k, therefore the
      // heap is holding the top k largest predictions. Increase correct by one
      // when the index j equals to labelData.
      // When poping predictions out of the heap, check if its index
      // equals to labelData. if yes, decrease correct by one and break, because
      // the next index are greater than labelData, so no need to check further
      for (int j = 0; j < D; ++j) {
        auto pred = Xdata[i * D + j];
        const auto label_data = labelData[i];

        if (PQ.size() < top_k || pred > PQ.top().first) {
          if (j == label_data) {
            ++correct;
          }
          PQ.push(std::make_pair(pred, j));
          if (PQ.size() > top_k) {
            if (PQ.top().second == label_data) {
              --correct;
              break;
            }
            PQ.pop();
          }
        } else if (label_data == j) {
          // The correct answer did not make it into the top K heap,
          // so we can short circuit
          break;
        }
      }
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
  .ScalarType(TensorProto::FLOAT)
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
