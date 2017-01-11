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
  const auto* labelData = label.data<int>();
  const int top_k = top_k_;
  int correct = 0;
  for (int i = 0; i < N; ++i) {
    // Make a vector of pairs(prediction, index) so that 
    // the index of elements can be extracted after sort.
    // top-k algorithm rewritten based on algorithm in 
    // Caffe accuracy layer
    std::vector<std::pair<float, int> > Xdata_pairs;
    for (int j = 0; j < D; ++j) {
      Xdata_pairs.push_back(std::make_pair(Xdata[i * D + j], j));
    }
    // Sort so that the k maximum predictions appear 
    // at the beginning of vector.
    std::partial_sort(
        Xdata_pairs.begin(),
        Xdata_pairs.begin() + top_k,
        Xdata_pairs.end(),
        [](std::pair<float, int> lhs, std::pair<float, int> rhs) {
            if(lhs.first == rhs.first) {
                return lhs.second < rhs.second;
            }   
            else {
                return lhs.first > rhs.first;
            }   
        });
    // Increment accuracy if any of the top k predictions 
    // are equal to the expected label.
    for (int k = 0; k < top_k; k++) { 
      if (Xdata_pairs[k].second == labelData[i]) {                                               
        ++correct;
        break;
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
