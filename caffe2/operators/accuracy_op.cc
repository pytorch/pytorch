#include "caffe2/operators/accuracy_op.h"

namespace caffe2 {

template <>
bool AccuracyOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  CAFFE_DCHECK_EQ(label.ndim(), 1);
  CAFFE_DCHECK_EQ(label.dim(0), N);
  Y->Reshape(std::vector<int>{1});
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
  CAFFE_DCHECK_LE(correct, N);
  Y->mutable_data<float>()[0] = static_cast<float>(correct) / N;
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Accuracy, AccuracyOp<float, CPUContext>);
SHOULD_NOT_DO_GRADIENT(Accuracy);
}  // namespace
}  // namespace caffe2
