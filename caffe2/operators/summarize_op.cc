#include "caffe2/operators/summarize_op.h"

namespace caffe2 {

template<>
bool SummarizeOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  const int N = X.size();
  CAFFE_DCHECK_GT(N, 0);
  const float* Xdata = X.data<float>();
  float mean = 0;
  float max = Xdata[0];
  float min = Xdata[0];
  for (int i = 0; i < N; ++i) {
    mean += Xdata[i];
    max = std::max(max, Xdata[i]);
    min = std::min(min, Xdata[i]);
  }
  mean /= N;
  // We will simply do a two-pass. More efficient solutions can be written but
  // I'll keep code simple for now.
  float standard_deviation = 0;
  for (int i = 0; i < N; ++i) {
    float diff = Xdata[i] - mean;
    standard_deviation += diff * diff;
  }
  // Unbiased or biased? Let's do unbiased now.
  standard_deviation = N == 1 ? 0 : std::sqrt(standard_deviation / (N - 1));
  if (to_file_) {
    (*log_file_) << min << " " << max << " " << mean << " "
                 << standard_deviation << std::endl;
  }
  if (OutputSize()) {
    auto* Y = Output(0);
    Y->Reshape(std::vector<int>{NUM_STATS});
    float* Ydata = Y->mutable_data<float>();
    Ydata[MIN_IDX] = min;
    Ydata[MAX_IDX] = max;
    Ydata[MEAN_IDX] = mean;
    Ydata[STD_IDX] = standard_deviation;
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Summarize, SummarizeOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2
