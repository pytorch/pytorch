#include "caffe2/operators/summarize_op.h"

namespace caffe2 {

template <>
bool SummarizeOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  const auto N = X.numel();
  CAFFE_ENFORCE_GT(N, 0);

  const float* Xdata = X.data<float>();
  double mean = 0;
  float max = Xdata[0];
  float min = Xdata[0];
  for (auto i = 0; i < N; ++i) {
    mean += static_cast<double>(Xdata[i]) / N;
    max = std::max(max, Xdata[i]);
    min = std::min(min, Xdata[i]);
  }
  // We will simply do a two-pass. More efficient solutions can be written but
  // I'll keep code simple for now.
  double standard_deviation = 0;
  for (auto i = 0; i < N; ++i) {
    double diff = Xdata[i] - mean;
    standard_deviation += diff * diff;
  }
  // Unbiased or biased? Let's do unbiased now.
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  standard_deviation = N == 1 ? 0 : std::sqrt(standard_deviation / (N - 1));
  if (to_file_) {
    (*log_file_) << min << " " << max << " " << mean << " "
                 << standard_deviation << std::endl;
  }
  if (OutputSize()) {
    auto* Y = Output(0, {NUM_STATS}, at::dtype<float>());
    float* Ydata = Y->template mutable_data<float>();
    Ydata[MIN_IDX] = min;
    Ydata[MAX_IDX] = max;
    Ydata[MEAN_IDX] = static_cast<float>(mean);
    Ydata[STD_IDX] = static_cast<float>(standard_deviation);
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Summarize, SummarizeOp<float, CPUContext>);

// Input: X; output: if set, a summarized Tensor of shape 4, with the values
// being min, max, mean and std respectively.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Summarize)
    .NumInputs(1)
    .NumOutputs(0, 1)
    .SetDoc(R"DOC(
Summarize computes four statistics of the input tensor (Tensor)- min,
max, mean and standard deviation. The output will be written to a 1-D tensor of
size 4 if an output tensor is provided. Else, if the argument 'to_file' is
greater than 0, the values are written to a log file in the root folder.
)DOC")
    .Arg(
        "to_file",
        "(int, default 0) flag to indicate if the summarized "
        "statistics have to be written to a log file.")
    .Input(0, "data", "The input data as Tensor.")
    .Output(
        0,
        "output",
        "1-D tensor (Tensor) of size 4 containing min, "
        "max, mean and standard deviation");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Summarize);
} // namespace caffe2
