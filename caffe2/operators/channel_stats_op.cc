#include "caffe2/operators/channel_stats_op.h"

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <>
bool ChannelStatsOp<CPUContext>::ComputeChannelStatsNCHW<float>(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* sum,
    float* sumsq) {
  ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
  for (int i = 0; i < C; ++i) {
    sum[i] = X_arr.col(i).sum();
    sumsq[i] = X_arr.col(i).square().sum();
  }
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      const int c = i * C + j;
      sum[j] += X_arr.col(c).sum();
      sumsq[j] += X_arr.col(c).square().sum();
    }
  }
  return true;
}

template <>
template <>
bool ChannelStatsOp<CPUContext>::ComputeChannelStatsNHWC<float>(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* sum,
    float* sumsq) {
  ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
  EigenVectorArrayMap<float> sum_arr(sum, C);
  EigenVectorArrayMap<float> sumsq_arr(sumsq, C);
  sum_arr = X_arr.col(0);
  sumsq_arr = X_arr.col(0).square();
  for (int i = 1; i < N * HxW; ++i) {
    sum_arr += X_arr.col(i);
    sumsq_arr += X_arr.col(i).square();
  }
  return true;
}

REGISTER_CPU_OPERATOR(ChannelStats, ChannelStatsOp<CPUContext>);

OPERATOR_SCHEMA(ChannelStats)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Given an input tensor in NCHW format, computes the sum of all elements per
channel and the sum of all elements squared per channel. These values can be
reduced across multiple batches and used to obtain the mean and variance across
the full set of batches. Using the new mean and variance as input to SpatialBN
has the effect of changing the batch size over which SpatialBN is applied.
)DOC")
    .Input(0, "X", "The input 4-dimensional tensor of shape NCHW")
    .Output(
        0,
        "sum",
        "The output 1-dimensional tensor of size C containing the sum of "
        "elements of X per channel.")
    .Output(
        1,
        "sumsq",
        "The output 1-dimensional tensor of size C containing the sum of "
        "elements squared per channel.");

SHOULD_NOT_DO_GRADIENT(ChannelStats);

} // namespace caffe2
