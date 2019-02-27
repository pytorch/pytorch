#include "caffe2/operators/channel_backprop_stats_op.h"

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <>
bool ChannelBackpropStatsOp<CPUContext>::ChannelStatsBackwardNCHW<float>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    float* dscale,
    float* dbias) {
  ConstEigenArrayMap<float> dY_arr(dY, HxW, N * C);
  ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
  ConstEigenVectorArrayMap<float> mean_arr(mean, C);
  ConstEigenVectorArrayMap<float> rstd_arr(rstd, C);
  EigenVectorArrayMap<float> dscale_arr(dscale, C);
  EigenVectorArrayMap<float> dbias_arr(dbias, C);
  for (int i = 0; i < C; ++i) {
    dscale_arr(i) = (dY_arr.col(i) * X_arr.col(i)).sum();
    dbias_arr(i) = dY_arr.col(i).sum();
  }
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      const int c = i * C + j;
      dscale_arr(j) += (dY_arr.col(c) * X_arr.col(c)).sum();
      dbias_arr(j) += dY_arr.col(c).sum();
    }
  }
  dscale_arr = (dscale_arr - mean_arr * dbias_arr) * rstd_arr;
  return true;
}

template <>
template <>
bool ChannelBackpropStatsOp<CPUContext>::ChannelStatsBackwardNHWC<float>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    float* dscale,
    float* dbias) {
  ConstEigenArrayMap<float> dY_arr(dY, C, N * HxW);
  ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
  ConstEigenVectorArrayMap<float> mean_arr(mean, C);
  ConstEigenVectorArrayMap<float> rstd_arr(rstd, C);
  EigenVectorArrayMap<float> dscale_arr(dscale, C);
  EigenVectorArrayMap<float> dbias_arr(dbias, C);
  dscale_arr = dY_arr.col(0) * X_arr.col(0);
  dbias_arr = dY_arr.col(0);
  for (int i = 1; i < N * HxW; ++i) {
    dscale_arr += dY_arr.col(i) * X_arr.col(i);
    dbias_arr += dY_arr.col(i);
  }
  dscale_arr = (dscale_arr - mean_arr * dbias_arr) * rstd_arr;
  return true;
}

REGISTER_CPU_OPERATOR(ChannelBackpropStats, ChannelBackpropStatsOp<CPUContext>);

OPERATOR_SCHEMA(ChannelBackpropStats)
    .NumInputs(4)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Given an input tensor in NCHW format, the gradient for the output of SpatialBN
and the per-channel mean and inverse std var vectors for the input, computes the
per-channel bias and scale gradient to be used during the backward pass for
subsequent spatial batch normalization gradient calculation. Typically, the
results of this op are subsequently reduced over multiple devices to obtain
statistics over a larger batch size in cases where the batch size for a single
model copy is too low to yield the full benefit of batch normalization. The
resulting bias and scale can then be plugged back into SpatialBNGradient to get
results over the larger batch size )DOC")
    .Input(0, "X", "The input 4-dimensional tensor of shape NCHW")
    .Input(
        1,
        "mean",
        "The mean saved from the forward pass as a 1-dimensional "
        "tensor of size C.")
    .Input(
        2,
        "inv_std",
        "The saved inverse standard deviation as a 1-dimensional tensor "
        "of size C.")
    .Input(
        3,
        "output_grad",
        "Gradient for the output layer of SpatialBN, here used as input "
        "because we are on the backward pass")
    .Output(0, "scale_grad", "Gradient for the scale vector")
    .Output(1, "bias_grad", "Gradient for the bias vector");

SHOULD_NOT_DO_GRADIENT(ChannelBackpropStats);

} // namespace caffe2
