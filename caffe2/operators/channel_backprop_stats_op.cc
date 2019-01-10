#include "caffe2/operators/channel_backprop_stats_op.h"

namespace caffe2 {

template <>
bool ChannelBackpropStatsOp<CPUContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& dY = Input(OUTPUT_GRAD);
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.ndim() > 3 ? X.dim32(3) : 1;
  const int D = X.ndim() > 4 ? X.dim32(4) : 1;

  const int sampleSize = H * W * D;

  Output(SCALE_GRAD)->Resize(C);
  Output(BIAS_GRAD)->Resize(C);
  auto* dScale = Output(SCALE_GRAD);
  auto* dBias = Output(BIAS_GRAD);

  ConstEigenArrayMap<float> X_arr(X.data<float>(), sampleSize, N * C);
  ConstEigenArrayMap<float> dY_arr(dY.data<float>(), sampleSize, N * C);
  ConstEigenVectorArrayMap<float> mean_arr(Input(SAVED_MEAN).data<float>(), C);
  ConstEigenVectorArrayMap<float> inv_stddev_arr(
      Input(SAVED_INV_STDDEV).data<float>(), C);
  EigenVectorArrayMap<float> dBias_arr(dBias->mutable_data<float>(), C);
  EigenVectorArrayMap<float> dScale_arr(dScale->mutable_data<float>(), C);

  dBias_arr.setZero();
  dScale_arr.setZero();

  for (int nc = 0; nc < N * C; ++nc) {
    int c = nc % C;
    dBias_arr(c) += dY_arr.col(nc).sum();
    dScale_arr(c) +=
        ((X_arr.col(nc) - mean_arr(c)) * inv_stddev_arr(c) * dY_arr.col(nc))
            .sum();
  }
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
