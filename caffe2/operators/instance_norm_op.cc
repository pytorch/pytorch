#include "caffe2/operators/instance_norm_op.h"

namespace caffe2 {

// Here lives two separate implementations of the forward and backward passes of
// instance normalization, one for NHWC order and the other for NCHW order.
// Two implementations allow us to make use of Eigen vectorized operations
// without an expensive tensor transpose operation.

template <typename T, typename Context>
bool InstanceNormOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(INPUT);
  auto* Y = Output(OUTPUT);
  CAFFE_ENFORCE(Y != &X, "Can't run InstanceNorm NHWC in-place");
  auto* mean = OutputSize() > 1 ? Output(MEAN) : &mean_;
  auto* inv_stdev = OutputSize() > 1 ? Output(INV_STDEV) : &inv_stdev_;
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const size_t offset = H * W * C;

  CAFFE_ENFORCE_EQ(Input(SCALE).size(), C);
  CAFFE_ENFORCE_EQ(Input(BIAS).size(), C);

  Y->ResizeLike(X);
  mean->Resize(N, C);
  inv_stdev->Resize(N, C);
  ConstEigenVectorArrayMap<T> scale(Input(SCALE).template data<T>(), C);
  ConstEigenVectorArrayMap<T> bias(Input(BIAS).template data<T>(), C);
  for (int n = 0; n < N; ++n) {
    ConstEigenArrayMap<T> Xmat(X.template data<T>() + offset * n, C, H * W);
    EigenArrayMap<T> Ymat(Y->template mutable_data<T>() + offset * n, C, H * W);
    EigenVectorArrayMap<T> mean_arr(
        mean->template mutable_data<T>() + n * C, C);
    EigenVectorArrayMap<T> inv_stdev_arr(
        inv_stdev->template mutable_data<T>() + n * C, C);

    // The following effectively does the row wise mean computation:
    //   mean_arr = Xmat.rowwise().mean();
    // but manually vectorizes over columns.
    mean_arr = Xmat.col(0);
    for (int i = 1; i < H * W; ++i) {
      mean_arr += Xmat.col(i);
    }
    mean_arr *= 1. / (H * W);
    Ymat = Xmat.colwise() - mean_arr;
    // The following effectively does row wise squared norm computation,
    // but manually vectorizes over columns similar to the mean case.
    inv_stdev_arr = Ymat.col(0) * Ymat.col(0);
    for (int i = 1; i < H * W; ++i) {
      inv_stdev_arr += Ymat.col(i) * Ymat.col(i);
    }
    inv_stdev_arr = (inv_stdev_arr / (H * W) + epsilon_).sqrt().inverse();
    Ymat = (Ymat.colwise() * (inv_stdev_arr * scale)).colwise() + bias;
  }
  return true;
}

template <typename T, typename Context>
bool InstanceNormOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);
  auto* Y = Output(OUTPUT);
  auto* mean = OutputSize() > 1 ? Output(MEAN) : &mean_;
  auto* inv_stdev = OutputSize() > 1 ? Output(INV_STDEV) : &inv_stdev_;
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  CAFFE_ENFORCE_EQ(scale.size(), C);
  CAFFE_ENFORCE_EQ(bias.size(), C);

  Y->ResizeLike(X);
  mean->Resize(N, C);
  inv_stdev->Resize(N, C);

  const auto* Xdata = X.template data<T>();
  auto* Ydata = Y->template mutable_data<T>();
  const auto* scale_data = scale.template data<T>();
  const auto* bias_data = bias.template data<T>();
  auto* mean_data = mean->template mutable_data<T>();
  auto* inv_stdev_data = inv_stdev->template mutable_data<T>();

  // TODO: benchmark parallelization strategies.
  for (auto i = 0; i < N * C; ++i) {
    ConstEigenVectorArrayMap<T> Xi(Xdata + H * W * i, H * W);
    const T Xi_mean = Xi.mean();
    const T squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const T inv_stdev = 1.0 / std::sqrt(squared_norm / (H * W) + epsilon_);
    mean_data[i] = Xi_mean;
    inv_stdev_data[i] = inv_stdev;
    EigenVectorArrayMap<T> Yi(Ydata + H * W * i, H * W);
    const T channel_scale = inv_stdev * scale_data[i % C];
    const T channel_shift = bias_data[i % C] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
  }

  return true;
}

REGISTER_CPU_OPERATOR(InstanceNorm, InstanceNormOp<float, CPUContext>);

OPERATOR_SCHEMA(InstanceNorm)
    .NumInputs(3)
    .NumOutputs(1, 3)
    .AllowInplace({{0,0}})
    .SetDoc(R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

  * Output case #1: output
  * Output case #2: output, saved_mean
    - don't use, doesn't make sense but won't crash
  * Output case #3: output, saved_mean, saved_inv_stdev
    - Makes sense for training only

For training mode, type 3 is faster in the sense that for the backward
pass, it is able to reuse the saved mean and inv_stdev in the gradient
computation.
)DOC")
    .Arg("epsilon", "The epsilon value to use to avoid division by zero.")
    .Arg("order", "A StorageOrder string.")
    .Input(
        0,
        "input",
        "The input 4-dimensional tensor of shape NCHW or NHWC depending "
        "on the order parameter.")
    .Input(1, "scale", "The input 1-dimensional scale tensor of size C.")
    .Input(2, "bias", "The input 1-dimensional bias tensor of size C.")
    .Output(
        0,
        "output",
        "The output 4-dimensional tensor of the same shape as input.")
    .Output(
        1,
        "saved_mean",
        "Optional saved mean used during training to speed up gradient "
        "computation. Should not be used for testing.")
    .Output(
        2,
        "saved_inv_stdev",
        "Optional saved inverse stdev used during training to speed up "
        "gradient computation. Should not be used for testing.");

} // namespace caffe2
