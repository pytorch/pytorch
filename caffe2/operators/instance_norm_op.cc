#include "caffe2/operators/instance_norm_op.h"

namespace caffe2 {

template <>
bool InstanceNormOp<CPUContext>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(INPUT);
  auto* Y = Output(OUTPUT);
  auto* mean = OutputSize() > 1 ? Output(MEAN) : &mean_;
  auto* inv_var = OutputSize() > 1 ? Output(INV_VAR) : &inv_var_;
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const size_t offset = H * W * C;
  Y->ResizeLike(X);
  mean->Resize(N, C);
  inv_var->Resize(N, C);
  ConstEigenVectorArrayMap<float> scale(Input(SCALE).data<float>(), C);
  ConstEigenVectorArrayMap<float> bias(Input(BIAS).data<float>(), C);
  for (int n = 0; n < N; ++n) {
    ConstEigenArrayMap<float> Xmat(X.data<float>() + offset * n, C, H * W);
    EigenArrayMap<float> Ymat(Y->mutable_data<float>() + offset * n, C, H * W);
    EigenVectorArrayMap<float> mean_arr(mean->mutable_data<float>() + n * C, C);
    EigenVectorArrayMap<float> inv_var_arr(
        inv_var->mutable_data<float>() + n * C, C);

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
    inv_var_arr = Ymat.col(0) * Ymat.col(0);
    for (int i = 1; i < H * W; ++i) {
      inv_var_arr += Ymat.col(i) * Ymat.col(i);
    }
    inv_var_arr = (inv_var_arr / (H * W) + epsilon_).sqrt().inverse();
    Ymat = (Ymat.colwise() * (inv_var_arr * scale)).colwise() + bias;
  }
  return true;
}

template <>
bool InstanceNormOp<CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  auto* Y = Output(OUTPUT);
  auto* mean = OutputSize() > 1 ? Output(MEAN) : &mean_;
  auto* inv_var = OutputSize() > 1 ? Output(INV_VAR) : &inv_var_;
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  Y->ResizeLike(X);
  mean->Resize(N, C);
  inv_var->Resize(N, C);
  ConstEigenArrayMap<float> Xmat(X.data<float>(), H * W, N * C);
  ConstEigenVectorArrayMap<float> scale(Input(SCALE).data<float>(), C);
  ConstEigenVectorArrayMap<float> bias(Input(BIAS).data<float>(), C);
  EigenArrayMap<float> Ymat(Y->mutable_data<float>(), H * W, N * C);
  EigenVectorArrayMap<float> mean_arr(mean->mutable_data<float>(), N * C);
  EigenVectorArrayMap<float> inv_var_arr(inv_var->mutable_data<float>(), N * C);

  mean_arr = Xmat.colwise().mean();

  // TODO(jiayq): refactor the following 4 lines to be more concise.
  Ymat = Xmat.rowwise() - mean_arr.transpose();
  inv_var_arr = Ymat.matrix().colwise().squaredNorm();
  inv_var_arr = (inv_var_arr / (H * W) + epsilon_).sqrt().inverse();
  // Vectorizing over H*W
  for (int i = 0; i < N * C; ++i) {
    Ymat.col(i) = Ymat.col(i) * (inv_var_arr(i) * scale(i % C)) + bias(i % C);
  }
  return true;
}

REGISTER_CPU_OPERATOR(InstanceNorm, InstanceNormOp<CPUContext>);

OPERATOR_SCHEMA(InstanceNorm)
    .NumInputs(3)
    .NumOutputs(1, 3)
    .SetDoc(R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y (train mode type 1, and test mode)
Output case #2: Y, saved_mean, saved_inv_var
                (train mode type 2)

For training mode, type 2 is faster in the sense that for the backward
pass, it is able to reuse the saved mean and inv_var in the gradient
computation.
)DOC")
    .Arg(
        "is_test",
        "If set to nonzero, run spatial batch normalization in test mode.")
    .Arg("epsilon", "The epsilon value to use to avoid division by zero.")
    .Arg("order", "A StorageOrder string.")
    .Input(
        0,
        "X",
        "The input 4-dimensional tensor of shape NCHW or NHWC depending "
        "on the order parameter.")
    .Input(1, "scale", "The input 1-dimensional scale tensor of size C.")
    .Input(2, "bias", "The input 1-dimensional bias tensor of size C.")
    .Output(0, "Y", "The output 4-dimensional tensor of the same shape as X.")
    .Output(
        1,
        "saved_mean",
        "Optional saved mean used during training to speed up gradient "
        "computation. Should not be used for testing.")
    .Output(
        2,
        "saved_inv_var",
        "Optional saved inverse variance used during training to speed up "
        "gradient computation. Should not be used for testing.");

GRADIENT_NOT_IMPLEMENTED_YET(InstanceNorm);

} // namespace caffe2
