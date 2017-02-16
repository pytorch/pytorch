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
  auto* mean = OutputSize() > 1 ? Output(MEAN) : &mean_;
  auto* inv_stdev = OutputSize() > 1 ? Output(INV_STDEV) : &inv_stdev_;
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const size_t offset = H * W * C;
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
  Y->ResizeLike(X);
  mean->Resize(N, C);
  inv_stdev->Resize(N, C);

  const auto* Xdata = X.template data<T>();
  auto* Ydata = Y->template mutable_data<T>();
  const auto* scale_data = scale.template data<T>();
  const auto* bias_data = bias.template data<T>();
  auto* mean_data = mean->template mutable_data<T>();
  auto* inv_stdev_data = inv_stdev->template mutable_data<T>();

  auto f = [&](size_t i) {
    ConstEigenVectorArrayMap<T> Xi(Xdata + H * W * i, H * W);
    const T mean = Xi.mean();
    const T squared_norm = (Xi - mean).matrix().squaredNorm();
    const T inv_stdev = 1.0 / std::sqrt(squared_norm / (H * W) + epsilon_);
    mean_data[i] = mean;
    inv_stdev_data[i] = inv_stdev;
    EigenVectorArrayMap<T> Yi(Ydata + H * W * i, H * W);
    Yi = (Xi - mean) * (inv_stdev * scale_data[i % C]) + bias_data[i % C];
  };

  // TODO: benchmark parallelization strategies.
  for (auto i = 0; i < N * C; ++i) {
    f(i);
  }

  return true;
}

template <typename T, typename Context>
bool InstanceNormGradientOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const auto& input = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);
  const auto& output_grad = Input(OUTPUT_GRAD);
  const auto& mean = InputSize() >= 5 ? Input(MEAN) : mean_;
  const auto& inv_stdev = InputSize() >= 6 ? Input(INV_STDEV) : inv_stdev_;
  auto input_grad = Output(INPUT_GRAD);
  auto scale_grad = Output(SCALE_GRAD);
  auto bias_grad = Output(BIAS_GRAD);
  CAFFE_ENFORCE_EQ(4, input.ndim());
  const int N = input.dim32(0);
  const int H = input.dim32(1);
  const int W = input.dim32(2);
  const int C = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.ndim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.ndim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.ndim());
  CAFFE_ENFORCE_EQ(N, output_grad.dim32(0));
  CAFFE_ENFORCE_EQ(H, output_grad.dim32(1));
  CAFFE_ENFORCE_EQ(W, output_grad.dim32(2));
  CAFFE_ENFORCE_EQ(C, output_grad.dim32(3));
  input_grad->ResizeLike(input);
  scale_grad->ResizeLike(scale);
  bias_grad->ResizeLike(bias);

  ConstEigenVectorArrayMap<T> scale_arr(scale.template data<T>(), C);
  ConstEigenVectorArrayMap<T> bias_arr(bias.template data<T>(), C);
  EigenVectorArrayMap<T> scale_grad_arr(
      scale_grad->template mutable_data<T>(), C);
  EigenVectorArrayMap<T> bias_grad_arr(
      bias_grad->template mutable_data<T>(), C);

  // Resize before we get into the per-instance loop
  if (InputSize() < 5) {
    mean_.Resize(N, C);
  }
  if (InputSize() < 6) {
    inv_stdev_.Resize(N, C);
  }

  // looping over per-instance and using Eigen blocks to extract out
  // a chunk of channels
  for (int n = 0; n < N; ++n) {
    // All Eigen mats and arrs in here are per-instance.
    ConstEigenArrayMap<T> input_mat(
        input.template data<T>() + n * C * H * W, C, H * W);
    ConstEigenArrayMap<T> output_grad_mat(
        output_grad.template data<T>() + n * C * H * W, C, H * W);
    EigenArrayMap<T> input_grad_mat(
        input_grad->template mutable_data<T>() + n * C * H * W, C, H * W);

    // Compute mean if it wasn't passed in
    if (InputSize() < 5) {
      EigenVectorArrayMap<T> mean_mutable_arr(
          mean_.template mutable_data<T>() + n * C, C);
      mean_mutable_arr = input_mat.rowwise().mean();
    }
    CAFFE_ENFORCE_EQ(2, mean.ndim());
    CAFFE_ENFORCE_EQ(N, mean.dim32(0));
    CAFFE_ENFORCE_EQ(C, mean.dim32(1));
    ConstEigenVectorArrayMap<T> mean_arr(mean.template data<T>() + n * C, C);

    // subtract mean
    input_grad_mat = input_mat.colwise() - mean_arr;

    // Compute 1 / stdev if it wasn't passed in
    if (InputSize() < 6) {
      EigenVectorArrayMap<T> inv_stdev_mutable_arr(
          inv_stdev_.template mutable_data<T>() + n * C, C);

      // Square the diffs along each channel and take the mean to get var
      inv_stdev_mutable_arr = input_grad_mat.pow(2).rowwise().mean();
      // sqrt to get stdev and take the inverse
      inv_stdev_mutable_arr =
          (inv_stdev_mutable_arr + epsilon_).sqrt().inverse();
    }
    CAFFE_ENFORCE_EQ(2, inv_stdev.ndim());
    CAFFE_ENFORCE_EQ(N, inv_stdev.dim32(0));
    CAFFE_ENFORCE_EQ(C, inv_stdev.dim32(1));

    ConstEigenVectorArrayMap<T> inv_stdev_arr(
        inv_stdev.template data<T>() + n * C, C);

    // for each channel
    // dl/dbias = sum_j dl/dy_j
    bias_grad_arr += output_grad_mat.rowwise().sum();
    // for each channel
    // dl/dscale = sum_j dl/dy_j (x_j - mu) / stdev
    scale_grad_arr +=
        ((input_grad_mat.colwise() * inv_stdev_arr) * output_grad_mat)
            .rowwise()
            .sum();

    // dl/dx_j = this gross thing
    // Derived gradient and manually massaged it to minimize extra storage
    // and number of vectorized calls.  Verified it with the autograd package
    // in python.

    // a = -1/(HW) sum_j dl/dy_j * (x_j - mu) / stdev^3
    const auto temp = (inv_stdev_arr.pow(3) *
                       (input_grad_mat * output_grad_mat).rowwise().mean() *
                       -1).eval();
    // b_j = a * (x_j - mu)
    input_grad_mat.colwise() *= temp;

    // c_j = b_j + dl/dy_j / stdev
    input_grad_mat += output_grad_mat.colwise() * inv_stdev_arr;

    // dl/dx_j = s * (c_j - mean(c_j))
    const auto result_mean = input_grad_mat.rowwise().mean().eval();
    input_grad_mat.colwise() -= result_mean;
    input_grad_mat.colwise() *= scale_arr;
  }

  return true;
}

template <typename T, typename Context>
bool InstanceNormGradientOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& input = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);
  const auto& output_grad = Input(OUTPUT_GRAD);
  const auto& mean = InputSize() >= 5 ? Input(MEAN) : mean_;
  const auto& inv_stdev = InputSize() >= 6 ? Input(INV_STDEV) : inv_stdev_;
  auto input_grad = Output(INPUT_GRAD);
  auto scale_grad = Output(SCALE_GRAD);
  auto bias_grad = Output(BIAS_GRAD);
  CAFFE_ENFORCE_EQ(4, input.ndim());
  const int N = input.dim32(0);
  const int C = input.dim32(1);
  const int H = input.dim32(2);
  const int W = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.ndim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.ndim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.ndim());
  CAFFE_ENFORCE_EQ(N, output_grad.dim32(0));
  CAFFE_ENFORCE_EQ(C, output_grad.dim32(1));
  CAFFE_ENFORCE_EQ(H, output_grad.dim32(2));
  CAFFE_ENFORCE_EQ(W, output_grad.dim32(3));
  input_grad->ResizeLike(input);
  scale_grad->ResizeLike(scale);
  bias_grad->ResizeLike(bias);

  ConstEigenArrayMap<T> input_mat(input.template data<T>(), H * W, N * C);
  ConstEigenVectorArrayMap<T> scale_arr(scale.template data<T>(), C);
  ConstEigenVectorArrayMap<T> bias_arr(bias.template data<T>(), C);
  ConstEigenArrayMap<T> output_grad_mat(
      output_grad.template data<T>(), H * W, N * C);

  EigenArrayMap<T> input_grad_mat(
      input_grad->template mutable_data<T>(), H * W, N * C);
  EigenVectorArrayMap<T> scale_grad_arr(
      scale_grad->template mutable_data<T>(), C);
  EigenVectorArrayMap<T> bias_grad_arr(
      bias_grad->template mutable_data<T>(), C);

  // Compute mean if it wasn't passed in
  if (InputSize() < 5) {
    mean_.Resize(N, C);
    EigenVectorArrayMap<T> mean_mutable_arr(
        mean_.template mutable_data<T>(), N * C);
    mean_mutable_arr = input_mat.colwise().mean();
  }
  CAFFE_ENFORCE_EQ(2, mean.ndim());
  CAFFE_ENFORCE_EQ(N, mean.dim32(0));
  CAFFE_ENFORCE_EQ(C, mean.dim32(1));
  ConstEigenVectorArrayMap<T> mean_arr(mean.template data<T>(), N * C);

  // subtract mean
  input_grad_mat = input_mat.rowwise() - mean_arr.transpose();

  // compute 1 / stdev if not passed in
  if (InputSize() < 6) {
    inv_stdev_.Resize(N, C);
    EigenVectorArrayMap<T> inv_stdev_mutable_arr(
        inv_stdev_.template mutable_data<T>(), N * C);

    // Square the diffs along each column and take mean to get var
    inv_stdev_mutable_arr = input_grad_mat.pow(2).colwise().mean();
    // sqrt to get stdev and then invert
    inv_stdev_mutable_arr = (inv_stdev_mutable_arr + epsilon_).sqrt().inverse();
  }
  CAFFE_ENFORCE_EQ(2, inv_stdev.ndim());
  CAFFE_ENFORCE_EQ(N, inv_stdev.dim32(0));
  CAFFE_ENFORCE_EQ(C, inv_stdev.dim32(1));

  ConstEigenVectorArrayMap<T> inv_stdev_arr(
      inv_stdev.template data<T>(), N * C);

  // Visit comments in the NHWC version about these gradients.  scale and bias
  // grads are about the same, but the input grads no longer slice out one
  // example at a time and instead vectorize across all N * C feature maps.

  // scale and bias gradients
  scale_grad_arr.setZero();
  bias_grad_arr.setZero();
  for (int n = 0; n < N; ++n) {
    scale_grad_arr += ((input_grad_mat.rowwise() * inv_stdev_arr.transpose()) *
                       output_grad_mat)
                          .block(0, n * C, H * W, C)
                          .colwise()
                          .sum();
    bias_grad_arr += output_grad_mat.block(0, n * C, H * W, C).colwise().sum();
  }

  // input gradient
  const auto temp = ((inv_stdev_arr.pow(3).transpose() *
                      (input_grad_mat * output_grad_mat).colwise().mean()) *
                     -1).eval();
  input_grad_mat.rowwise() *= temp;

  input_grad_mat += output_grad_mat.rowwise() * inv_stdev_arr.transpose();

  const auto result_mean = input_grad_mat.colwise().mean().eval();
  input_grad_mat.rowwise() -= result_mean;

  for (int n = 0; n < N; ++n) {
    input_grad_mat.block(0, n * C, H * W, C).rowwise() *= scale_arr.transpose();
  }

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(InstanceNorm, InstanceNormOp<float, CPUContext>);

OPERATOR_SCHEMA(InstanceNorm)
    .NumInputs(3)
    .NumOutputs(1, 3)
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

OPERATOR_SCHEMA(InstanceNormGradient).NumInputs(4, 6).NumOutputs(3);

REGISTER_CPU_OPERATOR(
    InstanceNormGradient,
    InstanceNormGradientOp<float, CPUContext>);

class GetInstanceNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> inputs{I(0), I(1), I(2), GO(0)};
    if (def_.output_size() >= 2) {
      inputs.push_back(O(1));
    }
    if (def_.output_size() >= 3) {
      inputs.push_back(O(2));
    }
    return SingleGradientDef(
        "InstanceNormGradient",
        "",
        inputs,
        vector<string>{GI(0), GI(1), GI(2)});
  }
};
REGISTER_GRADIENT(InstanceNorm, GetInstanceNormGradient);

} // namespace
} // namespace caffe2
