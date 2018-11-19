#include "caffe2/operators/instance_norm_op.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

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
  CAFFE_ENFORCE_EQ(4, input.dim());
  const int N = input.dim32(0);
  const int H = input.dim32(1);
  const int W = input.dim32(2);
  const int C = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.dim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.dim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.dim());
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
    CAFFE_ENFORCE_EQ(2, mean.dim());
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
    CAFFE_ENFORCE_EQ(2, inv_stdev.dim());
    CAFFE_ENFORCE_EQ(N, inv_stdev.dim32(0));
    CAFFE_ENFORCE_EQ(C, inv_stdev.dim32(1));

    ConstEigenVectorArrayMap<T> inv_stdev_arr(
        inv_stdev.template data<T>() + n * C, C);

    // for each channel
    // dl/dbias = sum_j dl/dy_j
    auto bias_grad_delta = output_grad_mat.rowwise().sum();
    if (n == 0) {
      bias_grad_arr = bias_grad_delta;
    } else {
      bias_grad_arr += bias_grad_delta;
    }
    // for each channel
    // dl/dscale = sum_j dl/dy_j (x_j - mu) / stdev
    auto scale_grad_delta =
        ((input_grad_mat.colwise() * inv_stdev_arr) * output_grad_mat)
            .rowwise()
            .sum();
    if (n == 0) {
      scale_grad_arr = scale_grad_delta;
    } else {
      scale_grad_arr += scale_grad_delta;
    }

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
  CAFFE_ENFORCE_EQ(4, input.dim());
  const int N = input.dim32(0);
  const int C = input.dim32(1);
  const int H = input.dim32(2);
  const int W = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.dim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.dim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.dim());
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
  CAFFE_ENFORCE_EQ(2, mean.dim());
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
  CAFFE_ENFORCE_EQ(2, inv_stdev.dim());
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

REGISTER_CPU_OPERATOR(
    InstanceNormGradient,
    InstanceNormGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(InstanceNormGradient).NumInputs(4, 6).NumOutputs(3);

REGISTER_GRADIENT(InstanceNorm, GetInstanceNormGradient);
}
