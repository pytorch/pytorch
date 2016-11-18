#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

template <>
bool SpatialBNOp<CPUContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);

  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int C = (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(3));
  const int H = (order_ == StorageOrder::NCHW ? X.dim32(2) : X.dim32(1));
  const int W = (order_ == StorageOrder::NCHW ? X.dim32(3) : X.dim32(2));
  DCHECK_EQ(scale.ndim(), 1);
  DCHECK_EQ(bias.ndim(), 1);
  DCHECK_EQ(scale.dim32(0), C);
  DCHECK_EQ(bias.dim32(0), C);

  ConstEigenVectorArrayMap<float> scale_arr(scale.data<float>(), C);
  ConstEigenVectorArrayMap<float> bias_arr(bias.data<float>(), C);

  auto* Y = Output(OUTPUT);
  Y->ResizeLike(X);

  if (!is_test_) {
    // training mode
    // Get the mean and variance.
    // Note that, to be consistent with cudnn, we will output saved inverse
    // std as output 5, but we will still use the same storage place to
    // compute var as well. The inverse is going to be carried out at the end
    // of the op.
    Output(SAVED_MEAN)->Resize(C);
    Output(SAVED_INV_VAR)->Resize(C);
    EigenVectorArrayMap<float> mean(
        Output(SAVED_MEAN)->mutable_data<float>(), C);
    EigenVectorArrayMap<float> var(
        Output(SAVED_INV_VAR)->mutable_data<float>(), C);

    mean.setZero();
    var.setZero();
    switch (order_) {
      case StorageOrder::NCHW: {
        ConstEigenArrayMap<float> X_arr(X.data<float>(), H * W, N * C);
        for (int nc = 0; nc < N * C; ++nc) {
          mean(nc % C) += X_arr.col(nc).sum();
        }
        mean /= N * H * W;
        for (int nc = 0; nc < N * C; ++nc) {
          var(nc % C) += (X_arr.col(nc) - mean(nc % C)).matrix().squaredNorm();
        }
        var /= N * H * W;
        break;
      }
      case StorageOrder::NHWC: {
        ConstEigenArrayMap<float> X_arr(X.data<float>(), C, N * H * W);
        for (int i = 0; i < N * H * W; ++i) {
          mean += X_arr.col(i);
        }
        mean /= N * H * W;
        for (int i = 0; i < N * H * W; ++i) {
          var += (X_arr.col(i) - mean) * (X_arr.col(i) - mean);
        }
        var /= N * H * W;
        break;
      }
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }

    // Compute the running mean and running inv variance.
    auto* running_mean = Output(RUNNING_MEAN);
    auto* running_var = Output(RUNNING_VAR);
    // Check if they are initialized
    if (!running_mean->size()) {
      running_mean->Resize(C);
      EigenVectorArrayMap<float> running_mean_map(running_mean->mutable_data<float>(), C);
      running_mean_map.setZero();
    }
    if (!running_var->size()) {
      running_var->Resize(C);
      EigenVectorArrayMap<float> running_var_map(running_var->mutable_data<float>(), C);
      running_var_map.setZero();
    }
    EigenVectorArrayMap<float> running_mean_arr(
        running_mean->mutable_data<float>(), C);
    EigenVectorArrayMap<float> running_var_arr(
        running_var->mutable_data<float>(), C);
    running_mean_arr = running_mean_arr * momentum_ + mean * (1. - momentum_);
    running_var_arr = running_var_arr * momentum_ + var * (1. - momentum_);
  }

  // Regardless of training or testing, we will apply the estimated mean
  // and standard deviation to the input. For testing, they are
  // specified directly by the input, and for training, they are computed
  // by the op.
  Eigen::Array<float, Eigen::Dynamic, 1> inv_std(C);
  if (is_test_) {
    ConstEigenVectorArrayMap<float> var_arr(Input(EST_VAR).data<float>(), C);
    inv_std = (var_arr + epsilon_).sqrt().inverse();
  } else {
    EigenVectorArrayMap<float> saved_inv_std(
        Output(SAVED_INV_VAR)->mutable_data<float>(), C);
    saved_inv_std = (saved_inv_std + epsilon_).inverse().sqrt();
    inv_std = saved_inv_std;
  }
  ConstEigenVectorArrayMap<float> mean_arr(
      is_test_ ? Input(EST_MEAN).data<float>()
               : Output(SAVED_MEAN)->data<float>(),
      C);
  // We can fuse the output computation as follows:
  //   ((x - est_mean) * (inv_var) * scale + bias
  // to
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
  Eigen::Array<float, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<float, Eigen::Dynamic, 1> new_bias =
      bias_arr - mean_arr * inv_std * scale_arr;
  switch (order_) {
    case StorageOrder::NHWC: {
      EigenArrayMap<float>(Y->mutable_data<float>(), C, N * H * W) =
          (ConstEigenArrayMap<float>(X.data<float>(), C, N * H * W).colwise() *
           new_scale)
              .colwise() +
          new_bias;
      break;
    }
    case StorageOrder::NCHW: {
      EigenArrayMap<float> Y_arr(Y->mutable_data<float>(), H * W, N * C);
      ConstEigenArrayMap<float> X_arr(X.data<float>(), H * W, N * C);
      for (int nc = 0; nc < N * C; ++nc) {
        Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
  return true;
}

template <>
bool SpatialBNGradientOp<CPUContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& dY = Input(OUTPUT_GRAD);
  const auto& scale = Input(SCALE);

  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int C = (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(3));
  const int H = (order_ == StorageOrder::NCHW ? X.dim32(2) : X.dim32(1));
  const int W = (order_ == StorageOrder::NCHW ? X.dim32(3) : X.dim32(2));
  DCHECK_EQ(scale.ndim(), 1);
  DCHECK_EQ(scale.dim32(0), C);

  ConstEigenVectorArrayMap<float> scale_arr(scale.data<float>(), C);
  ConstEigenVectorArrayMap<float> mean_arr(Input(SAVED_MEAN).data<float>(), C);
  ConstEigenVectorArrayMap<float> inv_var_arr(
      Input(SAVED_INV_VAR).data<float>(), C);

  auto* dX = Output(INPUT_GRAD);
  auto* dScale = Output(SCALE_GRAD);
  auto* dBias = Output(BIAS_GRAD);
  dX->ResizeLike(X);
  dScale->ResizeLike(scale);
  dBias->ResizeLike(scale);

  // dBias = np.sum(dY, axis=0)
  // dScale = np.sum((X - mean) / inv_std * dy, axis=0)
  // dX = (1. / N) * scale * inv_var * (N * dY - np.sum(dY, axis=0) - (X - mean)
  //   * inv_var * inv_var * np.sum(dY * (X - mean), axis=0))

  EigenVectorArrayMap<float> dBias_arr(dBias->mutable_data<float>(), C);
  EigenVectorArrayMap<float> dScale_arr(dScale->mutable_data<float>(), C);

  dBias_arr.setZero();
  dScale_arr.setZero();

  const auto scaleInvVarNHW = scale_arr * inv_var_arr / (N * H * W);

  switch (order_) {
    case StorageOrder::NCHW: {
      ConstEigenArrayMap<float> X_arr(X.data<float>(), H * W, N * C);
      ConstEigenArrayMap<float> dY_arr(dY.data<float>(), H * W, N * C);
      EigenArrayMap<float> dX_arr(dX->mutable_data<float>(), H * W, N * C);
      dX_arr.setZero();

      for (int nc = 0; nc < N * C; ++nc) {
        int c = nc % C;
        dBias_arr(c) += dY_arr.col(nc).sum();
        dScale_arr(c) +=
            ((X_arr.col(nc) - mean_arr(c)) * inv_var_arr(c) * dY_arr.col(nc))
                .sum();
      }
      for (int nc = 0; nc < N * C; ++nc) {
        int c = nc % C;
        dX_arr.col(nc) += scaleInvVarNHW(c) *
            (dY_arr.col(nc) * N * H * W - dBias_arr(c) -
             (X_arr.col(nc) - mean_arr[c]) * dScale_arr(c) * inv_var_arr(c));
      }
      break;
    }
    case StorageOrder::NHWC: {
      ConstEigenArrayMap<float> X_arr(X.data<float>(), C, N * H * W);
      ConstEigenArrayMap<float> dY_arr(dY.data<float>(), C, N * H * W);
      EigenArrayMap<float> dX_arr(dX->mutable_data<float>(), C, N * H * W);
      dX_arr.setZero();

      const auto dYRowSum = dY_arr.rowwise().sum();
      const auto XMinusMean = X_arr.colwise() - mean_arr;
      const auto dYMulXMinusMeanRowSum = (dY_arr * XMinusMean).rowwise().sum();
      const auto invVarSqr = inv_var_arr * inv_var_arr;
      for (int nhw = 0; nhw < N * H * W; ++nhw) {
        dBias_arr += dY_arr.col(nhw);
        dScale_arr +=
            (X_arr.col(nhw) - mean_arr) * inv_var_arr * dY_arr.col(nhw);
        dX_arr.col(nhw) += scaleInvVarNHW *
            (dY_arr.col(nhw) * N * H * W - dYRowSum -
             XMinusMean.col(nhw) * invVarSqr * dYMulXMinusMeanRowSum);
      }
      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(SpatialBN, SpatialBNOp<CPUContext>);
REGISTER_CPU_OPERATOR(SpatialBNGradient, SpatialBNGradientOp<CPUContext>);

OPERATOR_SCHEMA(SpatialBN)
    .NumInputs(5)
    .NumOutputs({1, 5})
    .EnforceInplace({{3, 1}, {4, 2}})
    .SetDoc(R"DOC(
Carries out spatial batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var
                (training mode)
Output case #2: Y (test mode)
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
    .Input(
        1,
        "scale",
        "The scale as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Input(
        2,
        "bias",
        "The bias as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Input(
        3,
        "mean",
        "The running mean (training) or the estimated mean (testing) "
        "as a 1-dimensional tensor of size C.")
    .Input(
        4,
        "var",
        "The running variance (training) or the estimated "
        "variance (testing) as a 1-dimensional tensor of size C.")
    .Output(0, "Y", "The output 4-dimensional tensor of the same shape as X.")
    .Output(
        1,
        "mean",
        "The running mean after the spatial BN operator. Must be in-place "
        "with the input mean. Should not be used for testing.")
    .Output(
        2,
        "var",
        "The running variance after the spatial BN operator. Must be "
        "in-place with the input var. Should not be used for testing.")
    .Output(
        3,
        "saved_mean",
        "Saved mean used during training to speed up gradient "
        "computation. Should not be used for testing.")
    .Output(
        4,
        "saved_var",
        "Saved variance used during training to speed up "
        "gradient computation. Should not be used for testing.");

// Input: X, scale, dY, mean, variance
// Output: dX, dscale, dbias
OPERATOR_SCHEMA(SpatialBNGradient).NumInputs(5).NumOutputs(3);

// Spatial batch normalization's gradient, depending on the various input sizes,
// is a bit more complex than usual gradient operators.
class GetSpatialBNGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // Check if we are in training or testing mode.
    bool is_test = false;
    if (HasArgument(def_, "is_test")) {
      const auto& arg = GetArgument(def_, "is_test");
      CAFFE_ENFORCE(arg.has_i());
      is_test = arg.i();
    }
    vector<string> grad_outputs{GI(0), GI(1), GI(2)};
    vector<string> grad_inputs;
    if (is_test) {
      // This is in testing mode. The operator should have five input:
      //     X, scale, bias, estimated_mean, estimated_variance
      // The gradient inputs are:
      //     X, scale, dY, estimated_mean, estimated_variance
      CAFFE_ENFORCE_EQ(def_.input_size(), 5);
      CAFFE_ENFORCE_EQ(def_.output_size(), 1);
      grad_inputs = vector<string>{I(0), I(1), GO(0), I(3), I(4)};
    } else {
      CAFFE_ENFORCE_EQ(def_.input_size(), 5);
      CAFFE_ENFORCE_EQ(def_.output_size(), 5);
      grad_inputs = vector<string>{I(0), I(1), GO(0), O(3), O(4)};
    }
    return SingleGradientDef(
        "SpatialBNGradient", "", grad_inputs, grad_outputs);
  }
};
REGISTER_GRADIENT(SpatialBN, GetSpatialBNGradient);
} // namespace caffe2
