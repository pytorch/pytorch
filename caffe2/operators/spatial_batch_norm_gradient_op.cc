#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

template <>
bool SpatialBNGradientOp<CPUContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& dY = Input(OUTPUT_GRAD);
  const auto& scale = Input(SCALE);

  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  const int N = X.dim32(0);
  const int C =
      (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(X.ndim() - 1));
  const int H = (order_ == StorageOrder::NCHW ? X.dim32(2) : X.dim32(1));
  const int W = X.ndim() > 3
      ? (order_ == StorageOrder::NCHW ? X.dim32(3) : X.dim32(2))
      : 1;
  const int D = X.ndim() > 4
      ? (order_ == StorageOrder::NCHW ? X.dim32(4) : X.dim32(3))
      : 1;

  const int sample_size = H * W * D;

  CAFFE_ENFORCE_EQ(scale.ndim(), 1);
  CAFFE_ENFORCE_EQ(scale.dim32(0), C);

  ConstEigenVectorArrayMap<float> scale_arr(scale.data<float>(), C);
  ConstEigenVectorArrayMap<float> mean_arr(Input(SAVED_MEAN).data<float>(), C);
  ConstEigenVectorArrayMap<float> inv_var_arr(
      Input(SAVED_INV_VAR).data<float>(), C);

  auto* dX = Output(INPUT_GRAD);
  dX->ResizeLike(X);

  auto* dScale = Output(SCALE_GRAD);
  auto* dBias = Output(BIAS_GRAD);

  if (num_batches_ == 1) {
    dScale->ResizeLike(scale);
    dBias->ResizeLike(scale);
  }

  // dBias = np.sum(dY, axis=0)
  // dScale = np.sum((X - mean) / inv_std * dy, axis=0)
  // dX = (1. / N) * scale * inv_var * (N * dY - np.sum(dY, axis=0) - (X - mean)
  //   * inv_var * inv_var * np.sum(dY * (X - mean), axis=0))

  EigenVectorArrayMap<float> dBias_arr(dBias->mutable_data<float>(), C);
  EigenVectorArrayMap<float> dScale_arr(dScale->mutable_data<float>(), C);

  if (num_batches_ == 1) {
    dBias_arr.setZero();
    dScale_arr.setZero();
  }

  const auto scaleInvVarNHW = scale_arr * inv_var_arr / (N * sample_size);

  switch (order_) {
    case StorageOrder::NCHW: {
      ConstEigenArrayMap<float> X_arr(X.data<float>(), sample_size, N * C);
      ConstEigenArrayMap<float> dY_arr(dY.data<float>(), sample_size, N * C);
      EigenArrayMap<float> dX_arr(
          dX->mutable_data<float>(), sample_size, N * C);
      dX_arr.setZero();

      if (num_batches_ == 1) {
        for (int nc = 0; nc < N * C; ++nc) {
          int c = nc % C;
          dBias_arr(c) += dY_arr.col(nc).sum();
          dScale_arr(c) +=
              ((X_arr.col(nc) - mean_arr(c)) * inv_var_arr(c) * dY_arr.col(nc))
                  .sum();
        }
      } else {
        for (int c = 0; c < C; ++c) {
          dBias_arr(c) /= num_batches_;
          dScale_arr(c) /= num_batches_;
        }
      }
      for (int nc = 0; nc < N * C; ++nc) {
        int c = nc % C;
        dX_arr.col(nc) += scaleInvVarNHW(c) *
            (dY_arr.col(nc) * N * sample_size - dBias_arr(c) -
             (X_arr.col(nc) - mean_arr[c]) * dScale_arr(c) * inv_var_arr(c));
      }
      break;
    }
    case StorageOrder::NHWC: {
      ConstEigenArrayMap<float> X_arr(X.data<float>(), C, N * sample_size);
      ConstEigenArrayMap<float> dY_arr(dY.data<float>(), C, N * sample_size);
      EigenArrayMap<float> dX_arr(
          dX->mutable_data<float>(), C, N * sample_size);
      dX_arr.setZero();

      const auto dYRowSum = dY_arr.rowwise().sum();
      const auto XMinusMean = X_arr.colwise() - mean_arr;
      const auto dYMulXMinusMeanRowSum = (dY_arr * XMinusMean).rowwise().sum();
      const auto invVarSqr = inv_var_arr * inv_var_arr;
      for (int nhw = 0; nhw < N * sample_size; ++nhw) {
        dBias_arr += dY_arr.col(nhw);
        dScale_arr +=
            (X_arr.col(nhw) - mean_arr) * inv_var_arr * dY_arr.col(nhw);
        dX_arr.col(nhw) += scaleInvVarNHW *
            (dY_arr.col(nhw) * N * sample_size - dYRowSum -
             XMinusMean.col(nhw) * invVarSqr * dYMulXMinusMeanRowSum);
      }
      break;
    }
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(SpatialBNGradient, SpatialBNGradientOp<CPUContext>);

// Input: X, scale, dY, mean, variance, dscale, dbias
// Output: dX, dscale, dbias
OPERATOR_SCHEMA(SpatialBNGradient)
    .NumInputs({5, 7})
    .NumOutputs(3)
    .AllowInplace({{5, 1}, {6, 2}});

// Spatial batch normalization's gradient, depending on the various input sizes,
// is a bit more complex than usual gradient operators.
class GetSpatialBNGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // Check if we are in training or testing mode.
    bool is_test =
        ArgumentHelper::GetSingleArgument(def_, OpSchema::Arg_IsTest, 0);
    int num_batches = ArgumentHelper::GetSingleArgument(def_, "num_batches", 1);
    vector<string> grad_outputs{GI(0), GI(1), GI(2)};
    vector<string> grad_inputs;
    if (is_test) {
      // This is in testing mode. The operator should have five inputs:
      //     X, scale, bias, estimated_mean, estimated_variance
      // The gradient inputs are:
      //     X, scale, dY, estimated_mean, estimated_variance
      CAFFE_ENFORCE_EQ(def_.input_size(), 5);
      CAFFE_ENFORCE_EQ(def_.output_size(), 1);
      grad_inputs = vector<string>{I(0), I(1), GO(0), I(3), I(4)};
    } else if (num_batches > 1) {
      CAFFE_ENFORCE_EQ(def_.input_size(), 7);
      CAFFE_ENFORCE_EQ(def_.output_size(), 5);
      grad_inputs = vector<string>{I(0), I(1), GO(0), O(3), O(4), GI(1), GI(2)};
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
}
