#include "caffe2/quantization/server/spatial_batch_norm_dnnlowp_op.h"

#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"

namespace caffe2 {

template <typename T>
SpatialBNDNNLowPOp<T>::SpatialBNDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : DNNLowPOp<T, SpatialBNOp<CPUContext>>(operator_def, ws),
      OP_SINGLE_ARG(double, "epsilon", epsilon_, 1e-5),
      order_(StringToStorageOrder(
          this->template GetSingleArgument<std::string>("order", "NCHW"))) {
  bool is_test = this->template GetSingleArgument<bool>("is_test", false);
  OPERATOR_NEEDS_FEATURE(
      is_test, "SpatialBN DNNLOWP op only works for inference.");
  CAFFE_ENFORCE_NE(
      order_,
      StorageOrder::UNKNOWN,
      "order should be either \"NCHW\" or \"NHWC\".");
  CAFFE_ENFORCE(OutputSize() == 1);
  CAFFE_ENFORCE_GT(epsilon_, 0);
}

template <typename T>
void SpatialBNDNNLowPOp<T>::ComputeFusedParam_(
    const int C,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* var,
    float* alpha,
    float* beta) {
  EigenVectorArrayMap<float> alpha_arr(alpha, C);
  EigenVectorArrayMap<float> beta_arr(beta, C);
  alpha_arr = ConstEigenVectorArrayMap<float>(scale, C) *
      (ConstEigenVectorArrayMap<float>(var, C) + epsilon_).rsqrt();
  beta_arr = ConstEigenVectorArrayMap<float>(bias, C) -
      alpha_arr * ConstEigenVectorArrayMap<float>(mean, C);

  // Adjust alpha and beta considering quantization scales
  alpha_arr = alpha_arr * (in_qparams_[0].scale / out_qparams_.scale);
  beta_arr = beta_arr / out_qparams_.scale;
}

template <typename T>
bool SpatialBNDNNLowPOp<T>::RunOnDevice() {
  const auto& X = InputTensorCPU_(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);

  const int ndim = X.dim();
  CAFFE_ENFORCE_GE(ndim, 3);
  const int N = X.dim32(0);
  const int C = (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
  const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
  const int HxW =
      std::accumulate(
          X_dims.cbegin() + 1, X_dims.cend(), 1, std::multiplies<int>()) /
      C;
  CAFFE_ENFORCE_EQ(scale.numel(), C);
  CAFFE_ENFORCE_EQ(bias.numel(), C);

  GetOutputQuantizationParams_();

  in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  const float* scale_data = scale.template data<float>();
  const float* bias_data = bias.template data<float>();
  ReinitializeTensor(
      &alpha_, {C}, at::dtype<float>().device(CPUContext::GetDeviceType()));
  ReinitializeTensor(
      &beta_, {C}, at::dtype<float>().device(CPUContext::GetDeviceType()));
  float* alpha_data = alpha_.template mutable_data<float>();
  float* beta_data = beta_.template mutable_data<float>();
  if (N == 0) {
    return true;
  }
  const auto& mean = Input(EST_MEAN);
  const auto& var = Input(EST_VAR);
  CAFFE_ENFORCE_EQ(mean.numel(), C);
  CAFFE_ENFORCE_EQ(var.numel(), C);
  ComputeFusedParam_(
      C,
      scale_data,
      bias_data,
      mean.template data<float>(),
      var.template data<float>(),
      alpha_data,
      beta_data);

  vector<T> X_temp;
  const T* X_data =
      dnnlowp::QuantizeInputIfNeeded(this, 0, in_qparams_[0], X_temp);
  auto* Y = OutputTensorCPU_(OUTPUT);
  Y->Resize(X.sizes());
  T* Y_data = GetQuantizedOutputData_();

  if (order_ == StorageOrder::NCHW) {
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < HxW; ++j) {
          long quantized_down = out_qparams_.zero_point +
              std::lrintf(alpha_data[c] *
                              (X_data[(i * C + c) * HxW + j] -
                               in_qparams_[0].zero_point) +
                          beta_data[c]);
          Y_data[(i * C + c) * HxW + j] =
              fbgemm::clamp<long, T>(quantized_down, 8);
        }
      }
    }
  } else {
    for (int i = 0; i < N * HxW; ++i) {
      for (int c = 0; c < C; ++c) {
        long quantized_down = out_qparams_.zero_point +
            std::lrintf(alpha_data[c] *
                            (X_data[i * C + c] - in_qparams_[0].zero_point) +
                        beta_data[c]);
        Y_data[i * C + c] = fbgemm::clamp<long, T>(quantized_down, 8);
      }
    }
  }

  RunOnDeviceEpilogue_();

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    SpatialBN,
    DNNLOWP,
    SpatialBNDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8SpatialBN,
    DNNLOWP,
    SpatialBNDNNLowPOp<uint8_t>);

} // namespace caffe2
