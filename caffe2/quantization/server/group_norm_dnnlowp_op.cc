#include "caffe2/quantization/server/group_norm_dnnlowp_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
template <typename T>
GroupNormDNNLowPOp<T>::GroupNormDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : BaseType(operator_def, ws),
      OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, true),
      OP_SINGLE_ARG(int, "group", group_, 32),
      OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
      order_(StringToStorageOrder(
          this->template GetSingleArgument<std::string>("order", "NCHW"))),
      OP_SINGLE_ARG(bool, "is_param_constant", is_param_constant_, true) {
  CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  if (!is_param_constant_) {
    LOG(INFO) << operator_def.output(0) << " is_param_constant "
              << is_param_constant_;
  }
}

template <typename T>
bool GroupNormDNNLowPOp<T>::RunOnDevice() {
  this->ParseDNNLowPOperatorArguments_();
  if (!GetQuantizationParameters()) {
    return false;
  }
  return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                      : RunOnDeviceWithOrderNHWC();
}

template <typename T>
bool GroupNormDNNLowPOp<T>::GetQuantizationParameters() {
  // Choose quantization for X
  in_qparams_[INPUT] =
      GetInputTensorQuantizationParamsOf(this, INPUT, qfactory_.get());
  QuantizeGamma();
  QuantizeBeta();
  if (!dequantize_output_) {
    GetOutputQuantizationParams_();
  } else if (measure_quantization_error_) {
    // to measure quantization error, run ref impl.
    Fp32Op_()->DequantizeInput();
    Fp32Op_()->Get()->RunOnDevice();
  }
  return true;
}

template <typename T>
void GroupNormDNNLowPOp<T>::QuantizeGamma() {
  if (is_param_constant_) {
    if (gamma_quantized_data_ == nullptr &&
        gamma_dequantized_data_ == nullptr) {
      const auto& gamma = InputTensorCPU_(GAMMA);
      const int C = gamma.size();
      gamma_quantized_.resize(C);
      gamma_quantized_data_ = gamma_quantized_.data();
      if (this->template InputIsType<int8::Int8TensorCPU>(GAMMA)) {
        const auto& gamma_int8 =
            this->template Input<int8::Int8TensorCPU>(GAMMA);
        auto& gamma_qparams = in_qparams_[GAMMA];
        gamma_qparams.scale = gamma_int8.scale;
        const T* gamma_data = gamma.template data<T>();
        EigenVectorArrayMap<int32_t>(gamma_quantized_.data(), C) =
            ConstEigenVectorArrayMap<T>(gamma_data, C)
                .template cast<int32_t>() -
            gamma_int8.zero_point;
        gamma_qparams.zero_point = 0;
        if (dequantize_output_) {
          gamma_dequantized_.resize(C);
          gamma_dequantized_data_ = gamma_dequantized_.data();
          fbgemm::Dequantize<int32_t>(
              gamma_quantized_data_,
              gamma_dequantized_.data(),
              C,
              gamma_qparams);
        }
      } else {
        QuantizeGammaImpl();
      }
    }
  } else {
    QuantizeGammaImpl();
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::QuantizeGammaImpl() {
  const auto& gamma = InputTensorCPU_(GAMMA);
  const int C = gamma.size();
  auto& gamma_qparams = in_qparams_[GAMMA];
  gamma_qparams = GetInputTensorQuantizationParamsOf(
      this, GAMMA, qfactory_.get(), true /* is_weight */);
  gamma_qparams.zero_point = 0;
  gamma_quantized_.resize(C);
  gamma_quantized_data_ = gamma_quantized_.data();
  gamma_dequantized_data_ = gamma.template data<float>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < C; ++i) {
    gamma_quantized_[i] = fbgemm::Quantize<int32_t>(
        gamma_dequantized_data_[i],
        gamma_qparams.zero_point,
        gamma_qparams.scale,
        32);
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::QuantizeBeta() {
  if (!is_param_constant_ ||
      (beta_quantized_data_ == nullptr && beta_dequantized_data_ == nullptr) ||
      cached_X_qparams_scale_ != in_qparams_[INPUT].scale) {
    const auto& beta = InputTensorCPU_(BETA);
    const int C = beta.size();
    const auto& X_qparams = in_qparams_[INPUT];
    const auto& gamma_qparams = in_qparams_[GAMMA];
    auto& beta_qparams = in_qparams_[BETA];
    if (this->template InputIsType<int8::Int8TensorCPU>(BETA)) {
      const auto& beta_int8 = this->template Input<int8::Int8TensorCPU>(BETA);
      beta_qparams.scale = beta_int8.scale;
      beta_qparams.zero_point = beta_int8.zero_point;
      const auto& X = InputTensorCPU_(INPUT);
      const int N = X.dim32(0);
      if (N > 0) {
        CAFFE_ENFORCE_LE(
            std::abs(
                beta_qparams.scale - X_qparams.scale * gamma_qparams.scale),
            1e-4);
      }
      CAFFE_ENFORCE_EQ(beta_qparams.zero_point, 0);
      beta_quantized_data_ = beta.template data<int32_t>();
      if (dequantize_output_) {
        beta_dequantized_.resize(C);
        beta_dequantized_data_ = beta_dequantized_.data();
        fbgemm::Dequantize<int32_t>(
            beta_quantized_data_, beta_dequantized_.data(), C, beta_qparams);
      }
    } else {
      beta_qparams.scale = X_qparams.scale * gamma_qparams.scale;
      beta_qparams.zero_point = 0;
      beta_quantized_.resize(C);
      beta_quantized_data_ = beta_quantized_.data();
      beta_dequantized_data_ = beta.template data<float>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < C; ++i) {
        beta_quantized_[i] = fbgemm::Quantize<int32_t>(
            beta_dequantized_data_[i],
            beta_qparams.zero_point,
            beta_qparams.scale,
            32);
      }
    }
    cached_X_qparams_scale_ = in_qparams_[INPUT].scale;
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::QuantizedGroupMomentsNCHW(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
    int32_t* mu,
    int32_t* rsig) {
  const int outer_size = N * G;
  const int inner_size = K * HxW;
  const auto& X_qparams = in_qparams_[INPUT];
  auto var_qparams = X_qparams;
  var_qparams.scale = X_qparams.scale * X_qparams.scale;
  var_qparams.zero_point = 0;
  rsig_dequantized_.resize(outer_size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_size; ++i) {
    int64_t sum = 0;
    int64_t sumsq = 0;
    if (GetCpuId().avx2()) {
      internal::VectorMomentsAVX2<T>(
          inner_size, X + i * inner_size, &sum, &sumsq);
    } else {
      ConstEigenVectorArrayMap<T> X_arr(X + i * inner_size, inner_size);
      sum = X_arr.template cast<int64_t>().sum();
      sumsq = X_arr.template cast<int64_t>().square().sum();
    }
    const float mean = static_cast<float>(sum) / static_cast<float>(inner_size);
    mu[i] = static_cast<int32_t>(std::round(mean)) - X_qparams.zero_point;
    const float var =
        static_cast<float>(sumsq) / static_cast<float>(inner_size) -
        mean * mean;
    rsig_dequantized_[i] = fbgemm::Dequantize<float>(var, var_qparams);
  }
  ComputeQuantizedInvStd(
      outer_size, rsig_dequantized_.data(), rsig_dequantized_.data(), rsig);
}

template <typename T>
void GroupNormDNNLowPOp<T>::QuantizedGroupMomentsNHWC(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
    int32_t* mu,
    int32_t* rsig) {
  const int outer_size = N * G;
  const int inner_size = K * HxW;
  const auto& X_qparams = in_qparams_[INPUT];
  auto var_qparams = X_qparams;
  var_qparams.scale = X_qparams.scale * X_qparams.scale;
  var_qparams.zero_point = 0;
  rsig_dequantized_.resize(outer_size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_size; ++i) {
    const int n = i / G;
    const int g = i % G;
    int64_t sum = 0;
    int64_t sumsq = 0;
    for (int j = 0; j < HxW; ++j) {
      const T* X_ptr = X + ((n * HxW + j) * G + g) * K;
      if (GetCpuId().avx2()) {
        internal::VectorMomentsAVX2<T>(K, X_ptr, &sum, &sumsq);
      } else {
        ConstEigenVectorArrayMap<T> X_arr(X + ((n * HxW + j) * G + g) * K, K);
        sum += X_arr.template cast<int64_t>().sum();
        sumsq += X_arr.template cast<int64_t>().square().sum();
      }
    }
    const float mean = static_cast<float>(sum) / static_cast<float>(inner_size);
    mu[i] = static_cast<int32_t>(std::round(mean)) - X_qparams.zero_point;
    const float var =
        static_cast<float>(sumsq) / static_cast<float>(inner_size) -
        mean * mean;
    rsig_dequantized_[i] = fbgemm::Dequantize<float>(var, var_qparams);
  }
  ComputeQuantizedInvStd(
      outer_size, rsig_dequantized_.data(), rsig_dequantized_.data(), rsig);
}

template <typename T>
void GroupNormDNNLowPOp<T>::DequantizedGroupMomentsNCHW(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
    float* mu,
    float* rsig) {
  const int C = G * K;
  const int size = N * C * HxW;
  const int outer_size = N * G;
  const int inner_size = K * HxW;
  X_dequantized_.resize(size);
  fbgemm::Dequantize<T>(X, X_dequantized_.data(), size, in_qparams_[INPUT]);
  const std::array<int, 2> X_dims = {outer_size, inner_size};
  const std::array<int, 2> Y_dims = {outer_size, 1};
  math::Moments<float, CPUContext>(
      2,
      X_dims.data(),
      Y_dims.data(),
      X_dequantized_.data(),
      mu,
      rsig,
      &context_);
  math::InvStd<float>(outer_size, epsilon_, rsig, rsig, &context_);
}

template <typename T>
void GroupNormDNNLowPOp<T>::DequantizedGroupMomentsNHWC(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
    float* mu,
    float* rsig) {
  const int C = G * K;
  const int size = N * C * HxW;
  const int outer_size = N * G;
  X_dequantized_.resize(size);
  fbgemm::Dequantize<T>(X, X_dequantized_.data(), size, in_qparams_[INPUT]);
  const std::array<int, 4> X_dims = {N, HxW, G, K};
  const std::array<int, 4> Y_dims = {N, 1, G, 1};
  math::Moments<float, CPUContext>(
      4,
      X_dims.data(),
      Y_dims.data(),
      X_dequantized_.data(),
      mu,
      rsig,
      &context_);
  math::InvStd<float>(outer_size, epsilon_, rsig, rsig, &context_);
}

template <typename T>
bool GroupNormDNNLowPOp<T>::RunOnDeviceWithOrderNCHW() {
  const auto& X = InputTensorCPU_(INPUT);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int HxW = X.size_from_dim(2);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  auto* Y = OutputTensorCPU_(0);
  Y->ResizeLike(X);
  std::vector<T> X_temp;
  const T* X_data = dnnlowp::QuantizeInputIfNeeded<T>(
      this, INPUT, in_qparams_[INPUT], X_temp);

  if (dequantize_output_) {
    float* Y_data = Y->template mutable_data<float>();
    if (N == 0) {
      return true;
    }
    mu_dequantized_.resize(N * G);
    rsig_dequantized_.resize(N * G);
    float* mu_data = mu_dequantized_.data();
    float* rsig_data = rsig_dequantized_.data();
    DequantizedGroupMomentsNCHW(N, G, K, HxW, X_data, mu_data, rsig_data);
    scale_dequantized_.resize(N * C);
    bias_dequantized_.resize(N * C);
    float* scale_data = scale_dequantized_.data();
    float* bias_data = bias_dequantized_.data();
    ComputeDequantizedFusedParams(
        N,
        G,
        K,
        mu_data,
        rsig_data,
        gamma_dequantized_data_,
        beta_dequantized_data_,
        scale_data,
        bias_data);
    AffineBatchChannelDequantizedNCHW(
        N, C, HxW, X_dequantized_.data(), scale_data, bias_data, Y_data);
  } else {
    T* Y_data = GetQuantizedOutputData_();
    if (N == 0) {
      return true;
    }
    mu_quantized_.resize(N * G);
    rsig_quantized_.resize(N * G);
    int32_t* mu_data = mu_quantized_.data();
    int32_t* rsig_data = rsig_quantized_.data();
    QuantizedGroupMomentsNCHW(N, G, K, HxW, X_data, mu_data, rsig_data);
    scale_quantized_.resize(N * C);
    bias_quantized_.resize(N * C);
    int32_t* scale_data = scale_quantized_.data();
    int32_t* bias_data = bias_quantized_.data();
    ComputeQuantizedFusedParams(
        N,
        G,
        K,
        mu_data,
        rsig_data,
        gamma_quantized_data_,
        beta_quantized_data_,
        scale_data,
        bias_data);
    AffineBatchChannelQuantizedNCHW(
        N, C, HxW, X_data, scale_data, bias_data, Y_data);
    dnnlowp::PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
  }
  MeasureQuantizationError_();
  return true;
}

template <typename T>
bool GroupNormDNNLowPOp<T>::RunOnDeviceWithOrderNHWC() {
  const auto& X = InputTensorCPU_(INPUT);
  const int ndim = X.dim();
  const int N = X.dim32(0);
  const int C = X.dim32(ndim - 1);
  const int HxW = X.size_between_dim(0, ndim - 1);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  auto* Y = OutputTensorCPU_(0);
  Y->ResizeLike(X);
  std::vector<T> X_temp;
  const T* X_data = dnnlowp::QuantizeInputIfNeeded<T>(
      this, INPUT, in_qparams_[INPUT], X_temp);

  if (dequantize_output_) {
    float* Y_data = Y->template mutable_data<float>();
    if (N == 0) {
      return true;
    }
    mu_dequantized_.resize(N * G);
    rsig_dequantized_.resize(N * G);
    float* mu_data = mu_dequantized_.data();
    float* rsig_data = rsig_dequantized_.data();
    DequantizedGroupMomentsNHWC(N, G, K, HxW, X_data, mu_data, rsig_data);
    scale_dequantized_.resize(N * C);
    bias_dequantized_.resize(N * C);
    float* scale_data = scale_dequantized_.data();
    float* bias_data = bias_dequantized_.data();
    ComputeDequantizedFusedParams(
        N,
        G,
        K,
        mu_data,
        rsig_data,
        gamma_dequantized_data_,
        beta_dequantized_data_,
        scale_data,
        bias_data);
    AffineBatchChannelDequantizedNHWC(
        N, C, HxW, X_dequantized_.data(), scale_data, bias_data, Y_data);
  } else {
    T* Y_data = GetQuantizedOutputData_();
    if (N == 0) {
      return true;
    }
    mu_quantized_.resize(N * G);
    rsig_quantized_.resize(N * G);
    int32_t* mu_data = mu_quantized_.data();
    int32_t* rsig_data = rsig_quantized_.data();
    QuantizedGroupMomentsNHWC(N, G, K, HxW, X_data, mu_data, rsig_data);
    scale_quantized_.resize(N * C);
    bias_quantized_.resize(N * C);
    int32_t* scale_data = scale_quantized_.data();
    int32_t* bias_data = bias_quantized_.data();
    ComputeQuantizedFusedParams(
        N,
        G,
        K,
        mu_data,
        rsig_data,
        gamma_quantized_data_,
        beta_quantized_data_,
        scale_data,
        bias_data);
    AffineBatchChannelQuantizedNHWC(
        N, C, HxW, X_data, scale_data, bias_data, Y_data);
    dnnlowp::PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
  }
  MeasureQuantizationError_();
  return true;
}

template <typename T>
void GroupNormDNNLowPOp<T>::ComputeQuantizedInvStd(
    const int N,
    const float* var,
    float* rsig,
    int32_t* rsig_quantized) {
  math::InvStd<float, CPUContext>(N, epsilon_, var, rsig, &context_);
  rsig_qparams_ = qfactory_->ChooseQuantizationParams(
      rsig,
      N,
      dnnlowp::QuantizationFactory::MIN_MAX_QUANTIZATION,
      qfactory_->GetWeightPrecision(),
      qfactory_->GetPreserveWeightSparsity());
  rsig_qparams_.zero_point = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    rsig_quantized[i] = fbgemm::Quantize<int32_t>(
        rsig[i], rsig_qparams_.zero_point, rsig_qparams_.scale, 32);
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::ComputeQuantizedFusedParams(
    const int N,
    const int G,
    const int K,
    const int32_t* mu,
    const int32_t* rsig,
    const int32_t* gamma,
    const int32_t* beta,
    int32_t* scale,
    int32_t* bias) {
  const int C = G * K;
  ConstEigenArrayMap<int32_t> gamma_arr(gamma, K, G);
  const auto& X_qparams = in_qparams_[INPUT];
  const auto& gamma_qparams = in_qparams_[GAMMA];
  internal_qparams_.scale =
      rsig_qparams_.scale * gamma_qparams.scale * X_qparams.scale;
  internal_qparams_.zero_point = 0;
  internal_qparams_.precision = 32;
  const float real_multiplier = 1.0f / rsig_qparams_.scale;
  const auto beta_requantization_params =
      qfactory_->ChooseRequantizationMultiplier(
          real_multiplier, internal_qparams_);
  for (int i = 0; i < C; ++i) {
    bias[i] = fbgemm::Requantize<int32_t>(
        beta[i],
        internal_qparams_.zero_point,
        beta_requantization_params.multiplier,
        beta_requantization_params.right_shift,
        internal_qparams_.precision,
        true);
  }

  if (GetCpuId().avx2()) {
    internal::ComputeQuantizedFusedParamsAVX2(
        N, G, K, X_qparams.zero_point, mu, rsig, gamma, scale, bias);
  } else {
    ConstEigenArrayMap<int32_t> beta_arr(bias, K, G);
    // Reverse order for-loop to avoid overriding bias data.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = N - 1; i >= 0; --i) {
      EigenArrayMap<int32_t> scale_arr(scale + i * C, K, G);
      scale_arr = gamma_arr.rowwise() *
          ConstEigenVectorArrayMap<int32_t>(rsig + i * G, G).transpose();
      EigenArrayMap<int32_t>(bias + i * C, K, G) = beta_arr -
          scale_arr.rowwise() *
              (ConstEigenVectorArrayMap<int32_t>(mu + i * G, G).transpose() +
               X_qparams.zero_point);
    }
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::ComputeDequantizedFusedParams(
    const int N,
    const int G,
    const int K,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float* beta,
    float* scale,
    float* bias) {
  const int C = G * K;
  ConstEigenArrayMap<float> gamma_arr(gamma, K, G);
  ConstEigenArrayMap<float> beta_arr(beta, K, G);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<float> scale_arr(scale + i * C, K, G);
    scale_arr = gamma_arr.rowwise() *
        ConstEigenVectorArrayMap<float>(rsig + i * G, G).transpose();
    EigenArrayMap<float>(bias + i * C, K, G) = beta_arr -
        scale_arr.rowwise() *
            ConstEigenVectorArrayMap<float>(mu + i * G, G).transpose();
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::AffineBatchChannelQuantizedNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y) {
  const float real_multiplier = internal_qparams_.scale / out_qparams_.scale;
  const auto out_requantization_params =
      qfactory_->ChooseRequantizationMultiplier(real_multiplier, out_qparams_);
  if (GetCpuId().avx2()) {
    internal::AffineBatchChannelAndRequantizeNCHWAVX2<T>(
        N, C, HxW, out_requantization_params, X, scale, bias, Y);
  } else {
    const int size = N * C * HxW;
    Y_int32_.resize(size);
    int32_t* Y_int32_data = Y_int32_.data();
    EigenArrayMap<int32_t>(Y_int32_data, HxW, N * C) =
        (ConstEigenArrayMap<T>(X, HxW, N * C)
             .template cast<int32_t>()
             .rowwise() *
         ConstEigenVectorArrayMap<int32_t>(scale, N * C).transpose())
            .rowwise() +
        ConstEigenVectorArrayMap<int32_t>(bias, N * C).transpose();
    fbgemm::Requantize<T>(Y_int32_data, Y, size, out_requantization_params);
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::AffineBatchChannelQuantizedNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y) {
  const int size = N * C * HxW;
  const int stride = HxW * C;
  const float real_multiplier = internal_qparams_.scale / out_qparams_.scale;
  const auto out_requantization_params =
      qfactory_->ChooseRequantizationMultiplier(real_multiplier, out_qparams_);
  if (GetCpuId().avx2()) {
    internal::AffineBatchChannelAndRequantizeNHWCAVX2<T>(
        N, C, HxW, out_requantization_params, X, scale, bias, Y);
  } else {
    Y_int32_.resize(size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
      EigenArrayMap<int32_t>(Y_int32_.data() + i * stride, C, HxW) =
          (ConstEigenArrayMap<T>(X + i * stride, C, HxW)
               .template cast<int32_t>()
               .colwise() *
           ConstEigenVectorArrayMap<int32_t>(scale + i * C, C))
              .colwise() +
          ConstEigenVectorArrayMap<int32_t>(bias + i * C, C);
    }
    fbgemm::Requantize<T>(Y_int32_.data(), Y, size, out_requantization_params);
  }
}

template <typename T>
void GroupNormDNNLowPOp<T>::AffineBatchChannelDequantizedNCHW(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  EigenArrayMap<float>(Y, HxW, N * C) =
      (ConstEigenArrayMap<float>(X, HxW, N * C).rowwise() *
       ConstEigenVectorArrayMap<float>(scale, N * C).transpose())
          .rowwise() +
      ConstEigenVectorArrayMap<float>(bias, N * C).transpose();
}

template <typename T>
void GroupNormDNNLowPOp<T>::AffineBatchChannelDequantizedNHWC(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int stride = HxW * C;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<float>(Y + i * stride, C, HxW) =
        (ConstEigenArrayMap<float>(X + i * stride, C, HxW).colwise() *
         ConstEigenVectorArrayMap<float>(scale + i * C, C))
            .colwise() +
        ConstEigenVectorArrayMap<float>(bias + i * C, C);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    GroupNorm,
    DNNLOWP,
    GroupNormDNNLowPOp<uint8_t>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8GroupNorm,
    DNNLOWP,
    GroupNormDNNLowPOp<uint8_t>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8GroupNorm).NumInputs(3).NumOutputs({1, 3});

} // namespace caffe2
