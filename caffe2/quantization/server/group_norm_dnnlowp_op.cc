#include "caffe2/quantization/server/group_norm_dnnlowp_op.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
void SegmentMomentsAVX2(
    const int N,
    const T* src,
    int64_t* sum,
    int64_t* sumsq);

template <>
void SegmentMomentsAVX2<uint8_t>(
    const int N,
    const uint8_t* src,
    int64_t* sum,
    int64_t* sumsq) {
  constexpr int kVLen = 16;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  const __m256i kOneInt16 = _mm256_set1_epi16(0x01);
  __m256i sum_v = _mm256_setzero_si256();
  __m256i sumsq_v = _mm256_setzero_si256();
  for (int i = 0; i < n; i += kVLen) {
    const __m256i cur_v =
        _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src + i)));
    sum_v = _mm256_add_epi32(sum_v, _mm256_madd_epi16(cur_v, kOneInt16));
    sumsq_v = _mm256_add_epi32(sumsq_v, _mm256_madd_epi16(cur_v, cur_v));
  }
  std::array<int32_t, 8> sum_arr;
  std::array<int32_t, 8> sumsq_arr;
  _mm256_storeu_si256((__m256i*)(sum_arr.data()), sum_v);
  _mm256_storeu_si256((__m256i*)(sumsq_arr.data()), sumsq_v);
  for (int i = 0; i < 8; ++i) {
    *sum += static_cast<int64_t>(sum_arr[i]);
    *sumsq += static_cast<int64_t>(sumsq_arr[i]);
  }
  for (int i = 0; i < r; ++i) {
    *sum += static_cast<int64_t>(src[n + i]);
    *sumsq +=
        static_cast<int64_t>(src[n + i]) * static_cast<int64_t>(src[n + i]);
  }
}

template <typename T>
void VectorMomentsAVX2(const int N, const T* src, int64_t* sum, int64_t* sumsq);

template <>
void VectorMomentsAVX2<uint8_t>(
    const int N,
    const uint8_t* src,
    int64_t* sum,
    int64_t* sumsq) {
  constexpr int kVLen = 32768;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  for (int i = 0; i < n; i += kVLen) {
    SegmentMomentsAVX2<uint8_t>(kVLen, src + i, sum, sumsq);
  }
  if (r > 0) {
    SegmentMomentsAVX2<uint8_t>(r, src + n, sum, sumsq);
  }
}

void ComputeQuantizedFusedParamsAVX2(
    const int N,
    const int G,
    const int K,
    const int32_t X_zero_point,
    const int32_t* mu,
    const int32_t* rsig,
    const int32_t* gamma,
    int32_t* scale,
    int32_t* bias) {
  constexpr int kVLen = 8;
  const int k = K / kVLen * kVLen;
  const int r = K % kVLen;
  for (int n = N - 1; n >= 0; --n) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int g = 0; g < G; ++g) {
      const __m256i mu_v = _mm256_set1_epi32(mu[n * G + g] + X_zero_point);
      const __m256i rsig_v = _mm256_set1_epi32(rsig[n * G + g]);
      for (int i = 0; i < k; i += kVLen) {
        const __m256i gamma_v =
            _mm256_loadu_si256((const __m256i*)(gamma + g * K + i));
        const __m256i beta_v =
            _mm256_loadu_si256((const __m256i*)(bias + g * K + i));
        __m256i scale_v = _mm256_mullo_epi32(gamma_v, rsig_v);
        __m256i bias_v =
            _mm256_sub_epi32(beta_v, _mm256_mullo_epi32(scale_v, mu_v));
        const int offset = (n * G + g) * K + i;
        _mm256_storeu_si256((__m256i*)(scale + offset), scale_v);
        _mm256_storeu_si256((__m256i*)(bias + offset), bias_v);
      }
      for (int i = 0; i < r; ++i) {
        const int offset = (n * G + g) * K + k + i;
        scale[offset] = gamma[g * K + k + i] * rsig[n * G + g];
        bias[offset] = bias[g * K + k + i] -
            scale[offset] * (mu[n * G + g] + X_zero_point);
      }
    }
  }
}

#define INIT_REQUANTIZE_AVX2                                      \
  const __m256i b = _mm256_set1_epi32(params.multiplier);         \
  const __m256i prev_shift_nudge = _mm256_set1_epi64x(            \
      (1ll << (params.right_shift - 1)) + 0x8000000000000000ULL); \
  const __m256i post_shift_nudge = _mm256_set1_epi64x(            \
      params.target_qparams.zero_point -                          \
      (0x8000000000000000ULL >> params.right_shift));             \
  const __m256i min_v =                                           \
      _mm256_set1_epi32(std::numeric_limits<uint8_t>::min());     \
  const __m256i max_v =                                           \
      _mm256_set1_epi32(std::numeric_limits<uint8_t>::max());     \
  const __m256i shuffle_mask_v = _mm256_set_epi8(                 \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0x0c,                                                       \
      0x08,                                                       \
      0x04,                                                       \
      0x00,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0x0c,                                                       \
      0x08,                                                       \
      0x04,                                                       \
      0x00);                                                      \
  const __m256i permute_mask_v =                                  \
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

#define REQUANTIZE_AVX2(params, src, dst)                                     \
  do {                                                                        \
    __m256i a_v = (src);                                                      \
    __m256i a_even_v = a_v;                                                   \
    __m256i a_odd_v = _mm256_srli_si256(a_v, 4);                              \
    __m256i ab_even_v = _mm256_mul_epi32(a_even_v, b);                        \
    __m256i ab_odd_v = _mm256_mul_epi32(a_odd_v, b);                          \
    __m256i even_rounded_v = _mm256_add_epi64(ab_even_v, prev_shift_nudge);   \
    __m256i odd_rounded_v = _mm256_add_epi64(ab_odd_v, prev_shift_nudge);     \
    __m256i even_result_v = _mm256_add_epi64(                                 \
        _mm256_srli_epi64(even_rounded_v, params.right_shift),                \
        post_shift_nudge);                                                    \
    __m256i odd_result_v = _mm256_add_epi64(                                  \
        _mm256_srli_epi64(odd_rounded_v, params.right_shift),                 \
        post_shift_nudge);                                                    \
    odd_result_v = _mm256_slli_si256(odd_result_v, 4);                        \
    __m256i result_v = _mm256_blend_epi32(even_result_v, odd_result_v, 0xaa); \
    __m256i clipped_v =                                                       \
        _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, result_v));           \
    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);               \
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);       \
    *(int64_t*)(dst) = _mm256_extract_epi64(clipped_v, 0);                    \
  } while (false)

template <typename T>
void AffineBatchChannelAndRequantizeNCHWAVX2(
    const int N,
    const int C,
    const int HxW,
    const dnnlowp::RequantizationParams& params,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y);

template <>
void AffineBatchChannelAndRequantizeNCHWAVX2<uint8_t>(
    const int N,
    const int C,
    const int HxW,
    const dnnlowp::RequantizationParams& params,
    const uint8_t* X,
    const int32_t* scale,
    const int32_t* bias,
    uint8_t* Y) {
  INIT_REQUANTIZE_AVX2;
  constexpr int kVLen = 8;
  const int outer_size = N * C;
  const int n = HxW / kVLen * kVLen;
  const int r = HxW % kVLen;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_size; ++i) {
    const uint8_t* X_ptr = X + i * HxW;
    uint8_t* Y_ptr = Y + i * HxW;
    const __m256i scale_v = _mm256_set1_epi32(scale[i]);
    const __m256i bias_v = _mm256_set1_epi32(bias[i]);
    for (int j = 0; j < n; j += kVLen) {
      const __m256i cur_v =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(X_ptr + j)));
      REQUANTIZE_AVX2(
          params,
          _mm256_add_epi32(_mm256_mullo_epi32(cur_v, scale_v), bias_v),
          (Y_ptr + j));
    }
    for (int j = 0; j < r; ++j) {
      Y_ptr[n + j] = dnnlowp::Requantize<uint8_t>(
          static_cast<int32_t>(X_ptr[n + j]) * scale[i] + bias[i], params);
    }
  }
}

template <typename T>
void AffineBatchChannelAndRequantizeNHWCAVX2(
    const int N,
    const int C,
    const int HxW,
    const dnnlowp::RequantizationParams& params,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y);

template <>
void AffineBatchChannelAndRequantizeNHWCAVX2<uint8_t>(
    const int N,
    const int C,
    const int HxW,
    const dnnlowp::RequantizationParams& params,
    const uint8_t* X,
    const int32_t* scale,
    const int32_t* bias,
    uint8_t* Y) {
  INIT_REQUANTIZE_AVX2;
  constexpr int kVLen = 8;
  const int outer_size = N * HxW;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_size; ++i) {
    const int c = i / HxW * C;
    const int n = C / kVLen * kVLen;
    const int r = C % kVLen;
    const uint8_t* X_ptr = X + i * C;
    uint8_t* Y_ptr = Y + i * C;
    for (int j = 0; j < n; j += kVLen) {
      const __m256i cur_v =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(X_ptr + j)));
      const __m256i scale_v =
          _mm256_loadu_si256((const __m256i*)(scale + c + j));
      const __m256i bias_v = _mm256_loadu_si256((const __m256i*)(bias + c + j));
      REQUANTIZE_AVX2(
          params,
          _mm256_add_epi32(_mm256_mullo_epi32(cur_v, scale_v), bias_v),
          (Y_ptr + j));
    }
    for (int j = 0; j < r; ++j) {
      Y_ptr[n + j] = dnnlowp::Requantize<uint8_t>(
          static_cast<int32_t>(X_ptr[n + j]) * scale[c + n + j] +
              bias[c + n + j],
          params);
    }
  }
}

#undef REQUANTIZE_AVX2
#undef INIT_REQUANTIZE_AVX2

} // namespace

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
      if (OperatorBase::InputIsType<int8::Int8TensorCPU>(GAMMA)) {
        const auto& gamma_int8 =
            OperatorBase::Input<int8::Int8TensorCPU>(GAMMA);
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
          dnnlowp::Dequantize<int32_t>(
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
    gamma_quantized_[i] = dnnlowp::Quantize<int32_t>(
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
    if (OperatorBase::InputIsType<int8::Int8TensorCPU>(BETA)) {
      const auto& beta_int8 = OperatorBase::Input<int8::Int8TensorCPU>(BETA);
      beta_qparams.scale = beta_int8.scale;
      beta_qparams.zero_point = beta_int8.zero_point;
      CAFFE_ENFORCE_LE(
          std::abs(beta_qparams.scale - X_qparams.scale * gamma_qparams.scale),
          1e-4);
      CAFFE_ENFORCE_EQ(beta_qparams.zero_point, 0);
      beta_quantized_data_ = beta.template data<int32_t>();
      if (dequantize_output_) {
        beta_dequantized_.resize(C);
        beta_dequantized_data_ = beta_dequantized_.data();
        dnnlowp::Dequantize<int32_t>(
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
        beta_quantized_[i] = dnnlowp::Quantize<int32_t>(
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
      VectorMomentsAVX2<T>(inner_size, X + i * inner_size, &sum, &sumsq);
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
    rsig_dequantized_[i] = dnnlowp::Dequantize<float>(var, var_qparams);
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
        VectorMomentsAVX2<T>(K, X_ptr, &sum, &sumsq);
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
    rsig_dequantized_[i] = dnnlowp::Dequantize<float>(var, var_qparams);
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
  dnnlowp::Dequantize<T>(X, X_dequantized_.data(), size, in_qparams_[INPUT]);
  const std::array<int, 2> dims = {outer_size, inner_size};
  const int axis = 1;
  math::Moments<float, CPUContext>(
      2, dims.data(), 1, &axis, X_dequantized_.data(), mu, rsig, &context_);
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
  dnnlowp::Dequantize<T>(X, X_dequantized_.data(), size, in_qparams_[INPUT]);
  const std::array<int, 4> dims = {N, HxW, G, K};
  const std::array<int, 2> axes = {1, 3};
  math::Moments<float, CPUContext>(
      4,
      dims.data(),
      2,
      axes.data(),
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
  const int HxW = X.size() / (N * C);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  auto* Y = OutputTensorCPU_(0);
  Y->ResizeLike(X);
  std::vector<T> X_temp;
  const T* X_data = dnnlowp::QuantizeInputIfNeeded<T>(
      this, INPUT, in_qparams_[INPUT], X_temp, qfactory_.get());

  if (dequantize_output_) {
    float* Y_data = Y->template mutable_data<float>();
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
    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
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
  const int HxW = X.size() / (N * C);
  const int G = group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  auto* Y = OutputTensorCPU_(0);
  Y->ResizeLike(X);
  std::vector<T> X_temp;
  const T* X_data = dnnlowp::QuantizeInputIfNeeded<T>(
      this, INPUT, in_qparams_[INPUT], X_temp, qfactory_.get());
  if (dequantize_output_) {
    float* Y_data = Y->template mutable_data<float>();
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
    PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);
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
    rsig_quantized[i] = dnnlowp::Quantize<int32_t>(
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
    bias[i] = dnnlowp::Requantize<int32_t>(
        beta[i],
        internal_qparams_.zero_point,
        beta_requantization_params.multiplier,
        beta_requantization_params.right_shift,
        internal_qparams_.precision,
        true);
  }

  if (GetCpuId().avx2()) {
    ComputeQuantizedFusedParamsAVX2(
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
    AffineBatchChannelAndRequantizeNCHWAVX2<T>(
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
    dnnlowp::Requantize<T>(Y_int32_data, Y, size, out_requantization_params);
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
    AffineBatchChannelAndRequantizeNHWCAVX2<T>(
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
    dnnlowp::Requantize<T>(Y_int32_.data(), Y, size, out_requantization_params);
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

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    GroupNorm,
    DNNLOWP,
    GroupNormDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8GroupNorm,
    DNNLOWP,
    GroupNormDNNLowPOp<uint8_t>);

OPERATOR_SCHEMA(Int8GroupNorm).NumInputs(3).NumOutputs({1, 3});

} // namespace caffe2
