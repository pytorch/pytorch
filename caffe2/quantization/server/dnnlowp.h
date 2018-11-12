#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <limits>

#include <x86intrin.h>

#include "caffe2/quantization/server/dynamic_histogram.h"
#include "caffe2/utils/cpuid.h"

namespace dnnlowp {

// Copied from gemmlowp
//
// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct TensorQuantizationParams {
  float scale;
  std::int32_t zero_point;
  int precision;
  float Min() const;
  float Max() const;
};

// Parameters when we scale from one quantization parameter to another
struct RequantizationParams {
  float real_multiplier;
  std::int32_t multiplier;
  int right_shift;
  TensorQuantizationParams target_qparams;
};

////////////////////////////////////////////////////////////////////////////////
// Utility functions

/// Clamp src in T1 to the desired precision and convert it to T2
template <typename T1, typename T2 = std::uint8_t>
T2 clamp(T1 src, int precision, bool is_signed = false)
// TODO: T26263653 fix signed-integer-overflow undefined behavior
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
    __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
{
  std::int32_t min = is_signed ? -(1LL << (precision - 1)) : 0;
  std::int32_t max =
    is_signed ? ((1LL << (precision - 1)) - 1) : (1LL << precision) - 1;

  // Make sure T1 and T2 can represent the precision
  assert(min >= std::numeric_limits<T1>::lowest());
  assert(min >= std::numeric_limits<T2>::lowest());
  assert(max <= std::numeric_limits<T1>::max());
  assert(max <= std::numeric_limits<T2>::max());

  return std::min<T1>(std::max<T1>(src, min), max);
}

/// Quantize src using zero_point and scale, clamp to the specified precision,
/// and convert it to type T
template <typename T>
T Quantize(float src, std::int32_t zero_point, float scale,
           int result_precision,
           bool result_is_signed = std::is_signed<T>::value) {
  const float transformed_val = zero_point + src / scale;
  return clamp<std::int64_t, T>(
      (std::int64_t)round(transformed_val),
      result_precision, result_is_signed);
}

template <typename T>
T Quantize(float src, const TensorQuantizationParams& qparams) {
  return dnnlowp::Quantize<T>(
    src, qparams.zero_point, qparams.scale, qparams.precision);
}

template <typename T>
void Quantize(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams);

template <typename T>
float Dequantize(T src, const TensorQuantizationParams& qparams) {
  return qparams.scale * (src - qparams.zero_point);
}

template <typename T>
void Dequantize(const T *src, float *dst, int len,
                const TensorQuantizationParams& qparams) {
  for (std::size_t i = 0; i < len; i++) {
    dst[i] = Dequantize(src[i], qparams);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Requantization (pure-integer)

std::int64_t SaturatingRoundingMulWithShift(
  std::int32_t a, std::int32_t b, int right_shift);

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    std::int32_t zero_point,
    std::int32_t multiplier, int right_shift,
    int result_precision, bool result_is_signed = false) {
  std::int64_t quantized_down =
    zero_point + SaturatingRoundingMulWithShift(src, multiplier, right_shift);
  return clamp<std::int64_t, T>(
    quantized_down, result_precision, result_is_signed);
}

template <typename T>
T RequantizeFixedPoint(
    std::int32_t src, // int32 input before requantization
    const RequantizationParams& params) {
  return Requantize<T>(
    src, params.target_qparams.zero_point,
    params.multiplier, params.right_shift,
    params.target_qparams.precision);
}

void RequantizeFixedPointAvx2(
  const std::int32_t *src, std::uint8_t *dst, int len,
  const RequantizationParams& params);

template <typename T>
void RequantizeFixedPoint(
    const std::int32_t *src, T *dst, int len,
    const RequantizationParams& params) {
  if (std::is_same<T, uint8_t>::value &&
      params.target_qparams.precision == 8 &&
      caffe2::GetCpuId().avx2()) {
    RequantizeFixedPointAvx2(src, dst, len, params);
  } else {
    for (int i = 0; i < len; ++i) {
      dst[i] = RequantizeFixedPoint<T>(src[i], params);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Requantization (with floats)

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    std::int32_t zero_point,
    float multiplier,
    int result_precision, bool result_is_signed = false) {
  long quantized_down = zero_point + std::lrintf(src * multiplier);
  return clamp<long, T>(quantized_down, result_precision, result_is_signed);
}

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    const RequantizationParams& params) {
  return Requantize<T>(
      src,
      params.target_qparams.zero_point,
      params.real_multiplier,
      params.target_qparams.precision);
}

void RequantizeAvx2(
  const std::int32_t *src, std::uint8_t *dst, int len,
  const RequantizationParams& params);

template <typename T>
void Requantize(
    const std::int32_t *src, T *dst, int len,
    const RequantizationParams& params);

// Represents a quantization scheme that provides quantization parameter based
// on distribution of data to be quantized.
class QuantizationFactory {
 public:
  enum QuantizationKind {
    // A simple quantization scheme that determines quantization parameter by
    // just looking at min/max.
    MIN_MAX_QUANTIZATION,
    // Minimizes L2 norm of quantization error
    L2_MIN_QUANTIZATION,
    // fast search to remove histogram outliers and approximate L2 min
    L2_MIN_QUANTIZATION_APPROX,
    // Minimizes Kullback-Leibler divergence
    KL_MIN_QUANTIZATION,
    // Take 99 percentail (only works with sparsity preserving quantization)
    P99_QUANTIZATION,
    L1_MIN_QUANTIZATION,
  };

  /// Get the default factory whose policy is determined by gflags
  static QuantizationFactory *GetDefaultInstance();

  /// Choose quantization scale and zero_point that maps
  /// floating-point range [min, max] to the integer range of the specified
  /// precision
  TensorQuantizationParams ChooseQuantizationParams(
      float min, float max,
      int precision, bool preserve_sparsity, bool is_signed = false) const {
    TensorQuantizationParams qparams =
      ChooseQuantizationParams_(
        min, max,
        is_signed ? -(1 << (precision - 1)) : 0,
        is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
        preserve_sparsity);
    qparams.precision = precision;
    return qparams;
  }

  /// Choose quantization scale and zero_point that maps
  /// floating-point range [min, max] to the default integer range of
  /// this quantization factory
  TensorQuantizationParams
    ChooseQuantizationParams(
      float min, float max, bool is_weight = false) const {
    return ChooseQuantizationParams(
        min, max,
        is_weight ? GetWeightPrecision() : GetActivationPrecision(),
        is_weight
          ? GetPreserveWeightSparsity() : GetPreserveActivationSparsity());
  }

  /// Choose quantization based on the values in an array to optimize the
  /// quantization errors ignoring a few outliers
  TensorQuantizationParams ChooseQuantizationParams(
      const float *values, int len, QuantizationKind kind,
      int precision, bool preserve_sparsity) const;

  TensorQuantizationParams ChooseQuantizationParams(
      const float *values, int len, bool is_weight = false) const;

  /// Choose quantization based on histogram of values to optimize the
  /// quantization errors ignoring a few outliers
  TensorQuantizationParams ChooseQuantizationParams(
      const Histogram& hist, QuantizationKind kind,
      int precision, bool preserve_sparsity) const;

  TensorQuantizationParams ChooseQuantizationParams(
      const Histogram& hist, bool is_weight = false) const;

  // Given a real_multiplier, produces a pair (quantized_multiplier,
  // right_shift) where quantized_multiplier is an int32 representing a
  // fixed-point value (in practice we only produce positive values) and
  // right_shift is an amount to shift right by, so that the floating-point
  // multiplication of some int32 input value by real_multiplier,
  //
  //   return static_cast<int32>(int32_value * real_multiplier);
  //
  // is best approximated by the integer-arithmetic-only code
  //
  //   return RoundingRightShift(
  //       Multiplication(int32_value, quantized_multiplier),
  //       right_shift);
  //
  // Note: all this code only needs to run offline to generate the quantized
  // neural network workload, not at runtime on the device on which quantized
  // neural networks need to run. So it's not performance-critical at all.
  RequantizationParams ChooseRequantizationMultiplier(
      float real_multiplier, TensorQuantizationParams target_qparams) const;

  int GetActivationPrecision() const {
    return activation_precision_;
  }

  int GetWeightPrecision() const {
    return weight_precision_;
  }

  int GetEltwiseQuantizePrecision() const {
    return eltwise_quantize_precision_;
  }

  bool GetPreserveActivationSparsity() const {
    return preserve_activation_sparsity_;
  }

  bool GetPreserveWeightSparsity() const {
    return preserve_weight_sparsity_;
  }

  QuantizationKind GetActivationKind() const { return activation_kind_; }
  QuantizationKind GetWeightKind() const { return weight_kind_; }

  explicit QuantizationFactory(
      int activation_precision = 8,
        // precision used for activations in main operations like matmul
      int weight_precision = 8, // precision used for weights
      int requantization_multiplier_precision = 32,
        // precision used for the requantization multiplier
      int eltwise_quantize_precision = 16,
        // precision used for element-wise addition
      bool preserve_activation_sparsity = false,
        // preserve zeros in quantization
      bool preserve_weight_sparsity = false,
        // preserve zeros in quantization
      bool force_scale_power_of_two = false,
        // restrict scaling to a power of two
      QuantizationKind activation_kind = MIN_MAX_QUANTIZATION,
      QuantizationKind weight_kind = MIN_MAX_QUANTIZATION);

 private:
  /// Choose quantization scale and zero_point that maps
  /// floating-point range [min, max] to integer range [qmin, qmax]
  TensorQuantizationParams ChooseQuantizationParams_(
      float min, float max, std::int32_t qmin, std::int32_t qmax,
      bool preserve_sparsity) const;

  void ChooseRequantizationMultiplier_(float real_multiplier,
                                       std::int32_t* quantized_multiplier,
                                       int* right_shift) const;

  int activation_precision_;
  int weight_precision_;
  int requantization_multiplier_precision_;
  int eltwise_quantize_precision_;
  bool preserve_activation_sparsity_;
  bool preserve_weight_sparsity_;
  bool force_scale_power_of_two_;
  QuantizationKind activation_kind_, weight_kind_;
}; // class QuantizationFactory

/**
 * Parse a string to QuantizationKind
 */
QuantizationFactory::QuantizationKind StringToKind(const std::string& s);

/**
 * Find the min and max value in a float matrix.
 */
void FindMinMax(const float *m, float* min, float* max, int len);

} // namesapce dnnlowp
