#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

#include <x86intrin.h>

#include <fbgemm/QuantUtils.h>

#include "caffe2/quantization/server/dynamic_histogram.h"
#include "caffe2/utils/cpuid.h"

namespace dnnlowp {

using fbgemm::RequantizationParams;
using fbgemm::TensorQuantizationParams;

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
  static QuantizationFactory* GetDefaultInstance();

  /// Choose quantization scale and zero_point that maps
  /// floating-point range [min, max] to the integer range of the specified
  /// precision
  TensorQuantizationParams ChooseQuantizationParams(
      float min,
      float max,
      int precision,
      bool preserve_sparsity,
      bool is_signed = false) const {
    TensorQuantizationParams qparams = fbgemm::ChooseQuantizationParams(
        min,
        max,
        is_signed ? -(1 << (precision - 1)) : 0,
        is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
        preserve_sparsity,
        force_scale_power_of_two_);
    qparams.precision = precision;
    return qparams;
  }

  /// Choose quantization scale and zero_point that maps
  /// floating-point range [min, max] to the default integer range of
  /// this quantization factory
  TensorQuantizationParams
  ChooseQuantizationParams(float min, float max, bool is_weight = false) const {
    return ChooseQuantizationParams(
        min,
        max,
        is_weight ? GetWeightPrecision() : GetActivationPrecision(),
        is_weight ? GetPreserveWeightSparsity()
                  : GetPreserveActivationSparsity());
  }

  /// Choose quantization based on the values in an array to optimize the
  /// quantization errors ignoring a few outliers
  TensorQuantizationParams ChooseQuantizationParams(
      const float* values,
      int len,
      QuantizationKind kind,
      int precision,
      bool preserve_sparsity) const;

  TensorQuantizationParams ChooseQuantizationParams(
      const float* values,
      int len,
      bool is_weight = false) const;

  /// Choose quantization based on histogram of values to optimize the
  /// quantization errors ignoring a few outliers
  TensorQuantizationParams ChooseQuantizationParams(
      const Histogram& hist,
      QuantizationKind kind,
      int precision,
      bool preserve_sparsity,
      bool is_weight = false) const;

  TensorQuantizationParams ChooseQuantizationParams(
      const Histogram& hist,
      bool is_weight = false) const;

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
      float real_multiplier,
      TensorQuantizationParams target_qparams) const;

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

  QuantizationKind GetActivationKind() const {
    return activation_kind_;
  }
  QuantizationKind GetWeightKind() const {
    return weight_kind_;
  }

  void SetWeightP99Threshold(float threshold) {
    weight_p99_threshold_ = threshold;
  }
  void SetActivationP99Threshold(float threshold) {
    activation_p99_threshold_ = threshold;
  }

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
      QuantizationKind weight_kind = MIN_MAX_QUANTIZATION,
      float weight_p99_threshold = 0.99,
      // P99 percentage to select out from the full histogram for weights

      float activation_p99_threshold = 0.99
      // P99 percentage to select out from the full histogram for activations
  );

 private:
  int activation_precision_;
  int weight_precision_;
  int requantization_multiplier_precision_;
  int eltwise_quantize_precision_;
  bool preserve_activation_sparsity_;
  bool preserve_weight_sparsity_;
  bool force_scale_power_of_two_;
  QuantizationKind activation_kind_, weight_kind_;
  float weight_p99_threshold_;
  float activation_p99_threshold_;
}; // class QuantizationFactory

/**
 * Parse a string to QuantizationKind
 */
QuantizationFactory::QuantizationKind StringToKind(const std::string& s);

std::vector<float>
adjust_hist_to_include_zero(const Histogram& hist, float* min, float* max);

} // namespace dnnlowp
