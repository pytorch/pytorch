#pragma once

#include "dnnlowp.h"

#include <cmath>
#include <vector>

namespace dnnlowp {

/**
 * We use the 3-region approach described in "Efficient VLSI Implementation of
 * Neural Networks with Hyperbolic Tangent Activation Function", IEEE
 * Transactions on Very Large Scale Integration Systems, Zamanlooy and
 * Mirhassani.
 * The pass region (x < x_pq) is approximated as x.
 * The saturation region (x >= x_sq) is approximated as 1.
 * The processing region (x_pq <= x < x_sq) is divided into sub-ranges and the
 * average value of tanh(x) is used per sub-range.
 */
template <typename T>
class Tanh {
 public:
  Tanh(double max_abs_err = DEFAULT_MAX_ABS_ERR);

  T Compute(T x) const;

  TensorQuantizationParams GetInputQuantizationParams() const {
    return in_qparams_;
  }
  TensorQuantizationParams GetOutputQuantizationParams() const {
    return out_qparams_;
  }

  int GetPassRegionEnd() const {
    return x_pq_index_;
  }

  float GetPassRegionEndDequantized() const {
    return fbgemm::Dequantize<T>(
        static_cast<uint8_t>(x_pq_index_ + in_qparams_.zero_point),
        in_qparams_);
  }

  float GetSaturationRegionBegin() const {
    return fbgemm::Dequantize<T>(
        static_cast<T>((1 << num_in_bits_) - 1), in_qparams_);
  }

  static constexpr double DEFAULT_MAX_ABS_ERR = 0.02;
  static constexpr int DEFAULT_NUM_IN_BITS = 8;
  static constexpr int DEFAULT_NUM_OUT_BITS = 8;

 private:
  const double max_abs_err_;
  const int num_in_bits_ = DEFAULT_NUM_IN_BITS;
  const int num_out_bits_ = DEFAULT_NUM_OUT_BITS;

  int x_pq_index_;
  std::vector<T> processing_region_lut_;
  TensorQuantizationParams in_qparams_, out_qparams_;
}; // class TanhApproximation

} // namespace dnnlowp
