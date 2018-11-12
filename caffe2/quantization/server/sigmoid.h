#ifndef DNNLOWP_SIGMOID_H
#define DNNLOWP_SIGMOID_H

#include "tanh.h"

namespace dnnlowp {

/**
 * sigmoid(x) = (tanh(x/2) + 1)/2
 * Quantized sigmoid is computed as tanh under the hood, we just use different
 * input/output quantization parameters.
 */
template<typename T>
class Sigmoid {
 public:
  Sigmoid(double max_abs_err_ = Tanh<T>::DEFAULT_MAX_ABS_ERR);

  T Compute(T x) const;

  TensorQuantizationParams GetInputQuantizationParams() const {
    return in_qparams_;
  }
  TensorQuantizationParams GetOutputQuantizationParams() const {
    return out_qparams_;
  }

 private:
  const int num_in_bits_ = Tanh<T>::DEFAULT_NUM_IN_BITS;
  const int num_out_bits_ = Tanh<T>::DEFAULT_NUM_OUT_BITS;
  Tanh<T> tanh_;
  TensorQuantizationParams in_qparams_, out_qparams_;
}; // class Sigmoid

} // namespace dnnlowp

#endif // DNNLOWP_SIGMOID_H
