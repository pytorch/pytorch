#include "sigmoid.h"

namespace dnnlowp {

using namespace std;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
template <typename T>
Sigmoid<T>::Sigmoid(double max_abs_err) : tanh_(max_abs_err) {
  float x_sq = tanh_.GetSaturationRegionBegin();

  in_qparams_.scale = 2 * x_sq / ((1 << (num_in_bits_ - 1)) - 1);
  in_qparams_.zero_point = 1 << (num_in_bits_ - 1);
  in_qparams_.precision = num_in_bits_;
  // -2 x_sq is mapped to -127, 0 is mapped to 0, 2 x_sq is mapped to 127

  out_qparams_.scale = 0.5 / ((1 << (num_out_bits_ - 1)) - 1);
  out_qparams_.zero_point = 0;
  out_qparams_.precision = num_out_bits_;
  // 0 is mapped to 0, 1/2 is mapped to 127, 1 is mapped to 254
}

template <typename T>
T Sigmoid<T>::Compute(T x) const {
  T temp = tanh_.Compute(x);
  assert(temp >= 1);
  assert(temp < (1 << num_out_bits_));
  return temp - 1;
}

template class Sigmoid<uint8_t>;
template class Sigmoid<uint16_t>;
template class Sigmoid<int32_t>;

} // namespace dnnlowp
