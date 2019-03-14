#include <cmath>

namespace at { namespace native {

template <typename scalar_t>
bool _isnan(scalar_t val) {
  return false;
}

template <>
bool _isnan(float val) {
  return std::isnan(val);
}

template <>
bool _isnan(double val) {
  return std::isnan(val);
}


}} // namespace at::native
