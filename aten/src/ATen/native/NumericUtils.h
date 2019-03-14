#include <cmath>

namespace at { namespace native {

template <typename scalar_t>
inline bool _isnan(scalar_t val) {
  return false;
}

template <>
inline bool _isnan(float val) {
  return std::isnan(val);
}

template <>
inline bool _isnan(double val) {
  return std::isnan(val);
}


}} // namespace at::native
