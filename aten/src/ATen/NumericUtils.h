#include <cmath>

namespace at {

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.
template <typename scalar_t>
inline bool _isnan(scalar_t val) {
  if (std::is_floating_point<scalar_t>::value) {
    return std::isnan(val);
  } else {
    return false;
  }
}

} // namespace at
