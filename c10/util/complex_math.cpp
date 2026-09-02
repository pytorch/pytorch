#include <c10/util/complex.h>

// Note [ Complex Square root in libc++]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Export c10_complex_math::_detail::sqrt/acos as C10_API for libtorch ABI.
// The implementations live in torch::headeronly::complex_math_detail; these
// wrappers delegate to them.

#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))

namespace c10_complex_math {
namespace _detail {
c10::complex<float> sqrt(const c10::complex<float>& in) {
  return ::torch::headeronly::complex_math_detail::sqrt(in);
}

c10::complex<double> sqrt(const c10::complex<double>& in) {
  return ::torch::headeronly::complex_math_detail::sqrt(in);
}

c10::complex<float> acos(const c10::complex<float>& in) {
  return ::torch::headeronly::complex_math_detail::acos(in);
}

c10::complex<double> acos(const c10::complex<double>& in) {
  return ::torch::headeronly::complex_math_detail::acos(in);
}

} // namespace _detail
} // namespace c10_complex_math
#endif
