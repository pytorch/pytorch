#if !defined(C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "c10/util/complex_math.h is not meant to be individually included. Include c10/util/complex.h instead."
#endif

#include <c10/macros/Macros.h>

#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))
namespace c10_complex_math {
namespace _detail {
C10_API c10::complex<float> sqrt(const c10::complex<float>& in);
C10_API c10::complex<double> sqrt(const c10::complex<double>& in);
C10_API c10::complex<float> acos(const c10::complex<float>& in);
C10_API c10::complex<double> acos(const c10::complex<double>& in);
} // namespace _detail
} // namespace c10_complex_math
#endif

#include <torch/headeronly/util/complex_math.h> // IWYU pragma: keep
