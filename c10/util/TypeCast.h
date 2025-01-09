#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <c10/util/overflows.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace c10 {

template <typename dest_t, typename src_t>
struct needs_real {
  constexpr static bool value =
      (is_complex<src_t>::value && !is_complex<dest_t>::value);
};

template <bool, typename src_t>
struct maybe_real {
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template <typename src_t>
struct maybe_real<true, src_t> {
  C10_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    return src.real();
  }
};

template <bool, typename src_t>
struct maybe_bool {
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template <typename src_t>
struct maybe_bool<true, src_t> {
  C10_HOST_DEVICE static inline decltype(auto) apply(src_t src) {
    // Don't use bool operator so as to also compile for ComplexHalf.
    return src.real() || src.imag();
  }
};

// PyTorch's type conversions can cause a variety of undefined behavior,
// including float to integral overflow and signed to unsigned integer overflow.
// Some of this undefined behavior is addressed below.
template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  C10_HOST_DEVICE static inline dest_t apply(src_t src) {
    constexpr bool real = needs_real<dest_t, src_t>::value;
    auto r = maybe_real<real, src_t>::apply(src);
    // Note: Converting from negative float values to unsigned integer types is
    // undefined behavior in C++, and current CPU and GPU compilers exhibit
    // divergent behavior. A consistent behavior is forced by clip it into the
    // result type. So this cast improves the consistency of type conversions
    // across compilers.
    if constexpr (::std::is_integral_v<dest_t>) {
      if constexpr (::std::is_floating_point_v<decltype(r)>) {
        constexpr auto max_int_value = std::numeric_limits<dest_t>::max();
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
        if (static_cast<double>(r) >= __ll2double_rz(max_int_value)) {
#else
        if (C10_UNLIKELY(
                static_cast<double>(r) >= ::std::rint(max_int_value))) {
#endif
          return max_int_value;
        }
        if constexpr (::std::is_unsigned_v<dest_t>) {
          if (r < 0) {
            return 0;
          }
        } else {
          constexpr auto min_int_value =
              std::numeric_limits<std::make_signed_t<dest_t>>::min();
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
          if (static_cast<double>(r) <= __ll2double_rz(min_int_value)) {
#else
          if (C10_UNLIKELY(
                  static_cast<double>(r) <= ::std::rint(min_int_value))) {
#endif
            return min_int_value;
          }
        }
      }
    }
    return static_cast<dest_t>(r);
  }
};

// Partial template specialization for casting to bool.
// Need to handle complex types separately, as we don't
// simply want to cast the real part to bool.
template <typename src_t>
struct static_cast_with_inter_type<bool, src_t> {
  C10_HOST_DEVICE static inline bool apply(src_t src) {
    constexpr bool complex = needs_real<bool, src_t>::value;
    return static_cast<bool>(maybe_bool<complex, src_t>::apply(src));
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::BFloat16> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(
      c10::BFloat16 src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::Float8_e5m2> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(
      c10::Float8_e5m2 src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e5m2fnuz> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(
      c10::Float8_e5m2fnuz src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e4m3fn> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(
      c10::Float8_e4m3fn src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e4m3fnuz> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(
      c10::Float8_e4m3fnuz src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::Half>, c10::Half> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(c10::Half src) {
    return static_cast<c10::complex<c10::Half>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::complex<double>> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(
      c10::complex<double> src) {
    return static_cast<c10::complex<c10::Half>>(
        static_cast<c10::complex<float>>(src));
  }
};

template <typename To, typename From>
C10_HOST_DEVICE To convert(From f) {
  return static_cast_with_inter_type<To, From>::apply(f);
}

// Define separately to avoid being inlined and prevent code-size bloat
[[noreturn]] C10_API void report_overflow(const char* name);

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  if (!std::is_same_v<To, bool> && overflows<To, From>(f)) {
    report_overflow(name);
  }
  return convert<To, From>(f);
}

} // namespace c10

// Trigger tests for D25440771. TODO: Remove this line any time you want.
