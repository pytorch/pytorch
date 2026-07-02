#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Float8_e8m0fnu.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <c10/util/overflows.h>

#include <limits>
#include <type_traits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

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

template <typename T>
using scalar_value_type_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
constexpr bool is_saturating_float_to_signed_int_source_v =
    std::is_floating_point_v<scalar_value_type_t<T>> ||
    std::is_same_v<scalar_value_type_t<T>, c10::Half> ||
    std::is_same_v<scalar_value_type_t<T>, c10::BFloat16>;

template <typename src_t>
C10_HOST_DEVICE static inline constexpr auto signed_int_compare_value(
    src_t src) {
  if constexpr (std::is_floating_point_v<scalar_value_type_t<src_t>>) {
    return src;
  } else {
    return static_cast<float>(src);
  }
}

// Float -> integer is UB on out-of-range/NaN values. For standard floating
// sources and CPU Half/BFloat16 sources to signed integer destinations on host,
// out-of-range values saturate and NaN maps to 0 before casting below.
// Device-compiled paths, unsigned destinations, and other reduced floating
// source types keep the platform-defined result for compatibility.
// Suppress UBSan only here, so the dispatching template below stays
// UBSan-clean.
template <typename dest_t, typename src_t>
C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline constexpr dest_t
unchecked_cast_to_int(src_t src) {
  return static_cast<dest_t>(src);
}

template <typename dest_t, typename src_t>
C10_HOST_DEVICE static inline constexpr dest_t saturated_cast_to_signed_int(
    src_t src) {
  static_assert(is_saturating_float_to_signed_int_source_v<src_t>);
  static_assert(std::is_integral_v<dest_t>);
  static_assert(std::is_signed_v<dest_t>);
  const auto src_for_compare = signed_int_compare_value(src);
  using compare_t = decltype(src_for_compare);
  constexpr auto min =
      static_cast<compare_t>(std::numeric_limits<dest_t>::lowest());
  constexpr auto max =
      static_cast<compare_t>(std::numeric_limits<dest_t>::max());
  if (src_for_compare != src_for_compare) {
    return static_cast<dest_t>(0);
  }
  if (src_for_compare < min) {
    return std::numeric_limits<dest_t>::lowest();
  }
  if (src_for_compare >= max) {
    return std::numeric_limits<dest_t>::max();
  }
  return unchecked_cast_to_int<dest_t>(src_for_compare);
}

template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  C10_HOST_DEVICE static inline dest_t apply(src_t src) {
    constexpr bool real = needs_real<dest_t, src_t>::value;
    auto r = maybe_real<real, src_t>::apply(src);
    if constexpr (
        std::is_integral_v<dest_t> && std::is_signed_v<dest_t> &&
        is_saturating_float_to_signed_int_source_v<decltype(r)>) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || \
    defined(__HIP_ARCH__)
      return unchecked_cast_to_int<dest_t>(r);
#else
      return saturated_cast_to_signed_int<dest_t>(r);
#endif
    } else if constexpr (
        std::is_integral_v<dest_t> && !std::is_integral_v<decltype(r)>) {
      return unchecked_cast_to_int<dest_t>(r);
    } else {
      return static_cast<dest_t>(r);
    }
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

// Route float -> uint8 via int64 to get consistent results across CPU/GPU
// compilers; float -> unsigned directly is UB and platform-divergent.
template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
  C10_HOST_DEVICE static inline uint8_t apply(src_t src) {
    constexpr bool real = needs_real<uint8_t, src_t>::value;
    auto r = maybe_real<real, src_t>::apply(src);
    if constexpr (std::is_integral_v<decltype(r)>) {
      return static_cast<uint8_t>(static_cast<int64_t>(r));
    } else {
      return static_cast<uint8_t>(unchecked_cast_to_int<int64_t>(r));
    }
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

// TODO(#146647): Can we make all these template specialization happen
// based off our apply macros?
template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::Float8_e8m0fnu> {
  C10_HOST_DEVICE static inline c10::complex<c10::Half> apply(
      c10::Float8_e8m0fnu src) {
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

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::BFloat16>,
    c10::Float8_e5m2> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::Float8_e5m2 src) {
    return static_cast<c10::complex<c10::BFloat16>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::BFloat16>,
    c10::Float8_e5m2fnuz> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::Float8_e5m2fnuz src) {
    return static_cast<c10::complex<c10::BFloat16>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::BFloat16>,
    c10::Float8_e4m3fn> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::Float8_e4m3fn src) {
    return static_cast<c10::complex<c10::BFloat16>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::BFloat16>,
    c10::Float8_e4m3fnuz> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::Float8_e4m3fnuz src) {
    return static_cast<c10::complex<c10::BFloat16>>(c10::complex<float>{src});
  }
};

// TODO(#146647): Can we make all these template specialization happen
// based off our apply macros?
template <>
struct static_cast_with_inter_type<
    c10::complex<c10::BFloat16>,
    c10::Float8_e8m0fnu> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::Float8_e8m0fnu src) {
    return static_cast<c10::complex<c10::BFloat16>>(c10::complex<float>{src});
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::BFloat16>, c10::Half> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::Half src) {
    return static_cast<c10::complex<c10::BFloat16>>(
        static_cast<c10::complex<float>>(src));
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::BFloat16>,
    c10::complex<double>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::complex<double> src) {
    return static_cast<c10::complex<c10::BFloat16>>(
        static_cast<c10::complex<float>>(src));
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::BFloat16>,
    c10::complex<c10::Half>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::complex<c10::Half> src) {
    return static_cast<c10::complex<c10::BFloat16>>(
        static_cast<c10::complex<float>>(src));
  }
};

template <>
struct static_cast_with_inter_type<
    c10::complex<c10::Half>,
    c10::complex<c10::BFloat16>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::Half>
  apply(c10::complex<c10::BFloat16> src) {
    return static_cast<c10::complex<c10::Half>>(
        static_cast<c10::complex<float>>(src));
  }
};

template <>
struct static_cast_with_inter_type<c10::Half, c10::complex<c10::BFloat16>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::Half apply(
      c10::complex<c10::BFloat16> src) {
    return static_cast<c10::Half>(static_cast<float>(src.real()));
  }
};

template <>
struct static_cast_with_inter_type<c10::BFloat16, c10::complex<c10::Half>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::BFloat16 apply(
      c10::complex<c10::Half> src) {
    return static_cast<c10::BFloat16>(static_cast<float>(src.real()));
  }
};

template <>
struct static_cast_with_inter_type<c10::BFloat16, c10::complex<c10::BFloat16>> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::BFloat16 apply(
      c10::complex<c10::BFloat16> src) {
    return src.real();
  }
};

template <>
struct static_cast_with_inter_type<c10::complex<c10::BFloat16>, c10::BFloat16> {
  C10_HOST_DEVICE __ubsan_ignore_undefined__ static inline c10::complex<
      c10::BFloat16>
  apply(c10::BFloat16 src) {
    return c10::complex<c10::BFloat16>{src, 0};
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

C10_CLANG_DIAGNOSTIC_POP()

// Trigger tests for D25440771. TODO: Remove this line any time you want.
