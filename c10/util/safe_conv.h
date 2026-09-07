#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>

#include <type_traits>

// c10::safe_conv is the deliberately lightweight, INTEGER-ONLY, strict
// narrowing conversion. This header intentionally avoids <c10/util/TypeCast.h>
// (and its Half/BFloat16/Float8/complex machinery) so it stays cheap to include
// in ubiquitous low-level headers and, especially, in CUDA/HIP kernels.
//
// For general conversions -- floating-point or complex sources, or any case
// where a lossy-but-in-range cast is acceptable -- use c10::checked_convert
// (c10/util/TypeCast.h) instead. If you deliberately want signed->unsigned
// two's-complement wraparound, use c10::unsafe_wrapping_convert.

namespace c10 {

namespace detail {
// Defined out-of-line in safe_conv.cpp (part of //c10/util:base) so the throw
// is not inlined at every call site (avoids code-size bloat), and so this
// header stays lightweight.
[[noreturn]] C10_API void report_narrowing_overflow(const char* name);

#if defined(__cpp_concepts)
template <typename T>
concept SafeConvIntegral = std::is_integral_v<T> && !std::is_same_v<T, bool>;
#endif
} // namespace detail

// Strict, range-checked INTEGER narrowing conversion.
//
// Rejects ANY value not representable in To -- in particular there is NO
// signed->unsigned two's-complement wraparound (a negative source converted to
// an unsigned type is rejected, not wrapped). Use this wherever a narrowing
// truncation would be a bug.
//
// For floating-point/complex sources or general conversions, use
// c10::checked_convert (c10/util/TypeCast.h). For intentional modular wrap, use
// c10::unsafe_wrapping_convert.
//
// Usable in both host and device code: on host it throws via
// report_narrowing_overflow; in CUDA/HIP device code (which cannot throw) it
// traps via CUDA_KERNEL_ASSERT_PRINTF, printing name if one is provided.
#if defined(__cpp_concepts)
template <detail::SafeConvIntegral To, detail::SafeConvIntegral From>
#else
template <typename To, typename From>
#endif
C10_HOST_DEVICE To safe_conv(From f, const char* name = nullptr) {
#if !defined(__cpp_concepts)
  // The integer-only restriction is load-bearing for correctness, not just
  // ergonomics: the TypeSafeSignMath range check below is exact for
  // integer->integer, but its digits-based logic is silently wrong for
  // floating-point sources. Route float/complex through c10::checked_convert.
  static_assert(
      std::is_integral_v<To> && !std::is_same_v<To, bool>,
      "safe_conv requires an integral, non-bool destination type");
  static_assert(
      std::is_integral_v<From> && !std::is_same_v<From, bool>,
      "safe_conv requires an integral, non-bool source type");
#endif
  // Exact for integer->integer and device-safe (plain constexpr comparisons).
  // TODO: Use std::in_range after ExecuTorch moves to C++20.
  const bool representable =
      !c10::less_than_lowest<To>(f) && !c10::greater_than_max<To>(f);
  if (!representable) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // Device code cannot throw; trap on overflow instead, printing name if set.
    CUDA_KERNEL_ASSERT_PRINTF(
        false,
        "value cannot be safely converted without overflow: %s",
        name != nullptr ? name : "<unknown>");
#else
    detail::report_narrowing_overflow(name);
#endif
  }
  return static_cast<To>(f);
}

} // namespace c10
