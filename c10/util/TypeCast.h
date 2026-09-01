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
#include <c10/util/safe_conv.h>

#include <torch/headeronly/util/TypeCast.h>

#include <type_traits>

namespace c10 {

// Thin wrappers inherit from torch::headeronly so c10 remains the customization
// point and downstream explicit specializations in namespace c10 stay valid.
template <typename dest_t, typename src_t>
struct needs_real : torch::headeronly::needs_real<dest_t, src_t> {};

template <bool B, typename src_t>
struct maybe_real : torch::headeronly::maybe_real<B, src_t> {};

template <bool B, typename src_t>
struct maybe_bool : torch::headeronly::maybe_bool<B, src_t> {};

template <typename dest_t, typename src_t>
struct static_cast_with_inter_type
    : torch::headeronly::static_cast_with_inter_type<dest_t, src_t> {};

template <typename To, typename From>
C10_HOST_DEVICE To convert(From f) {
  return static_cast_with_inter_type<To, From>::apply(f);
}

using torch::headeronly::report_overflow;
using torch::headeronly::unchecked_cast_to_int;

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  if (!std::is_same_v<To, bool> && overflows<To, From>(f)) {
    report_overflow(name);
  }
  return convert<To, From>(f);
}

// Range-checked conversion that PERMITS signed->unsigned two's-complement
// wraparound (via overflows() with its default strict_unsigned=false). Retained
// only to preserve the historical behavior of the few call sites that relied on
// the wrap. DO NOT use in new code: use c10::safe_conv (strict integer
// narrowing, c10/util/safe_conv.h) or checked_convert (general, above).
template <typename To, typename From>
To unsafe_wrapping_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  if (!std::is_same_v<To, bool> && overflows<To, From>(f)) {
    report_overflow(name);
  }
  return convert<To, From>(f);
}

} // namespace c10
