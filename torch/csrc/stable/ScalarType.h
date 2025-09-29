#pragma once

/*
  Keep the set of dtypes that are supported by torch stable ABI in
  sync in the following files:
    this file
    torch/csrc/stable/stableivalue_conversions.h
    test/cpp_extensions/libtorch_agnostic_extension/test/test_libtorch_agnostic.py

 */

#include <torch/headeronly/core/ScalarType.h>

#define STABLE_FORALL_SUPPORTED_SCALAR_TYPES(_) \
  _(uint8_t, Byte)                              \
  _(int8_t, Char)                               \
  _(int16_t, Short)                             \
  _(int32_t, Int)                               \
  _(int64_t, Long)                              \
  _(c10::Half, Half)                            \
  _(float, Float)                               \
  _(double, Double)                             \
  _(c10::complex<c10::Half>, ComplexHalf)       \
  _(c10::complex<float>, ComplexFloat)          \
  _(c10::complex<double>, ComplexDouble)        \
  _(bool, Bool)                                 \
  _(c10::BFloat16, BFloat16)                    \
  _(c10::Float8_e5m2, Float8_e5m2)              \
  _(c10::Float8_e4m3fn, Float8_e4m3fn)          \
  _(c10::Float8_e5m2fnuz, Float8_e5m2fnuz)      \
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz)      \
  _(uint16_t, UInt16)                           \
  _(uint32_t, UInt32)                           \
  _(uint64_t, UInt64)

namespace torch::stable {
using torch::headeronly::ScalarType;

inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name) \
  case ScalarType::name:     \
    return #name;
  switch (t) {
    STABLE_FORALL_SUPPORTED_SCALAR_TYPES(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}
} // namespace torch::stable
