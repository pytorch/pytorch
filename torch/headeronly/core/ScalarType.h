#pragma once

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float4_e2m1fn_x2.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Float8_e8m0fnu.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/bits.h>
#include <torch/headeronly/util/complex.h>
#include <torch/headeronly/util/qint32.h>
#include <torch/headeronly/util/qint8.h>
#include <torch/headeronly/util/quint2x4.h>
#include <torch/headeronly/util/quint4x2.h>
#include <torch/headeronly/util/quint8.h>

#include <cstdint>

namespace c10 {

// dummy struct for uint1 to uint7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
template <unsigned int N>
struct dummy_uint1_7_t {};

// dummy struct for int1 to int7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
template <unsigned int N>
struct dummy_int1_7_t {};

// See [dtype Macros note] in c10/core/ScalarType.h regarding macros

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(at::Half, Half) /* 5 */                              \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \
  _(c10::complex<float>, ComplexFloat) /* 9 */           \
  _(c10::complex<double>, ComplexDouble) /* 10 */        \
  _(bool, Bool) /* 11 */                                 \
  _(c10::qint8, QInt8) /* 12 */                          \
  _(c10::quint8, QUInt8) /* 13 */                        \
  _(c10::qint32, QInt32) /* 14 */                        \
  _(at::BFloat16, BFloat16) /* 15 */                     \
  _(c10::quint4x2, QUInt4x2) /* 16 */                    \
  _(c10::quint2x4, QUInt2x4) /* 17 */                    \
  _(c10::bits1x8, Bits1x8) /* 18 */                      \
  _(c10::bits2x4, Bits2x4) /* 19 */                      \
  _(c10::bits4x2, Bits4x2) /* 20 */                      \
  _(c10::bits8, Bits8) /* 21 */                          \
  _(c10::bits16, Bits16) /* 22 */                        \
  _(c10::Float8_e5m2, Float8_e5m2) /* 23 */              \
  _(c10::Float8_e4m3fn, Float8_e4m3fn) /* 24 */          \
  _(c10::Float8_e5m2fnuz, Float8_e5m2fnuz) /* 25 */      \
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz) /* 26 */      \
  _(uint16_t, UInt16) /* 27 */                           \
  _(uint32_t, UInt32) /* 28 */                           \
  _(uint64_t, UInt64) /* 29 */                           \
  _(dummy_uint1_7_t<1>, UInt1) /* 30 */                  \
  _(dummy_uint1_7_t<2>, UInt2) /* 31 */                  \
  _(dummy_uint1_7_t<3>, UInt3) /* 32 */                  \
  _(dummy_uint1_7_t<4>, UInt4) /* 33 */                  \
  _(dummy_uint1_7_t<5>, UInt5) /* 34 */                  \
  _(dummy_uint1_7_t<6>, UInt6) /* 35 */                  \
  _(dummy_uint1_7_t<7>, UInt7) /* 36 */                  \
  _(dummy_int1_7_t<1>, Int1) /* 37 */                    \
  _(dummy_int1_7_t<2>, Int2) /* 38 */                    \
  _(dummy_int1_7_t<3>, Int3) /* 39 */                    \
  _(dummy_int1_7_t<4>, Int4) /* 40 */                    \
  _(dummy_int1_7_t<5>, Int5) /* 41 */                    \
  _(dummy_int1_7_t<6>, Int6) /* 42 */                    \
  _(dummy_int1_7_t<7>, Int7) /* 43 */                    \
  _(c10::Float8_e8m0fnu, Float8_e8m0fnu) /* 44 */        \
  _(c10::Float4_e2m1fn_x2, Float4_e2m1fn_x2) /* 45 */

enum class ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};

constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

} // namespace c10

namespace torch::headeronly {
using c10::dummy_int1_7_t;
using c10::dummy_uint1_7_t;
using c10::NumScalarTypes;
using c10::ScalarType;
} // namespace torch::headeronly
