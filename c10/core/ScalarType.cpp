#include <c10/core/ScalarType.h>

namespace c10 {

ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
  constexpr auto u1 = ScalarType::Byte;
  constexpr auto i1 = ScalarType::Char;
  constexpr auto i2 = ScalarType::Short;
  constexpr auto i4 = ScalarType::Int;
  constexpr auto i8 = ScalarType::Long;
  constexpr auto f2 = ScalarType::Half;
  constexpr auto f4 = ScalarType::Float;
  constexpr auto f8 = ScalarType::Double;
  constexpr auto c2 = ScalarType::ComplexHalf;
  constexpr auto c4 = ScalarType::ComplexFloat;
  constexpr auto c8 = ScalarType::ComplexDouble;
  constexpr auto b1 = ScalarType::Bool;
  constexpr auto bf = ScalarType::BFloat16;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }

  // If the two types are equal, return that type
  if (a == b) {
    return a;
  }

  // Handle identically equal types
  if (isQIntType(a) || isQIntType(b)) {
    TORCH_CHECK(
        false,
        "promoteTypes with quantized numbers is not handled yet; figure out what the correct rules should be, offending types: ",
        toString(a),
        " ",
        toString(b));
  }

  if (isBitsType(a) || isBitsType(b)) {
    return ScalarType::Undefined;
  }

  if (isFloat8Type(a) || isFloat8Type(b)) {
    TORCH_CHECK(
        false,
        "Promotion for Float8 Types is not supported, attempted to promote ",
        toString(a),
        " and ",
        toString(b));
  }

  // Bits, Quantized and Float8 are 14 dtypes already handled and not included
  // in the promotion table below.
  static constexpr int num_bits_types = static_cast<int>(ScalarType::Bits16) -
      static_cast<int>(ScalarType::Bits1x8) + 1;

  static constexpr int num_float8_types =
      static_cast<int>(ScalarType::Float8_e4m3fnuz) -
      static_cast<int>(ScalarType::Float8_e5m2) + 1;

  static constexpr int num_qint_types = static_cast<int>(ScalarType::QInt32) -
      static_cast<int>(ScalarType::QInt8) + 1;

  static constexpr int num_quint_types =
      static_cast<int>(ScalarType::QUInt2x4) -
      static_cast<int>(ScalarType::QUInt4x2) + 1;

  static constexpr int num_quantized_types = num_qint_types + num_quint_types;

  static constexpr int num_missing_types =
      num_bits_types + num_float8_types + num_quantized_types;

  // Bfloat16 is at position 15 in the ScalerType enum, There are three types
  // below bf16 not included in the table, Qint8, QUInt8, QInt32. Every other
  // type above bf16, i.e. {Bits, Quantized, Float8} are not included in the
  // table.

  // If either of the types is bf16, we need to shift the type down by the one
  // missing section in the table that is less then bf16 i.e {QInt8, QUInt8,
  // QInt32}
  a = a == bf ? static_cast<ScalarType>(static_cast<int>(a) - num_qint_types)
              : a;
  b = b == bf ? static_cast<ScalarType>(static_cast<int>(b) - num_qint_types)
              : b;

  // We decrease the promotion table by the number of missing types -> 14
  // and then subtract 1 more from the table since we don't store ud to ud
  // mapping.
  static constexpr int NUM_PROMOTE_TYPES =
      static_cast<int>(ScalarType::NumOptions) - num_missing_types - 1;

  // this matrix has to be consistent with
  // AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS undefined is used where we
  // are not sure about the correct value for type promotion.
  // clang-format off
  static constexpr std::
  array<std::array<ScalarType, NUM_PROMOTE_TYPES>, NUM_PROMOTE_TYPES>
      _promoteTypesLookup = {{
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  bf*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1, bf},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1, bf},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2, bf},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4, bf},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8, bf},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8, f8},
      /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2, c4},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c4, c8, bf, bf},
  }};
  // clang-format on
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

} // namespace c10
