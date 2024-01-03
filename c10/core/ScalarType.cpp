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
  constexpr auto b8 = ScalarType::Float8_e5m2;
  constexpr auto h8 = ScalarType::Float8_e4m3fn;
  constexpr auto a8 = ScalarType::Float8_e5m2fnuz;
  constexpr auto d8 = ScalarType::Float8_e4m3fnuz;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }

  // For QInt types, we only allow exact match
  if (isQIntType(a) && a == b) {
    return a;
  }

  if (isQIntType(a) || isQIntType(b)) {
    TORCH_CHECK(
        false,
        "promoteTypes with quantized numbers is not handled yet; figure out what the correct rules should be, offending types: ",
        toString(a),
        " ",
        toString(b));
  }

  if (isBitsType(a) && a == b) {
    return a;
  } else if (isBitsType(a) || isBitsType(b)) {
    return ScalarType::Undefined;
  }

  // Bits and Quantized are 7 dtypes already handled and not included
  // in the promotion table below. Therefore every dtype above them
  // needs to be shifted down to account for that.
  static constexpr int shift_distance =
      static_cast<int>(ScalarType::Float8_e5m2) -
      static_cast<int>(ScalarType::QUInt4x2);
  static constexpr int shift_threshold = static_cast<int>(ScalarType::Bits16);

  if (static_cast<int>(a) > shift_threshold) {
    a = static_cast<ScalarType>(static_cast<int>(a) - shift_distance);
  }

  if (static_cast<int>(b) > shift_threshold) {
    b = static_cast<ScalarType>(static_cast<int>(b) - shift_distance);
  }

  // For the same reason the size of promotion table is decreased by 7
  // comparing to the number of all dtypes.
  static constexpr int NUM_PROMOTE_TYPES =
      static_cast<int>(ScalarType::NumOptions) - shift_distance;

  // this matrix has to be consistent with
  // AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS undefined is used where we
  // are not sure about the correct value for type promotion.
  // clang-format off
  static constexpr ScalarType _promoteTypesLookup[
      NUM_PROMOTE_TYPES][NUM_PROMOTE_TYPES] = {
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  q1  q2  q3  bf  b8  h8  a8  d8*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1, ud, ud, ud, bf, b8, h8, a8, d8},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1, ud, ud, ud, bf, b8, h8, a8, d8},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2, ud, ud, ud, bf, b8, h8, a8, d8},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4, ud, ud, ud, bf, b8, h8, a8, d8},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8, ud, ud, ud, bf, b8, h8, a8, d8},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2, ud, ud, ud, f4, f4, f4, f4, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4, ud, ud, ud, f4, f4, f4, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8, ud, ud, ud, f8, f8, f8, f8, f8},
      /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2, ud, ud, ud, c4, c4, c4, c4, c4},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, ud, ud, ud, c4, c4, c4, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, ud, ud, ud, c8, c8, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, ud, ud, ud, bf, b8, h8, a8, d8},
      /* q1 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* q2 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* q3 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c4, c8, bf, ud, ud, ud, bf, bf, bf, bf, bf},
      /* b8 */ {b8, b8, b8, b8, b8, f4, f4, f8, c4, c4, c8, b8, ud, ud, ud, bf, b8, ud, a8, ud},
      /* h8 */ {h8, h8, h8, h8, h8, f4, f4, f8, c4, c4, c8, h8, ud, ud, ud, bf, ud, h8, ud, d8},
      /* a8 */ {a8, a8, a8, a8, a8, f4, f4, f8, c4, c4, c8, a8, ud, ud, ud, bf, a8, ud, a8, ud},
      /* d8 */ {d8, d8, d8, d8, d8, f4, f4, f8, c4, c4, c8, d8, ud, ud, ud, bf, ud, d8, ud, d8},
  };
  // clang-format on
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

} // namespace c10
