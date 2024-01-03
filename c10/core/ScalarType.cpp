#include <c10/core/ScalarType.h>
#include <array>

namespace c10 {

namespace {

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

constexpr int64_t NUM_PROMOTE_TYPES = 20;

constexpr std::array<ScalarType, NUM_PROMOTE_TYPES> index2dtype =
    {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf, b8, h8, a8, d8};

constexpr std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)>
calculate_dtype2index() {
  std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)> inverse = {};
  for (int64_t i = 0; i < static_cast<int64_t>(ScalarType::NumOptions); i++) {
    inverse[i] = -1;
  }
  for (int64_t i = 0; i < static_cast<int64_t>(index2dtype.size()); i++) {
    inverse[static_cast<int64_t>(index2dtype[i])] = i;
  }
  return inverse;
}

constexpr auto dtype2index = calculate_dtype2index();

} // anonymous namespace

ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
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

  auto ix_a = dtype2index[static_cast<int64_t>(a)];
  TORCH_INTERNAL_ASSERT(ix_a != -1);
  auto ix_b = dtype2index[static_cast<int64_t>(b)];
  TORCH_INTERNAL_ASSERT(ix_b != -1);

  // This table axes must be consistent with index2dtype
  // clang-format off
  static constexpr ScalarType _promoteTypesLookup[
      NUM_PROMOTE_TYPES][NUM_PROMOTE_TYPES] = {
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  bf  b8  h8  a8  d8*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1, bf, b8, h8, a8, d8},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1, bf, b8, h8, a8, d8},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2, bf, b8, h8, a8, d8},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4, bf, b8, h8, a8, d8},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8, bf, b8, h8, a8, d8},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2, f4, f4, f4, f4, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4, f4, f4, f4, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8, f8, f8, f8, f8, f8},
      /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2, c4, c4, c4, c4, c4},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, c4, c4, c4, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf, b8, h8, a8, d8},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c4, c8, bf, bf, bf, bf, bf, bf},
      /* b8 */ {b8, b8, b8, b8, b8, f4, f4, f8, c4, c4, c8, b8, bf, b8, ud, a8, ud},
      /* h8 */ {h8, h8, h8, h8, h8, f4, f4, f8, c4, c4, c8, h8, bf, ud, h8, ud, d8},
      /* a8 */ {a8, a8, a8, a8, a8, f4, f4, f8, c4, c4, c8, a8, bf, a8, ud, a8, ud},
      /* d8 */ {d8, d8, d8, d8, d8, f4, f4, f8, c4, c4, c8, d8, bf, ud, d8, ud, d8},
  };
  // clang-format on
  return _promoteTypesLookup[ix_a][ix_b];
}

} // namespace c10
