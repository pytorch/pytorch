#include <c10/core/ScalarType.h>
#include <c10/util/Array.h>
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
constexpr auto ud = ScalarType::Undefined;

constexpr auto index2dtype = array_of<
    c10::ScalarType>(u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf);

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

  if (isBarebonesUnsignedType(a) || isBarebonesUnsignedType(b)) {
    // There are two problems with promotion here:
    //
    // - Our promotion rule for uint8 is inconsistent with Numpy; Numpy
    //   promotes to uint64, but since we never had uint64 for the longest
    //   time, we promote to int64.  Changing this is BC-breaking
    //
    // - We must not promote uint64 to int64 because this will overflow.
    //
    // It'll be a bit of work to fix it, so we're punting on it for now.
    // However, float promotion is fine, so we handle that.
    if (isFloatingType(a)) {
      return a;
    }
    if (isFloatingType(b)) {
      return b;
    }
    TORCH_CHECK(
        false,
        "Promotion for uint16, uint32, uint64 types is not supported, attempted to promote ",
        toString(a),
        " and ",
        toString(b));
  }

  auto ix_a = dtype2index[static_cast<int64_t>(a)];
  TORCH_INTERNAL_ASSERT(ix_a != -1);
  auto ix_b = dtype2index[static_cast<int64_t>(b)];
  TORCH_INTERNAL_ASSERT(ix_b != -1);

  // This table axes must be consistent with index2dtype
  // clang-format off
  static constexpr std::
  array<std::array<ScalarType, index2dtype.size()>, index2dtype.size()>
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
  return _promoteTypesLookup[ix_a][ix_b];
}

std::pair<std::string, std::string> getDtypeNames(c10::ScalarType scalarType) {
  switch (scalarType) {
    case c10::ScalarType::UInt1:
      return std::make_pair("uint1", "bit");
    case c10::ScalarType::UInt2:
      return std::make_pair("uint2", "");
    case c10::ScalarType::UInt3:
      return std::make_pair("uint3", "");
    case c10::ScalarType::UInt4:
      return std::make_pair("uint4", "");
    case c10::ScalarType::UInt5:
      return std::make_pair("uint5", "");
    case c10::ScalarType::UInt6:
      return std::make_pair("uint6", "");
    case c10::ScalarType::UInt7:
      return std::make_pair("uint7", "");
    case c10::ScalarType::Byte:
      // no "byte" because byte is signed in numpy and we overload
      // byte to mean bool often
      return std::make_pair("uint8", "");
    case c10::ScalarType::UInt16:
      return std::make_pair("uint16", "");
    case c10::ScalarType::UInt32:
      return std::make_pair("uint32", "");
    case c10::ScalarType::UInt64:
      return std::make_pair("uint64", "");
    case c10::ScalarType::Char:
      // no "char" because it is not consistently signed or unsigned; we want
      // to move to int8
      return std::make_pair("int8", "");
    case c10::ScalarType::Double:
      return std::make_pair("float64", "double");
    case c10::ScalarType::Float:
      return std::make_pair("float32", "float");
    case c10::ScalarType::Int:
      return std::make_pair("int32", "int");
    case c10::ScalarType::Long:
      return std::make_pair("int64", "long");
    case c10::ScalarType::Short:
      return std::make_pair("int16", "short");
    case c10::ScalarType::Half:
      return std::make_pair("float16", "half");
    case c10::ScalarType::ComplexHalf:
      return std::make_pair("complex32", "chalf");
    case c10::ScalarType::ComplexFloat:
      return std::make_pair("complex64", "cfloat");
    case c10::ScalarType::ComplexDouble:
      return std::make_pair("complex128", "cdouble");
    case c10::ScalarType::Bool:
      return std::make_pair("bool", "");
    case c10::ScalarType::QInt8:
      return std::make_pair("qint8", "");
    case c10::ScalarType::QUInt8:
      return std::make_pair("quint8", "");
    case c10::ScalarType::QInt32:
      return std::make_pair("qint32", "");
    case c10::ScalarType::BFloat16:
      return std::make_pair("bfloat16", "");
    case c10::ScalarType::QUInt4x2:
      return std::make_pair("quint4x2", "");
    case c10::ScalarType::QUInt2x4:
      return std::make_pair("quint2x4", "");
    case c10::ScalarType::Bits1x8:
      return std::make_pair("bits1x8", "");
    case c10::ScalarType::Bits2x4:
      return std::make_pair("bits2x4", "");
    case c10::ScalarType::Bits4x2:
      return std::make_pair("bits4x2", "");
    case c10::ScalarType::Bits8:
      return std::make_pair("bits8", "");
    case c10::ScalarType::Bits16:
      return std::make_pair("bits16", "");
    case c10::ScalarType::Float8_e5m2:
      return std::make_pair("float8_e5m2", "");
    case c10::ScalarType::Float8_e4m3fn:
      return std::make_pair("float8_e4m3fn", "");
    case c10::ScalarType::Float8_e5m2fnuz:
      return std::make_pair("float8_e5m2fnuz", "");
    case c10::ScalarType::Float8_e4m3fnuz:
      return std::make_pair("float8_e4m3fnuz", "");
    default:
      throw std::runtime_error("Unimplemented scalar type");
  }
}

const std::unordered_map<std::string, ScalarType>& getStringToDtypeMap() {
  static std::unordered_map<std::string, ScalarType> result;
  if (!result.empty()) {
    return result;
  }

#define DEFINE_SCALAR_TYPE(_1, n) c10::ScalarType::n,

  auto all_scalar_types = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

#undef DEFINE_SCALAR_TYPE

  for (auto scalar_type : all_scalar_types) {
    auto names = getDtypeNames(scalar_type);
    result[std::get<0>(names)] = scalar_type;
    if (!std::get<1>(names).empty()) {
      result[std::get<1>(names)] = scalar_type;
    }
  }
  return result;
}

} // namespace c10
