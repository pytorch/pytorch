#pragma once

#include <ATen/ATen.h>

#include <tuple>

// Promotion and casting rules for mixed-type operations.
//
// Type promotion works similar to NumPy type promotion. Operands with a higher
// category (i.e. integral, floating or complex) have precedence over other
// operands. Within the same category, only the operands with the highest
// available priority participate in the final type. The priority order is:
//  - Operands that are explicit ScalarTypes
//  - Tensor operands with >=1 dimensions
//  - Tensor operands with 0 dimensions
//  - Scalar or bool operands (same priority)
//
// Some operations will be needed to be casted into a user specified type.
// In this case, only typecasts that do not demote the type category are allowed
// ("same_kind" in NumPy). Thus, an operand can be downcasted into another type
// within the same category but never to a lower category.
//
// The function "resultType" applies the promotion rules and the function
// "castOperands" applies the casting rules laid out above.
//
// If you are converting a function to support mixed types, also change
// the backward function to go back to the original type.
//
// e.g.
// BEFORE:
//   - name: op(Tensor self, Scalar alpha)
//     self: op_backward(grad, Scalar alpha)
// AFTER:
//   - name: op(Tensor self, Scalar alpha)
//     self: op_backward(grad, Scalar alpha).to(self.scalar_type())

namespace at {

struct Type;

namespace detail {

template<typename T>
static inline T castToType(ScalarType type, T arg) = delete;

template<>
Tensor castToType(ScalarType type, Tensor arg) {
  return arg.to(type);
}

template<>
Scalar castToType(ScalarType type, Scalar arg) {
  bool is_i = false;
  bool is_d = false;
  bool is_z = false;
  #define DEFINE_CASE(_1, name, tag) \
      case ScalarType::name:         \
      is_##tag = true;               \
      break;
    switch(type) {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
      default:
        break;
    }
  #undef DEFINE_CASE
  if (is_i) {
    return Scalar(arg.toLong());
  } else if (is_d) {
    return Scalar(arg.toDouble());
  } else if (is_z) {
    return Scalar(arg.toComplexDouble());
  } else {
    AT_ERROR("Unknown scalar type");
  }
}

} // namespace detail

// Applies type promotion to a list of operands according to PyTorch promotion
// rules.
//
// The exact algorithm is to successively apply "promoteTypes" to the operands
// with the highest category and the highest priority in that category.
//
// The order for categories is: complex, floating, integral
// The order for priorities is: ScalarType, Tensor, Scalar/bool
//
// Examples:
//  - resultType(float_tensor, int_tensor) -> float
//      because promotTypes(Int, Float) is float
//  - resultType(float_scalar, int_tensor) -> float
//      because the scalar is the only operands with the highest category
//  - resultType(long_scalar,  int_tensor) -> int
//      because tensor has higher priority than scalar
CAFFE2_API ScalarType resultType(ArrayRef<ScalarTypeSource> inputs);

// Casts mixed type operands to either the target type or the result of
// "resultType" if no target type is provided.
//
// When a "dtype" is provided, each operand will be checked against
// "canCastSameKind" and an exception will be thrown if the cast would result
// in a category demotion (e.g. floating to integral). This casting checks
// prevents coercing operands to a lower precision. Operands can still lose
// range and, in the case of integral types, overflow when downcasted.
//
// When a dtype is not provided, the final type of operands will be determined
// with the "resultType" function. Since "promoteTypes" always results in the
// highest category of the operands, this would never lead to a casting error.
//
// This interface is suited to pass an optional dtype parameter from a native
// function implementation.
template<typename... T>
static inline std::tuple<T...> castOperands(c10::optional<ScalarType> dtype, T... args) {
  SmallVector<ScalarTypeSource, 4> type_sources = {std::forward<T>(args)...};
  if (!dtype) {
    dtype = resultType(type_sources);
  }
  AT_CHECK(*dtype != ScalarType::Undefined);
  for (const ScalarTypeSource& type_source : type_sources) {
    // Input type cannot be downcasted into dtype.
    AT_CHECK(
        canCastSameKind(type_source.scalarType(), *dtype),
        "Cannot coerce input of type ",
        type_source.scalarType(),
        " into ",
        *dtype);
  }
  return std::make_tuple(detail::castToType<T>(*dtype, std::forward<T>(args))...);
}

// Alternative to "castOperands", for cases when there is no target dtype to
// use. Returns the operands casted into the output of applying "resultType"
// on them.
template<typename... T>
static inline std::tuple<T...> castOperandsToResultType(T... args) {
  return castOperands<T...>(c10::nullopt, std::forward<T>(args)...);
}

}  // namespace at
