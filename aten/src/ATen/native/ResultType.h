#pragma once

#include <ATen/ATen.h>

#include <tuple>

// Type determination rules for mixed-type operations.
//
// The result type is computed using the operands with the following precedence:
//
// 1) Tensors with dim 1 or higher
// 2) Tensors with dim 0 that aren't wrapped numbers (e.g. `tensor(5)`)
// 3) Tensors with dim 0 that are wrapped numbers (e.g. `5`)
//
// So if there are any tensors of dim 1 or higher, then 0-dim tensors do not
// affect the result type. This behavior was chosen to preserve backwards
// compatibility and is *likely to change* in the near future.
// (See https://github.com/pytorch/pytorch/issues/9515)
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

// Returns the result type from given mixed type Tensors.
//
// Fails if Tensor backends differ.
//
// Scalars need to b represented as 0-dim tensors with is_wrapped_number=true
// in order to be regarded as non-tensor numbers per the order rules above.
CAFFE2_API ScalarType resultType(ArrayRef<ScalarTypeSource> inputs);

// Tries to cast mixed-type operands to output type. If the output type
// is not provided, operands will be casted into resultType.
//
// Returned Tensor and Scalars share the same scalar type if the casting is
// successful.
//
// e.g.
//   Tensor op(Tensor& out, Tensor self, Tensor other, Scalar alpha) {
//     std::tie(self, other, alpha) =
//         castOperands(out.scalar_type(), self, other, alpha)
//     ...
//   }
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

// Promotes mixed-type operands to result type.
//
// Returned Tensor and Scalars share the same scalar type.
//
// e.g.
//   Tensor op(Tensor self, Tensor other, Scalar alpha) {
//     std::tie(self, other, alpha) = castOperandsToResultType(self, other, alpha)
//     ...
//   }
template<typename... T>
static inline std::tuple<T...> castOperandsToResultType(T... args) {
  return castOperands<T...>(c10::nullopt, std::forward<T>(args)...);
}

}  // namespace at
