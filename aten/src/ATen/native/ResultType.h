#pragma once

#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>

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

namespace at {

struct Type;

namespace detail {

template<typename T>
static inline Tensor maybeWrapScalar(T arg) = delete;

template<>
Tensor maybeWrapScalar(Tensor arg) {
  return arg;
}

template<>
Tensor maybeWrapScalar(Scalar arg) {
  auto tensor = scalar_to_tensor(arg);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

template<typename T>
static inline T castToType(Type& type, T arg) = delete;

template<>
Tensor castToType(Type& type, Tensor arg) {
  return arg.toType(type.scalarType());
}

template<>
Scalar castToType(Type& type, Scalar arg) {
  return AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX(type, "castToType", [&] {
    // Deliberate unchecked converts.
    if (arg.isIntegral()) {
      return Scalar(c10::convert<scalar_t, int64_t>(arg.toLong()));
    } else if (arg.isFloatingPoint()) {
      return Scalar(c10::convert<scalar_t, double>(arg.toDouble()));
    } else if (arg.isComplex()) {
      return Scalar(c10::convert<scalar_t, std::complex<double>>(arg.toComplexDouble()));
    } else {
      AT_ERROR("Unknown scalar type");
    }
    return Scalar();
  });
}

} // namespace detail

/* Returns the result type from given mixed type Tensors.
 *
 * Fails if Tensor backends differ.
 *
 * Scalars need to b represented as 0-dim tensors with is_wrapped_number=true
 * in order to be regarded as non-tensor numbers per the order rules above.
 */
CAFFE2_API Type& resultType(TensorList tensors);

// Tries to cast mixed-type operands to output type.
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
  constexpr int size = std::tuple_size<std::tuple<T...>>::value;
  const std::array<Tensor, size> wrapped_args = {detail::maybeWrapScalar<T>(args)...};
  Type& result_type = resultType(wrapped_args);
  Type& output_type = dtype
      ? at::globalContext().getNonVariableType(result_type.backend(), *dtype)
      : result_type;
  if (dtype) {
    AT_CHECK(*dtype != ScalarType::Undefined);
    // Downcast any operand without complaining if the result type can
    // be casted into dtype. This is because resultType already allows
    // downcasting into smaller kinds, e.g. tensor(int) + float is an
    // integer tensor.
    if (!canCastSameKind(result_type.scalarType(), *dtype)) {
      for (const Tensor& wrapped_arg : wrapped_args) {
        // Result type cannot be downcasted into dtype. Check if all
        // inputs can still be downcasted without degrading kind.
        AT_CHECK(
            canCastSameKind(wrapped_arg.scalar_type(), *dtype),
            "Cannot coerce input of type ",
            wrapped_arg.scalar_type(),
            " into ",
            *dtype);
      }
    }
  }
  return std::make_tuple(detail::castToType<T>(output_type, std::forward<T>(args))...);
}

// Promotes mixed-type operands to result type.
//
// Returned Tensor and Scalars share the same scalar type.
//
// e.g.
//   Tensor op(Tensor self, Tensor other, Scalar alpha) {
//     std::tie(self, other, alpha) = promoteOperands(self, other, alpha)
//     ...
//   }
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
//     self: op_backward(grad, Scalar alpha).to(self.type())
template<typename... T>
static inline std::tuple<T...> promoteOperands(T... args) {
  return castOperands<T...>(c10::nullopt, std::forward<T>(args)...);
}

}  // namespace at
