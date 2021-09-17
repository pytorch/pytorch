#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torch_lazy_tensors {

// Makes a deep copy of an ATEN tensor.
at::Tensor CopyTensor(const at::Tensor& ref);

// Same as above, with an additional cast.
at::Tensor CopyTensor(const at::Tensor& ref, at::ScalarType dest_type,
                      bool copy = true);

// Return at::ScalarType from at::Scalar
at::ScalarType GetScalarType(const at::Scalar& scalar);

template <typename T>
at::Scalar MakeIntScalar(T value) {
  return at::Scalar(static_cast<int64_t>(value));
}

template <typename T>
at::Scalar MakeFloatScalar(T value) {
  return at::Scalar(static_cast<double>(value));
}

template <typename T, typename S>
T OptionalOr(const c10::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

// Unwraps tensor to target dtype if it's a wrapped number.
at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype);

// Checks whether a c10::optional<Tensor> is defined.
inline bool IsDefined(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined();
}

}  // namespace torch_lazy_tensors
