#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Scalar.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/scalar_tensor.h>
#endif

namespace at::detail {
// When filling a number to 1-element CPU tensor, we want to skip
// everything but manipulate data ptr directly.
// Ideally this fast pass should be implemented in TensorIterator,
// but we also want to skip compute_types which in not avoidable
// in TensorIterator for now.
Tensor& scalar_fill(Tensor& self, const Scalar& value);
TORCH_API Tensor scalar_tensor_static(
    const Scalar& s,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Device> device_opt);
} // namespace at::detail

// This is in the c10 namespace because we use ADL to find the functions in it.
namespace c10 {

// FIXME: this should be (and was) Scalar::toTensor, but there is currently no
// way to implement this without going through Derived Types (which are not part
// of core).
inline at::Tensor scalar_to_tensor(
    const Scalar& s,
    const Device device = at::kCPU) {
  // This is the fast track we have for CPU scalar tensors.
  if (device == at::kCPU) {
    return at::detail::scalar_tensor_static(s, s.type(), at::kCPU);
  }
  return at::scalar_tensor(s, at::device(device).dtype(s.type()));
}

} // namespace c10

namespace at::native {

inline Tensor wrapped_scalar_tensor(
    const Scalar& scalar,
    const Device device = at::kCPU) {
  auto tensor = scalar_to_tensor(scalar, device);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

} // namespace at::native
