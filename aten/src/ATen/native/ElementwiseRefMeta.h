#pragma once

#include <optional>

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>

// C++ versions of what Python FakeTensorMode runs for elementwise binary ops
// (the torch._refs implementation and the fake_impls fast path), for
// SymInt-aware Meta kernels that match Python fake tensor exactly.
namespace at::native {

// ELEMENTWISE_TYPE_PROMOTION_KIND
enum class TypePromotionKind { DEFAULT, INT_TO_FLOAT, ALWAYS_BOOL };

Tensor binary_ref_meta(
    const Tensor& self,
    const Tensor& other,
    TypePromotionKind kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha = std::nullopt);

// _make_elementwise_binary_reference, with its Python scalar checks. name is
// the prim's name, used in the error messages.
Tensor elementwise_binary_ref_meta(
    const char* name,
    const Tensor& self,
    const Tensor& other,
    TypePromotionKind kind,
    bool fake_devices,
    bool supports_lhs_python_scalar = true);

// A Scalar argument as the Python number the refs see. Symbolic values are not
// kept; the refs only look at the Python type.
Tensor python_number(const Scalar& s);

// check_inplace_broadcast in _meta_registrations
void check_inplace_broadcast(c10::SymIntArrayRef self_shape, c10::SymIntArrayRef other_shape);

Tensor fast_binary_meta(const Tensor& self, const Tensor& other, TypePromotionKind kind);

} // namespace at::native
