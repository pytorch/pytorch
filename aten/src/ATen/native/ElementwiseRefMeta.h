#pragma once

#include <optional>
#include <utility>

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>

// C++ versions of what Python FakeTensorMode runs for elementwise binary ops
// (the torch._refs implementation and the fake_impls fast path), for
// SymInt-aware Meta kernels that match Python fake tensor exactly.
namespace at::native {

enum class ELEMENTWISE_TYPE_PROMOTION_KIND { DEFAULT, INT_TO_FLOAT, ALWAYS_BOOL };

// utils.elementwise_dtypes: the (computation, result) dtypes.
std::pair<ScalarType, ScalarType> elementwise_dtypes(
    const Tensor& a,
    const Tensor& b,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind);

Tensor binary_ref_meta(
    const Tensor& self,
    const Tensor& other,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha = std::nullopt);

// _make_elementwise_binary_reference, with its Python scalar checks. name is
// the prim's name, used in the error messages.
Tensor elementwise_binary_ref_meta(
    const char* name,
    const Tensor& self,
    const Tensor& other,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    bool supports_lhs_python_scalar = true);

// other is a Scalar argument, which the ref sees as a Python number.
Tensor elementwise_binary_ref_meta(
    const char* name,
    const Tensor& self,
    const Scalar& other,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    bool supports_lhs_python_scalar = true);

// Whether Python fake sees the operand as symbolic: symbolic sizes, or a SymInt
// passed for a Tensor argument.
bool is_symbolic_operand(const Tensor& t);

void check_inplace_broadcast(c10::SymIntArrayRef self_shape, c10::SymIntArrayRef other_shape);

Tensor fast_binary_impl(const Tensor& self, const Tensor& other, ELEMENTWISE_TYPE_PROMOTION_KIND kind);

} // namespace at::native
