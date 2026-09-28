#pragma once

#include <optional>

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>

// C++ versions of what Python FakeTensorMode runs for elementwise binary ops
// (the torch._refs implementation and the fake_impls fast path), for
// SymInt-aware Meta kernels that match Python fake tensor exactly.
namespace at::native {

// ELEMENTWISE_TYPE_PROMOTION_KIND
enum class TypePromotionKind { DEFAULT, INT_TO_FLOAT };

Tensor binary_ref_meta(
    const Tensor& self,
    const Tensor& other,
    TypePromotionKind kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha = std::nullopt);

Tensor fast_binary_meta(const Tensor& self, const Tensor& other, TypePromotionKind kind);

} // namespace at::native
