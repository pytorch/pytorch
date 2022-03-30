#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/TensorIterator.h>
#include <c10/core/Scalar.h>

namespace at {
namespace native {

using lerp_fn_scalar = void (*)(
    at::TensorIteratorBase& iter,
    const Scalar& weight);

using lerp_fn_tensor = void (*)(
    at::TensorIteratorBase& iter);

DECLARE_DISPATCH(lerp_fn_scalar, lerp_kernel_scalar_weight);
DECLARE_DISPATCH(lerp_fn_tensor, lerp_kernel_tensor_weight);

} // namespace native
} // namespace at
