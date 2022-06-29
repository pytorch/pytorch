#pragma once
#include <ATen/TensorIterator.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using spdiags_kernel_fn_t =
    void (*)(TensorIterator&, const Tensor&, Tensor&, Tensor&);

using spdiags_backward_kernel_fn_t = void (*)(TensorIterator&, Tensor&);

DECLARE_DISPATCH(spdiags_kernel_fn_t, spdiags_kernel_stub);
DECLARE_DISPATCH(spdiags_backward_kernel_fn_t, spdiags_backward_kernel_stub);

} // namespace native
} // namespace at
