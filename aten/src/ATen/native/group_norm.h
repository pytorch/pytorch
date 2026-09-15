#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/MemoryFormat.h>
#include <cstdint>

namespace at {
class Tensor;

namespace native {

// The memory format group_norm computes in: channels_last is only
// preserved on backends that have channels_last group norm kernels.
TORCH_API c10::MemoryFormat group_norm_memory_format(const Tensor& input);

using forward_fn = void (*)(
    const Tensor& /* X */,
    const Tensor& /* gamma */,
    const Tensor& /* beta */,
    int64_t /* N */,
    int64_t /* C */,
    int64_t /* HxW */,
    int64_t /* group */,
    double /* eps */,
    Tensor& /* Y */,
    Tensor& /* mean */,
    Tensor& /* rstd */);

using backward_fn = void (*)(
    const Tensor& /* dY */,
    const Tensor& /* X */,
    const Tensor& /* mean */,
    const Tensor& /* rstd */,
    const Tensor& /* gamma */,
    int64_t /* N */,
    int64_t /* C */,
    int64_t /* HxW */,
    int64_t /* group */,
    Tensor& /* dX */,
    Tensor& /* dgamma */,
    Tensor& /* dbeta */);

DECLARE_DISPATCH(forward_fn, GroupNormKernel)
DECLARE_DISPATCH(backward_fn, GroupNormBackwardKernel)

} // namespace native
} // namespace at
