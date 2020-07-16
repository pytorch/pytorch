#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using forward_fn = void (*)(
    const Tensor& /* X */,
    const Tensor& /* gamma */,
    const Tensor& /* beta */,
    int64_t /* N */,
    int64_t /* C */,
    int64_t /* HxW */,
    int64_t /* group */,
    double /* eps */,
    Tensor* /* Y */,
    Tensor* /* mean */,
    Tensor* /* rstd */);

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
    Tensor* /* dX */,
    Tensor* /* dgamma */,
    Tensor* /* dbeta */);

DECLARE_DISPATCH(forward_fn, GroupNormKernel);
DECLARE_DISPATCH(backward_fn, GroupNormBackwardKernel);

} // namespace native
} // namespace at
