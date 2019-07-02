#ifndef ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_
#define ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using forward_fn = void (*)(
    const Tensor& /* X */,
    const Tensor& /* gamma */,
    const Tensor& /* beta */,
    int64_t /* M */,
    int64_t /* N */,
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
    int64_t /* M */,
    int64_t /* N */,
    Tensor* /* dX */,
    Tensor* /* dgamma */,
    Tensor* /* dbeta */);

using double_backward_fn = void (*)(
    const Tensor& /* ddX */,
    const Tensor& /* ddgamma */,
    const Tensor& /* ddbeta */,
    const Tensor& /* dY */,
    const Tensor& /* X */,
    const Tensor& /* mean */,
    const Tensor& /* rstd */,
    const Tensor& /* gamma */,
    int64_t /* M */,
    int64_t /* N */,
    Tensor* /* ddY */,
    Tensor* /* dX */,
    Tensor* /* dgamma */);

DECLARE_DISPATCH(forward_fn, LayerNormKernel);
DECLARE_DISPATCH(backward_fn, LayerNormBackwardKernel);
DECLARE_DISPATCH(double_backward_fn, LayerNormDoubleBackwardKernel);

} // namespace native
} // namespace at

#endif // ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_
