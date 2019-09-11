#ifndef ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_
#define ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using forward_fn = void (*)(
    const Tensor& /* input */,
    const Tensor& /* weight */,
    const Tensor& /* bias */,
    int64_t /* M */,
    int64_t /* N */,
    double /* eps */,
    Tensor* /* Y */,
    Tensor* /* mean */,
    Tensor* /* rstd */);

using backward_fn = void (*)(
    const Tensor& /* grad_out */,
    const Tensor& /* input */,
    const Tensor& /* mean */,
    const Tensor& /* rstd */,
    const Tensor& /* weight */,
    int64_t /* M */,
    int64_t /* N */,
    Tensor* /* dinput */,
    Tensor* /* dweight */,
    Tensor* /* dbias */);

using double_backward_fn = void (*)(
    const Tensor& /* ddinput */,
    const Tensor& /* ddweight */,
    const Tensor& /* ddbias */,
    const Tensor& /* grad_out */,
    const Tensor& /* input */,
    const Tensor& /* mean */,
    const Tensor& /* rstd */,
    const Tensor& /* weight */,
    int64_t /* M */,
    int64_t /* N */,
    Tensor* /* dgrad_out */,
    Tensor* /* dinput */,
    Tensor* /* dweight */);

DECLARE_DISPATCH(forward_fn, LayerNormKernel);
DECLARE_DISPATCH(backward_fn, LayerNormBackwardKernel);
DECLARE_DISPATCH(double_backward_fn, LayerNormDoubleBackwardKernel);

} // namespace native
} // namespace at

#endif // ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_
