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
    Tensor* /* out */,
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
    Tensor* /* grad_input */,
    Tensor* /* grad_weight */,
    Tensor* /* grad_bias */);

DECLARE_DISPATCH(forward_fn, LayerNormKernel);
DECLARE_DISPATCH(backward_fn, LayerNormBackwardKernel);

} // namespace native
} // namespace at

#endif // ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_
