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

DECLARE_DISPATCH(forward_fn, LayerNormKernel);

} // namespace native
} // namespace at

#endif // ATEN_SRC_NATIVE_CPU_LAYER_NORM_KERNEL_H_
