#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

TORCH_API const float* get_mxfp8_values();

// The default implementation emulates MXFP in FP32; future native CPU kernels can register with this stub.
using mxfp_mm_fn = void (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    Tensor&);

DECLARE_DISPATCH(mxfp_mm_fn, mxfp_mm_stub)

} // namespace at::native
