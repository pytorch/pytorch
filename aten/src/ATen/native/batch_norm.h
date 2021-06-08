#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

namespace native {

using batch_norm_fn = void (*)(Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, double);

DECLARE_DISPATCH(batch_norm_fn, batch_norm_cpu_inference_contiguous_stub);

} // namespace native

} // namespace at
