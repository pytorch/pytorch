#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at { namespace native {

using batch_norm_cpu_inference_contiguous_fn = void (*)(Tensor&, const Tensor&, const Tensor&, const Tensor&);
using batch_norm_cpu_inference_channels_last_fn = void (*)(Tensor&, const Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(batch_norm_cpu_inference_contiguous_fn, batch_norm_cpu_inference_contiguous_stub);
DECLARE_DISPATCH(batch_norm_cpu_inference_channels_last_fn, batch_norm_cpu_inference_channels_last_stub);

}} // namespace at::native
