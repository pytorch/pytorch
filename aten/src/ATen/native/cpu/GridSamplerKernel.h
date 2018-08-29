#pragma once

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/native/DispatchStub.h"
#include "ATen/cpu/vml.h"

#include <tuple>

namespace at { namespace native {

using forward_fn = Tensor(*)(const Tensor &, const Tensor &, int64_t, int64_t);
using backward_fn = std::tuple<Tensor, Tensor>(*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t);
DECLARE_DISPATCH(forward_fn, grid_sampler_2d_cpu_kernel);
DECLARE_DISPATCH(backward_fn, grid_sampler_2d_backward_cpu_kernel);

}}  // namespace at::native
