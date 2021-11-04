#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/cpu/vml.h>

#include <tuple>

namespace at { namespace native {

using forward_2d_fn = Tensor(*)(const Tensor &, const Tensor &, int64_t, int64_t, bool);
using backward_2d_fn = std::tuple<Tensor, Tensor>(*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool, std::array<bool,2>);
DECLARE_DISPATCH(forward_2d_fn, grid_sampler_2d_cpu_kernel);
DECLARE_DISPATCH(backward_2d_fn, grid_sampler_2d_backward_cpu_kernel);

}}  // namespace at::native
