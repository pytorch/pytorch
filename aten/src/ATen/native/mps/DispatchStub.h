#pragma once

#include <ATen/native/mps/GridSamplerMPS.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using grid_sampler_2d_backward_fn = std::tuple<Tensor, Tensor>(*)(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, bool, std::array<bool, 2>);
DECLARE_DISPATCH(grid_sampler_2d_backward_fn, grid_sampler_2d_backward_stub);

} // namespace native
} // namespace at
