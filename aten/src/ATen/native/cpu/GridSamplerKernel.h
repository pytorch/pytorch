#pragma once

#include <ATen/native/DispatchStub.h>

#include <array>
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at { namespace native {

using forward_2d_fn = void (*) (
    const TensorBase &output,
    const TensorBase &input,
    const TensorBase &grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners);
using backward_2d_fn = void (*) (
    const TensorBase &grad_input,
    const TensorBase &grad_grid,
    const TensorBase &grad_output,
    const TensorBase &input,
    const TensorBase &grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask);
DECLARE_DISPATCH(forward_2d_fn, grid_sampler_2d_cpu_kernel);
DECLARE_DISPATCH(backward_2d_fn, grid_sampler_2d_backward_cpu_kernel);

}}  // namespace at::native
