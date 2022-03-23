#pragma once

// See NOTE: [Tensor vs. TensorBase]
// https://github.com/pytorch/pytorch/pull/66979
namespace at {
class TensorBase;
}

namespace at { namespace native {

void check_grid_sampler_common(const TensorBase& input, const TensorBase& grid);

void check_grid_sampler_2d(const TensorBase& input, const TensorBase& grid);

void check_grid_sampler_3d(
  const TensorBase& input,
  const TensorBase& grid,
  int64_t interpolation_mode);

bool cond_cudnn_grid_sampler(const TensorBase& input, const TensorBase& grid);

}}