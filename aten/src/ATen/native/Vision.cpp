#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native {

Tensor grid_sampler(const Tensor& input, const Tensor& grid, int64_t padding_mode) {
  if (input.dim() == 4) {
    return thnn_grid_sampler_bilinear2d(input, grid, padding_mode);
  }
  if (input.dim() == 5) {
    return thnn_grid_sampler_bilinear3d(input, grid, padding_mode);
  }
  AT_ERROR("grid_sampler(): input must be 4d or 5d but got input of shape: ", input.dim());
}

std::tuple<Tensor, Tensor> grid_sampler_backward(const Tensor & grad, const Tensor & input, const Tensor & grid, int64_t padding_mode) {
  if (input.dim() == 4) {
    return thnn_grid_sampler_bilinear2d_backward(grad, input, grid, padding_mode);
  }
  if (input.dim() == 5) {
    return thnn_grid_sampler_bilinear3d_backward(grad, input, grid, padding_mode);
  }
  AT_ERROR("grid_sampler_backward(): input must be 4d or 5d but got input of shape: ", input.dim());
}


}}  // namespace at::native
