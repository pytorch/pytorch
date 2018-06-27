#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#define MODE_ZEROS 0
#define MODE_BORDER 1

namespace at { namespace native {

Tensor grid_sampler(const Tensor& input, const Tensor& grid, int64_t padding_mode) {
  // as of cudnn 7102, will not work for larger than 1024
  if (at::native::cudnn_is_acceptable(input) && padding_mode == MODE_ZEROS && input.dim() == 4 && input.size(1) <= 1024) {
    return cudnn_grid_sampler(input, grid);
  }
  if (input.dim() == 4) {
    return thnn_grid_sampler_bilinear2d(input, grid, padding_mode);
  }
  if (input.dim() == 5) {
    return thnn_grid_sampler_bilinear3d(input, grid, padding_mode);
  }
  AT_ERROR("grid_sampler(): input must be 4d or 5d but got input of shape: ", input.dim());
}

}}  // namespace at::native
