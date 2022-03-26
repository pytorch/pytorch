#include <ATen/ATen.h>
#include <ATen/native/GridSampler.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/IndexingUtils.h>

namespace at { namespace native {

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

// See NOTE [ grid_sampler Native Functions ].
void check_grid_sampler_common(const TensorBase& input_, const TensorBase& grid_) {
  auto input = Tensor(input_);
  auto grid = Tensor(grid_);

  auto input_opt = input.options();
  auto grid_opt = grid.options();

  TORCH_CHECK(
    input.defined() && grid.defined(),
    "grid_sampler(): expected input and grid to not be undefined, but input "
    "is ", input, " and grid is ", grid);
  TORCH_CHECK(
    input_opt.device() == grid_opt.device(),
    "grid_sampler(): expected input and grid to be on same device, but input "
    "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
  TORCH_CHECK(
    input_opt.layout() == kStrided && grid_opt.layout() == kStrided,
    "grid_sampler(): expected input and grid to have torch.strided layout, but "
    "input has ", input_opt.layout(), " and grid has ", grid_opt.layout());
  TORCH_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());

  for (const auto i : c10::irange(2, input.dim())) {
    TORCH_CHECK(input.size(i) > 0,
      "grid_sampler(): expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }
}

// See NOTE [ grid_sampler Native Functions ].
void check_grid_sampler_2d(const TensorBase& input_, const TensorBase& grid_) {
  auto input = Tensor(input_);
  auto grid = Tensor(grid_);

  TORCH_CHECK(
    input.dim() == 4 && input.dim() == grid.dim(),
    "grid_sampler(): expected 4D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
}

// See NOTE [ grid_sampler Native Functions ].
void check_grid_sampler_3d(
  const TensorBase& input_,
  const TensorBase& grid_,
  int64_t interpolation_mode
) {
  auto input = Tensor(input_);
  auto grid = Tensor(grid_);

  TORCH_CHECK(
    input.dim() == 5 && input.dim() == grid.dim(),
    "grid_sampler(): expected 5D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    !(input.dim() == 5 &&
      static_cast<GridSamplerInterpolation>(interpolation_mode) ==
        GridSamplerInterpolation::Bicubic),
    "grid_sampler(): bicubic interpolation only supports 4D input");
}

// See NOTE [ grid_sampler Native Functions ].
// cudnn does not support inputs larger than 1024.
bool cond_cudnn_grid_sampler(const TensorBase& input_, const TensorBase& grid_) {
  auto input = Tensor(input_);
  auto grid = Tensor(grid_);

  return (
    at::native::cudnn_is_acceptable(input) &&
    at::native::cudnn_is_acceptable(grid) &&
    at::native::canUse32BitIndexMath(input) &&
    at::native::canUse32BitIndexMath(grid) &&
    input.dim() == 4 &&
    input.size(1) <= 1024);
}

}}
