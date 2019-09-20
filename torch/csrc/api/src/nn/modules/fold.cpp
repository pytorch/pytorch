#include <torch/nn/modules/fold.h>

#include <torch/expanding_array.h>
#include <torch/types.h>
#include <torch/utils.h>

namespace torch {
namespace nn {

Tensor FoldImpl::forward(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 3,
      "Input Error: Only 3D input Tensors are supported (got ",
      input.dim(),
      "D)");

  return torch::col2im(
      input,
      options.output_size(),
      options.kernel_size(),
      options.dilation(),
      options.padding(),
      options.stride());
}

} // namespace nn
} // namespace torch
