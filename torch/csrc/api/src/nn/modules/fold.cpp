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
      options.output_size_,
      options.kernel_size_,
      options.dilation_,
      options.padding_,
      options.stride_);
}

} // namespace nn
} // namespace torch
