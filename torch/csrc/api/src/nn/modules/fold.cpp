#include <torch/nn/modules/fold.h>

#include <torch/expanding_array.h>
#include <torch/types.h>
#include <torch/utils.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

FoldImpl::FoldImpl(const FoldOptions& options_) : options(options_) {
  reset();
}

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

// ============================================================================

UnfoldImpl::UnfoldImpl(const UnfoldOptions& options_) : options(options_) {}

void UnfoldImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Unfold(kernel_size=" << options.kernel_size()
         << ", dilation=" << options.dilation()
         << ", padding=" << options.padding()
         << ", stride=" << options.stride()
         << ")";
}

Tensor UnfoldImpl::forward(const Tensor& input) {
  return F::unfold(input, options);
}

} // namespace nn
} // namespace torch
