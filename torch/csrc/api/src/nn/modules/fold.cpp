#include <torch/nn/modules/fold.h>

#include <torch/expanding_array.h>
#include <torch/types.h>
#include <torch/utils.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

FoldImpl::FoldImpl(const FoldOptions& options_) : options(options_) {}

void FoldImpl::reset() {}

void FoldImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Fold(output_size=" << options.output_size()
         << ", kernel_size=" << options.kernel_size()
         << ", dilation=" << options.dilation()
         << ", padding=" << options.padding()
         << ", stride=" << options.stride()
         << ")";
}

Tensor FoldImpl::forward(const Tensor& input) {
  return F::detail::fold(
    input,
    options.output_size(),
    options.kernel_size(),
    options.dilation(),
    options.padding(),
    options.stride());
}

// ============================================================================

UnfoldImpl::UnfoldImpl(const UnfoldOptions& options_) : options(options_) {}

void UnfoldImpl::reset() {}

void UnfoldImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Unfold(kernel_size=" << options.kernel_size()
         << ", dilation=" << options.dilation()
         << ", padding=" << options.padding()
         << ", stride=" << options.stride()
         << ")";
}

Tensor UnfoldImpl::forward(const Tensor& input) {
  return F::detail::unfold(input, options.kernel_size(), options.dilation(), options.padding(), options.stride());
}

} // namespace nn
} // namespace torch
