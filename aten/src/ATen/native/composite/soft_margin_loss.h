#pragma once
#include <ATen/core/Reduction.h>
#include <ATen/core/TensorBody.h>

namespace at {
namespace native {

template <typename OPS>
inline Tensor apply_reduction(
    const Tensor& output,
    at::Reduction::Reduction reduction) {
  switch (reduction) {
    case at::Reduction::Mean:
      return OPS::mean(output);
    case at::Reduction::Sum:
      return OPS::sum(output);
    case at::Reduction::None:
      return output;
    default:
      TORCH_INTERNAL_ASSERT(false, "Invalid reduction: ", (int64_t) reduction);
  }
}

template <typename OPS>
Tensor& soft_margin_loss_out(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output) {
  // compute inplace variant of: output = at::log(1. + at::exp(-input *
  // target));
  OPS::neg_out(output, input);
  OPS::mul_(output, target);
  OPS::exp_(output);
  OPS::add_(output, 1.);
  OPS::log_(output);

  auto tmp = apply_reduction<OPS>(output, reduction);
  if (!tmp.is_same(output)) {
    OPS::resize_(output, {});
    OPS::copy_(output, tmp);
  }

  return output;
}

template <typename OPS>
Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  auto output = OPS::empty({0}, input.options());
  OPS::soft_margin_loss_out(output, input, target, reduction);
  return output;
}

} // namespace native
} // namespace at
