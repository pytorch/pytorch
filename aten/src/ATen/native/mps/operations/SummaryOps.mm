//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/bincount_native.h>
#include <ATen/ops/stack.h>
namespace at::native {

Tensor _bincount_mps(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  TORCH_CHECK_NOT_IMPLEMENTED(isIntegralType(self.scalar_type(), false) && !isBarebonesUnsignedType(self.scalar_type()),
                              "bincount not implemented for ",
                              self.dtype());
  TORCH_CHECK(minlength >= 0, "minlength should be >= 0");

  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, kLong, std::nullopt /* layout */, kMPS, std::nullopt /* pin_memory */);
  }
  TORCH_CHECK(self.dim() == 1, "bincount only supports 1-d non-negative integral inputs.");

  auto [min, max] = at::aminmax(self);
  auto minmax = at::stack({min, max}).cpu();
  TORCH_CHECK(minmax[0].item<int64_t>() >= 0, "bincount only supports 1-d non-negative integral inputs.");

  bool has_weights = weights.defined();
  TORCH_CHECK(!(has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))),
              "weights should be 1-d and have the same length as input");

  const int64_t nbins = std::max(minmax[1].item<int64_t>() + 1L, minlength);
  if (has_weights) {
    const auto weights_ = weights.to(kFloat);
    return at::zeros({nbins}, weights_.options()).index_add_(0, self.to(kLong), weights_);
  }
  return self.to(kLong).histc(nbins, 0, nbins);
}

} // namespace at::native
