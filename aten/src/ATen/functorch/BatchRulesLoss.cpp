// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at::functorch {
// Flattens out all dims except the batch dim, and also moves batch dim
// (if it exists) to front.
static at::Tensor flatten_logical(const Tensor& tensor, optional<int64_t> bdim) {
  if (bdim.has_value()) {
    auto result = moveBatchDimToFront(tensor, bdim);
    if (result.dim() > 1) {
      return result.flatten(1);
    } else {
      return result;
    }
  } else {
    return tensor.flatten();
  }
}

// Useful for many loss functions
template <typename Func>
static std::tuple<at::Tensor,optional<int64_t>>
loss_batch_rule_helper(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction,
          Func loss_fn) {
  auto self_ = flatten_logical(self, self_bdim);
  auto target_ = flatten_logical(target, target_bdim);
  auto result = loss_fn(self_, target_, Reduction::None);
  if (result.dim() == 1) {
    return std::make_tuple(result, 0);
  } else if (reduction == Reduction::None) {
    DimVector end_shape;
    const auto batched_elem = self_bdim.has_value() ?
        moveBatchDimToFront(self, self_bdim) : moveBatchDimToFront(target, target_bdim);
    return std::make_tuple(result.reshape(batched_elem.sizes()), 0);
  } else if (reduction == Reduction::Sum) {
    return std::make_tuple(result.sum(-1), 0);
  } else if (reduction == Reduction::Mean) {
    return std::make_tuple(result.mean(-1), 0);
  }
  TORCH_INTERNAL_ASSERT(false);
};

static std::tuple<at::Tensor,optional<int64_t>>
mse_loss_batch_rule(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction) {
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::mse_loss(self, target, reduction);
                                });
};

static std::tuple<at::Tensor,optional<int64_t>>
huber_loss_batch_rule(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction, double delta) {
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [delta](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::huber_loss(self, target, reduction, delta);
                                });
};

static std::tuple<at::Tensor,optional<int64_t>>
smooth_l1_loss_batch_rule(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction, double beta) {
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [beta](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::smooth_l1_loss(self, target, reduction, beta);
                                });
};

static Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

static Tensor binary_cross_entropy_plumbing(
    const Tensor& self, const Tensor& target,
    const optional<Tensor>& weight, int64_t reduction) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "binary_cross_entropy_plumbing");
  int64_t cur_level = maybe_layer->layerId();

  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(target, cur_level)
      && !isBatchedAtLevel(weight, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::binary_cross_entropy(self, target, weight, reduction);
  }

  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [target_value, target_bdim] = unwrapTensorAtLevel(target, cur_level);

  Tensor result;
  if (self_bdim || target_bdim) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    const auto bdim_size = get_bdim_size2(self_value, self_bdim, target_value, target_bdim);
    auto self_ = moveBatchDimToFront(self_value, self_bdim);
    auto target_ = moveBatchDimToFront(target_value, target_bdim);
    self_ = ensure_has_bdim(self_, self_bdim.has_value(), bdim_size);
    target_ = ensure_has_bdim(target_, target_bdim.has_value(), bdim_size);
    result = at::binary_cross_entropy(self_, target_, nullopt, Reduction::None);
    result = makeBatched(result, 0, cur_level);
  } else {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    result = at::binary_cross_entropy(self_value, target_value, nullopt, Reduction::None);
  }
  if (weight.has_value() && weight->defined()) {
    result = result * weight.value();
  }
  return apply_loss_reduction(result, reduction);
}

static Tensor binary_cross_entropy_backward_plumbing(
    const Tensor& grad, const Tensor& input, const Tensor& target,
    const c10::optional<Tensor>& weight_opt, int64_t reduction) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "binary_cross_entropy_backward_plumbing");
  int64_t cur_level = maybe_layer->layerId();

  if (!areAnyBatchedAtLevel({grad, input, target, weight_opt}, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::binary_cross_entropy_backward(grad, input, target, weight_opt, reduction);
  }

  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(
      reduction == Reduction::None ? grad : grad.expand_as(input), cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [target_value, target_bdim] = unwrapTensorAtLevel(target, cur_level);

  Tensor grad_input;
  if (grad_bdim || input_bdim || target_bdim) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    const auto bdim_size = get_bdim_size3(
        grad_value, grad_bdim, input_value, input_bdim, target_value, target_bdim);

    auto grad_ = moveBatchDimToFront(grad_value, grad_bdim);
    auto input_ = moveBatchDimToFront(input_value, input_bdim);
    auto target_ = moveBatchDimToFront(target_value, target_bdim);

    grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), bdim_size);
    input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
    target_ = ensure_has_bdim(target_, target_bdim.has_value(), bdim_size);

    grad_input = at::binary_cross_entropy_backward(
        grad_, input_, target_, nullopt, Reduction::None);
    grad_input = makeBatched(grad_input, 0, cur_level);
  } else {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    grad_input = at::binary_cross_entropy_backward(
        grad_value, input_value, target_value, nullopt, Reduction::None);
  }
  if (weight_opt.has_value() && weight_opt->defined()) {
    grad_input = grad_input * weight_opt.value();
  }
  if (reduction == Reduction::Mean) {
    grad_input.div_(input.numel());
  }
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(mse_loss, mse_loss_batch_rule);
  // mse_loss_backward uses a decomposition for its batch rule
  VMAP_SUPPORT(huber_loss, huber_loss_batch_rule);
  // huber_loss_backward uses a decomposition for its batch rule
  VMAP_SUPPORT(smooth_l1_loss, smooth_l1_loss_batch_rule);
  // smooth_l1_loss_backward uses a decomposition for its batch rule
  m.impl("binary_cross_entropy", binary_cross_entropy_plumbing);
  m.impl("binary_cross_entropy_backward", binary_cross_entropy_backward_plumbing);
}

} // namespace at::functorch
