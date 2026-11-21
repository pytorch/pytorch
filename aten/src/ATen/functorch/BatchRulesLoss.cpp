// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at::functorch {
// Flattens out all dims except the batch dim, and also moves batch dim
// (if it exists) to front.
static at::Tensor flatten_logical(const Tensor& tensor, std::optional<int64_t> bdim) {
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
static std::tuple<at::Tensor, std::optional<int64_t>>
loss_batch_rule_helper(const at::Tensor& self, std::optional<int64_t> self_bdim, const at::Tensor& target,
          std::optional<int64_t> target_bdim, int64_t reduction,
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
}

static std::tuple<at::Tensor, std::optional<int64_t>>
mse_loss_batch_rule(const at::Tensor& self, std::optional<int64_t> self_bdim, const at::Tensor& target,
          std::optional<int64_t> target_bdim, int64_t reduction) {
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::mse_loss(self, target, reduction);
                                });
}

static std::tuple<at::Tensor, std::optional<int64_t>>
huber_loss_batch_rule(const at::Tensor& self, std::optional<int64_t> self_bdim, const at::Tensor& target,
          std::optional<int64_t> target_bdim, int64_t reduction, double delta) {
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [delta](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::huber_loss(self, target, reduction, delta);
                                });
}

static std::tuple<at::Tensor, std::optional<int64_t>>
smooth_l1_loss_batch_rule(const at::Tensor& self, std::optional<int64_t> self_bdim, const at::Tensor& target,
          std::optional<int64_t> target_bdim, int64_t reduction, double beta) {
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [beta](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::smooth_l1_loss(self, target, reduction, beta);
                                });
}

static Tensor apply_loss_reduction(
    const at::Tensor& unreduced,
    int64_t reduction,
    std::optional<int64_t> dim = std::nullopt) {
  if (reduction == at::Reduction::Mean) {
    return dim.has_value() ? unreduced.mean(*dim) : unreduced.mean();
  }
  if (reduction == at::Reduction::Sum) {
    return dim.has_value() ? unreduced.sum(*dim) : unreduced.sum();
  }
  return unreduced;
}

static std::tuple<Tensor, Tensor, int64_t, bool, VmapDimVector> multi_margin_loss_prepare_inputs(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    const Tensor& target,
    std::optional<int64_t> target_bdim,
    int64_t bdim_size) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto target_ = moveBatchDimToFront(target, target_bdim);
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), bdim_size);
  target_ = ensure_has_bdim(target_, target_bdim.has_value(), bdim_size);

  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  TORCH_CHECK_VALUE(
      self_logical_rank <= 2,
      "vmap: Expected input for multi_margin_loss to have logical rank <= 2, but got ",
      self_logical_rank);

  VmapDimVector self_logical_sizes;
  self_logical_sizes.reserve(self_logical_rank);
  for (int64_t dim = 1; dim < self_.dim(); dim++) {
    self_logical_sizes.push_back(self_.size(dim));
  }

  Tensor self_flat;
  Tensor target_flat;
  int64_t nframe = 0;
  bool target_has_logical_dim = rankWithoutBatchDim(target, target_bdim) > 0;
  if (self_logical_rank <= 1) {
    self_flat = self_.reshape({bdim_size, -1});
    target_flat = target_.reshape({bdim_size});
    nframe = 1;
  } else {
    nframe = self_.size(1);
    const auto nclass = self_.size(2);
    self_flat = self_.reshape({bdim_size * nframe, nclass});
    target_flat = target_.reshape({bdim_size * nframe});
  }
  return std::make_tuple(
      std::move(self_flat),
      std::move(target_flat),
      nframe,
      target_has_logical_dim,
      std::move(self_logical_sizes));
}

static Tensor multi_margin_loss_restore_forward(
    const Tensor& result,
    int64_t bdim_size,
    int64_t nframe,
    bool target_has_logical_dim,
    int64_t reduction) {
  auto result_ = result.reshape({bdim_size, nframe});
  if (reduction == Reduction::None) {
    return target_has_logical_dim ? result_ : result_.squeeze(1);
  }
  if (nframe == 1) {
    return result_.squeeze(1);
  }
  return apply_loss_reduction(result_, reduction, /*dim=*/1);
}

// The native op takes weight as a single shared per-class vector (length nclass),
// so it has no batch dim to absorb a batched weight. Its only effect is the scalar
// factor weight[target] per frame, which we compute here to apply outside the op.
static Tensor multi_margin_loss_per_frame_weight(
    const Tensor& weight,
    std::optional<int64_t> weight_bdim,
    const Tensor& target_flat,
    int64_t bdim_size,
    int64_t nframe,
    int64_t nclass) {
  auto weight_ = moveBatchDimToFront(weight, weight_bdim);
  weight_ = ensure_has_bdim(weight_, weight_bdim.has_value(), bdim_size);
  // The native op validates weight against nclass, but the batched path passes
  // weight=None to it, so re-check here to stay in parity with eager instead of
  // silently gathering with a mismatched (or wrong-rank) weight.
  TORCH_CHECK(
      weight_.dim() == 2 && weight_.size(-1) == nclass,
      "inconsistent weight size, expected ", nclass, " but got ",
      weight_.sizes().slice(1));
  auto per_frame = weight_.gather(1, target_flat.view({bdim_size, nframe}));
  return per_frame.reshape({bdim_size * nframe});
}

static std::tuple<at::Tensor, std::optional<int64_t>>
multi_margin_loss_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    const Tensor& target,
    std::optional<int64_t> target_bdim,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight_opt,
    std::optional<int64_t> weight_bdim,
    int64_t reduction) {
  const bool weight_batched = weight_bdim.has_value();
  const auto bdim_size = weight_batched
      ? get_bdim_size3(self, self_bdim, target, target_bdim, *weight_opt, weight_bdim)
      : get_bdim_size2(self, self_bdim, target, target_bdim);
  // forward does not need self_logical_sizes (only the backward reshapes grad_input to it)
  [[maybe_unused]] auto [self_flat, target_flat, nframe, target_has_logical_dim, self_logical_sizes] =
      multi_margin_loss_prepare_inputs(self, self_bdim, target, target_bdim, bdim_size);
  auto result = at::multi_margin_loss(
      self_flat, target_flat, p, margin,
      weight_batched ? std::optional<Tensor>() : weight_opt, Reduction::None);
  if (weight_batched) {
    result = result * multi_margin_loss_per_frame_weight(
                          *weight_opt, weight_bdim, target_flat, bdim_size, nframe,
                          self_flat.size(1));
  }
  result = multi_margin_loss_restore_forward(
      result, bdim_size, nframe, target_has_logical_dim, reduction);
  return std::make_tuple(std::move(result), 0);
}

static Tensor multi_margin_loss_prepare_grad_output(
    const Tensor& grad_output,
    std::optional<int64_t> grad_output_bdim,
    int64_t bdim_size,
    int64_t nframe,
    int64_t reduction) {
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  grad_output_ = ensure_has_bdim(
      grad_output_, grad_output_bdim.has_value(), bdim_size);
  if (reduction == Reduction::None) {
    return grad_output_.reshape({bdim_size * nframe});
  }
  return grad_output_.reshape({bdim_size})
      .unsqueeze(1)
      .expand({bdim_size, nframe})
      .reshape({bdim_size * nframe});
}

static std::tuple<at::Tensor, std::optional<int64_t>>
multi_margin_loss_backward_batch_rule(
    const Tensor& grad_output,
    std::optional<int64_t> grad_output_bdim,
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    const Tensor& target,
    std::optional<int64_t> target_bdim,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight_opt,
    std::optional<int64_t> weight_bdim,
    int64_t reduction) {
  const bool weight_batched = weight_bdim.has_value();
  const auto bdim_size = weight_batched
      ? get_bdim_size4(grad_output, grad_output_bdim, self, self_bdim, target, target_bdim, *weight_opt, weight_bdim)
      : get_bdim_size3(grad_output, grad_output_bdim, self, self_bdim, target, target_bdim);
  // backward does not need target_has_logical_dim (only the forward uses it to squeeze)
  [[maybe_unused]] auto [self_flat, target_flat, nframe, target_has_logical_dim, self_logical_sizes] =
      multi_margin_loss_prepare_inputs(self, self_bdim, target, target_bdim, bdim_size);
  auto grad_output_flat = multi_margin_loss_prepare_grad_output(
      grad_output, grad_output_bdim, bdim_size, nframe, reduction);
  auto grad_input = at::multi_margin_loss_backward(
      grad_output_flat, self_flat, target_flat, p, margin,
      weight_batched ? std::optional<Tensor>() : weight_opt, Reduction::None);
  if (weight_batched) {
    // weight scales every frame's whole gradient row by weight[target].
    auto w = multi_margin_loss_per_frame_weight(
        *weight_opt, weight_bdim, target_flat, bdim_size, nframe, self_flat.size(1));
    grad_input = grad_input * w.unsqueeze(1);
  }
  if (reduction == Reduction::Mean && nframe > 1) {
    grad_input.div_(nframe);
  }

  VmapDimVector grad_input_shape;
  grad_input_shape.reserve(self_logical_sizes.size() + 1);
  grad_input_shape.push_back(bdim_size);
  grad_input_shape.insert(grad_input_shape.end(), self_logical_sizes.begin(), self_logical_sizes.end());
  grad_input = grad_input.reshape(grad_input_shape);
  return std::make_tuple(std::move(grad_input), 0);
}

static Tensor binary_cross_entropy_plumbing(
    const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "binary_cross_entropy_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
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
    result = at::binary_cross_entropy(self_, target_, std::nullopt, Reduction::None);
    result = makeBatched(result, 0, cur_level);
  } else {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    result = at::binary_cross_entropy(self_value, target_value, std::nullopt, Reduction::None);
  }
  if (weight.has_value() && weight->defined()) {
    result = result * weight.value();
  }
  return apply_loss_reduction(result, reduction);
}

static Tensor binary_cross_entropy_backward_plumbing(
    const Tensor& grad, const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight_opt, int64_t reduction) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "binary_cross_entropy_backward_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
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
        grad_, input_, target_, std::nullopt, Reduction::None);
    grad_input = makeBatched(grad_input, 0, cur_level);
  } else {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    grad_input = at::binary_cross_entropy_backward(
        grad_value, input_value, target_value, std::nullopt, Reduction::None);
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
  VMAP_SUPPORT(multi_margin_loss, multi_margin_loss_batch_rule);
  VMAP_SUPPORT(multi_margin_loss_backward, multi_margin_loss_backward_batch_rule);
  m.impl("binary_cross_entropy", binary_cross_entropy_plumbing);
  m.impl("binary_cross_entropy_backward", binary_cross_entropy_backward_plumbing);
}

} // namespace at::functorch
