// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {
// Flattens out all dims except the batch dim, and also moves batch dim
// (if it exists) to front.
at::Tensor flatten_logical(const Tensor& tensor, optional<int64_t> bdim) {
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

std::tuple<at::Tensor,optional<int64_t>>
mse_loss_batch_rule(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction) {
  auto self_ = flatten_logical(self, self_bdim);
  auto target_ = flatten_logical(target, target_bdim);
  auto result = at::mse_loss(self_, target_, Reduction::None);
  if (result.dim() == 1) {
    return std::make_tuple(result, 0);
  } else if (reduction == Reduction::None) {
    return std::make_tuple(result, 0);
  } else if (reduction == Reduction::Sum) {
    return std::make_tuple(result.sum(-1), 0);
  } else if (reduction == Reduction::Mean) {
    return std::make_tuple(result.mean(-1), 0);
  }
  TORCH_INTERNAL_ASSERT(false);
};

std::tuple<at::Tensor,optional<int64_t>>
mse_loss_backward_batch_rule(
    const at::Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const at::Tensor& self, optional<int64_t> self_bdim,
    const at::Tensor& target, optional<int64_t> target_bdim,
    int64_t reduction) {
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto target_ = moveBatchDimToFront(target, target_bdim);
  if (reduction != Reduction::None && grad_output_bdim.has_value()) {
    // grad_output_ is of shape [N]. Input is of shape [N?, ...].
    // We need to view grad_output_ as shape [N, ...].
    auto self_rank_without_bdim = rankWithoutBatchDim(self, self_bdim);
    DimVector view_shape(self_rank_without_bdim + 1, 1);
    view_shape[0] = grad_output_.size(0);
    grad_output_ = grad_output_.view(view_shape);
  }
  auto result = at::mse_loss_backward(grad_output_, self_, target_, Reduction::None);
  if (reduction == Reduction::Mean) {
    return std::make_tuple(result / numelWithoutBatchDim(self, self_bdim), 0);
  }
  return std::make_tuple(result, 0);
};

std::tuple<Tensor, Tensor> nll_loss_forward_decomposition(
    const Tensor & self,
    const Tensor & target,
    const c10::optional<Tensor> & weight,
    int64_t reduction, int64_t ignore_index) {

  bool has_ignore_index = ignore_index >= 0;
  if (has_ignore_index) {
    // fallback
    if (target.dim() > 1) {
      static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::nll_loss_nd", "");
      return slow_fallback<Tensor, Tensor>(op, {self, target, weight, reduction, ignore_index});
    } else {
      static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::nll_loss_forward", "");
      return slow_fallback<Tensor, Tensor>(op, {self, target, weight, reduction, ignore_index});
    }
  }
  // self can be [N, C, ...] or [C]
  // target can be [N, ...] or []

  int64_t channel_dim = 1;
  if (self.dim() < 2) {
    channel_dim = 0;
  }
  auto self_ = self;
  Tensor weight_;

  if (weight && weight->defined()) {
    // Here is a specific case with reduction mean and non-batched tensors
    // https://github.com/pytorch/pytorch/issues/61309
    // In this case weight is cancelled: w * x[t] / w -> x[t]
    if (!(reduction == Reduction::Mean && self_.dim() < 2)) {
      // reshape weights to [1, C, 1, ..., 1]
      auto shape = weight->sizes();
      VmapDimVector new_shape(self_.dim(), 1);
      new_shape[channel_dim] = shape[0];
      weight_ = weight->reshape(new_shape);
      self_ = self_ * weight_;
    }
  }
  auto target_ = target.unsqueeze(channel_dim);
  // target can be [N, 1, ...] or [1]

  auto result = -at::gather(self_, channel_dim, target_).squeeze(channel_dim);
  auto total_weight = at::full(
      {}, result.numel(), self_.scalar_type(),
      self_.layout(), self_.device(), nullopt);

  // Apply the reduction
  if (result.dim() > 0) {
    if (reduction == Reduction::Sum) {
      result = result.sum();
    } else if (reduction == Reduction::Mean) {
      if (!weight || !weight->defined()) {
        result = result.mean();
      } else {
        TORCH_INTERNAL_ASSERT(weight_.defined());
        weight_ = weight_.expand(self_.sizes());
        auto wsum = at::gather(weight_, channel_dim, target_).squeeze(channel_dim);
        wsum = wsum.sum();
        result = result.sum() / wsum;
        total_weight = wsum;
      }
    }
  } else if (reduction == Reduction::Mean && weight && weight->defined()) {
    // here weight is [C] and target is [1]
    auto wsum = at::gather(*weight, channel_dim, target_).squeeze(channel_dim);
    total_weight = wsum.sum();
  }

  return std::make_tuple(result, total_weight);
}

std::tuple<at::Tensor,optional<int64_t>>
nll_loss_backward_self_target_batch_rule(
    const at::Tensor & grad_output, optional<int64_t> grad_bdim,
    const at::Tensor & self, optional<int64_t> self_bdim,
    const at::Tensor & target, optional<int64_t> target_bdim,
    int64_t reduction, const at::Tensor & total_weight) {
  TORCH_INTERNAL_ASSERT(self.dim() == 3 && target.dim() == 2);

  if (reduction == Reduction::None) {
    int64_t batch_size = self.size(*self_bdim);
    auto self_ = reshape_dim_into(*self_bdim, 0, self);
    auto target_ = reshape_dim_into(*target_bdim, 0, target);
    Tensor grad_output_;
    if (grad_bdim) {
      TORCH_INTERNAL_ASSERT(grad_output.dim() == 2);
      grad_output_ = reshape_dim_into(*grad_bdim, 0, grad_output);
    } else {
      TORCH_INTERNAL_ASSERT(grad_output.dim() == 1);
      grad_output_ = grad_output.repeat({ batch_size });
    }
    auto result = at::nll_loss_backward(
        grad_output_, self_, target_,
        nullopt, reduction, -100, total_weight);
    return std::make_tuple(reshape_dim_outof(0, batch_size, result), 0);
  }
  TORCH_INTERNAL_ASSERT(reduction == Reduction::Sum || reduction == Reduction::Mean);
  int64_t batch_size = self.size(*self_bdim);
  auto self_ = reshape_dim_into(*self_bdim, 0, self);
  auto target_ = reshape_dim_into(*target_bdim, 0, target);
  int64_t mean_groups = self_.size(0) / batch_size;
  Tensor grad_output_ = grad_output;
  if (reduction == Reduction::Mean) {
    grad_output_ = grad_output_ / mean_groups;
  }
  if (grad_bdim) {
    TORCH_INTERNAL_ASSERT(grad_output_.dim() == 1);
    grad_output_ = grad_output_.repeat({mean_groups});
  } else {
    grad_output_ = grad_output_.expand({self_.size(0)});
  }
  TORCH_INTERNAL_ASSERT(grad_output_.dim() == 1 && grad_output_.size(0) == self_.size(0));
  auto result = at::nll_loss_backward(
      grad_output_, self_, target_,
      nullopt, Reduction::None, -100, total_weight);
  return std::make_tuple(reshape_dim_outof(0, batch_size, result), 0);
}

at::Tensor nll_loss_backward_plumbing(
    const at::Tensor & grad_output, const at::Tensor & self,
    const at::Tensor & target, const c10::optional<at::Tensor> & weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);

  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);

  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);

  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(*weight, cur_level);
  }

  Tensor total_weight_value;
  optional<int64_t> total_weight_bdim;
  std::tie(total_weight_value, total_weight_bdim) = unwrapTensorAtLevel(total_weight, cur_level);

  if (!self_bdim && !target_bdim && !weight_bdim && !grad_output_bdim && !total_weight_bdim) {
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    return at::nll_loss_backward(
        grad_output_value, self_value, target_value,
        weight_value, reduction, ignore_index, total_weight_value);
  }

  if (self_bdim && target_bdim && (!weight || !weight->defined()) &&
      !total_weight_bdim && ignore_index < 0) {
    auto result = nll_loss_backward_self_target_batch_rule(
        grad_output_value, grad_output_bdim,
        self_value, self_bdim,
        target_value, target_bdim,
        reduction, total_weight);
    return makeBatched(std::get<0>(result), std::get<1>(result), cur_level);
  }

  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::nll_loss_backward", "");
  return slow_fallback<Tensor>(op, {grad_output, self, target, weight, reduction, ignore_index, total_weight});
}


TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  m.impl("nll_loss_forward", nll_loss_forward_decomposition);
  m.impl("nll_loss_backward", nll_loss_backward_plumbing);
  VMAP_SUPPORT("mse_loss", mse_loss_batch_rule);
  VMAP_SUPPORT("mse_loss_backward", mse_loss_backward_batch_rule);
}

}}
