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

  bool has_ignore_index = ignore_index >= 0;
  Tensor ignore_index_mask;
  if (has_ignore_index) {
    ignore_index_mask = target != ignore_index;
    result = result * ignore_index_mask;
    total_weight = ignore_index_mask.sum().to(self_);
  }

  // Apply the reduction
  if (result.dim() > 0) {
    if (reduction == Reduction::Sum) {
      result = result.sum();
    } else if (reduction == Reduction::Mean) {
      if (!weight || !weight->defined()) {
        if (has_ignore_index) {
          TORCH_INTERNAL_ASSERT(ignore_index_mask.defined());
          // total_weight is ignore_index_mask.sum()
          result = result.sum() / total_weight;
        } else {
          result = result.mean();
        }
      } else {
        TORCH_INTERNAL_ASSERT(weight_.defined());
        weight_ = weight_.expand(self_.sizes());
        auto wsum = at::gather(weight_, channel_dim, target_).squeeze(channel_dim);
        if (has_ignore_index) {
          TORCH_INTERNAL_ASSERT(ignore_index_mask.defined());
          wsum = wsum * ignore_index_mask;
        }
        wsum = wsum.sum();
        result = result.sum() / wsum;
        total_weight = wsum;
      }
    }
  } else if (reduction == Reduction::Mean && weight && weight->defined()) {
    // here weight is [C] and target is [1]
    auto wsum = at::gather(*weight, channel_dim, target_).squeeze(channel_dim);
    if (has_ignore_index) {
      TORCH_INTERNAL_ASSERT(ignore_index_mask.defined());
      wsum = wsum * ignore_index_mask;
    }
    total_weight = wsum.sum();
  }

  return std::make_tuple(result, total_weight);
}

at::Tensor nll_loss_backward_decomposition(
    const at::Tensor & grad_output, const at::Tensor & self,
    const at::Tensor & target, const c10::optional<at::Tensor> & weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {

  int64_t channel_dim = 1;
  if (self.dim() < 2) {
    channel_dim = 0;
  }
  auto self_ = self;
  auto target_ = target.unsqueeze(channel_dim);

  auto grad_output_ = grad_output;
  if (reduction == Reduction::Mean) {
    grad_output_ = grad_output_ / total_weight;
  }

  auto grad_input = at::zeros_like(self);
  grad_input = at::scatter(grad_input, channel_dim, target_, -1.0);

  if (grad_output_.dim() < grad_input.dim() && grad_output_.dim() > 0) {
    grad_output_ = grad_output_.unsqueeze(channel_dim);
  }

  Tensor weight_;
  if (weight && weight->defined()) {
    auto shape = weight->sizes();
    VmapDimVector new_shape(self_.dim(), 1);
    new_shape[channel_dim] = shape[0];
    weight_ = weight->reshape(new_shape);
    grad_output_ = grad_output_ * weight_;
  }

  bool has_ignore_index = ignore_index >= 0;
  Tensor ignore_index_mask;
  if (has_ignore_index) {
    ignore_index_mask = target_ != ignore_index;
    grad_output_ = grad_output_ * ignore_index_mask;
  }

  return grad_input * grad_output_;
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  m.impl("nll_loss_forward", nll_loss_forward_decomposition);
  m.impl("nll_loss2d_forward", nll_loss_forward_decomposition);
  m.impl("nll_loss_backward", nll_loss_backward_decomposition);
  m.impl("nll_loss2d_backward", nll_loss_backward_decomposition);
  VMAP_SUPPORT("mse_loss", mse_loss_batch_rule);
  VMAP_SUPPORT("mse_loss_backward", mse_loss_backward_batch_rule);
}

}}
