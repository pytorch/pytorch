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

std::tuple<at::Tensor,optional<int64_t>,at::Tensor,optional<int64_t>>
nll_loss_forward_self_target_batch_rule(
    const at::Tensor & self, optional<int64_t> self_bdim,
    const at::Tensor & target, optional<int64_t> target_bdim,
    int64_t reduction) {
  TORCH_INTERNAL_ASSERT(self.dim() == 3 && target.dim() == 2);

  if (reduction == Reduction::None) {
    int64_t batch_size = self.size(*self_bdim);
    auto self_ = reshape_dim_into(*self_bdim, 0, self);
    auto target_ = reshape_dim_into(*target_bdim, 0, target);
    auto result = at::nll_loss_forward(self_, target_, nullopt, reduction, -100);
    return std::make_tuple(
      reshape_dim_outof(0, batch_size, std::get<0>(result)), 0,
      std::get<1>(result), nullopt
    );
  } else if (reduction == Reduction::Sum) {
    int64_t batch_size = self.size(*self_bdim);
    auto self_ = reshape_dim_into(*self_bdim, 0, self);
    auto target_ = reshape_dim_into(*target_bdim, 0, target);
    auto res = at::nll_loss_forward(self_, target_, nullopt, Reduction::None, -100);
    auto output = std::get<0>(res);
    output = reshape_dim_outof(0, batch_size, output);
    auto total_weight = self_.new_full({}, output.size(-1));
    return std::make_tuple(
      output.sum(-1), 0,
      // NB: total_weight = 0 after Reduction::None
      total_weight, nullopt
    );
  } else if (reduction == Reduction::Mean) {
    int64_t batch_size = self.size(*self_bdim);
    auto self_ = reshape_dim_into(*self_bdim, 0, self);
    auto target_ = reshape_dim_into(*target_bdim, 0, target);
    auto res = at::nll_loss_forward(self_, target_, nullopt, Reduction::None, -100);
    auto output = std::get<0>(res);
    output = reshape_dim_outof(0, batch_size, output);
    auto total_weight = self_.new_full({}, output.size(-1));
    return std::make_tuple(
      output.mean(-1), 0,
      // NB: total_weight = 0 after Reduction::None
      total_weight, nullopt
    );
  }
  TORCH_INTERNAL_ASSERT(false);
}

std::tuple<at::Tensor,at::Tensor> nll_loss_forward_plumbing(
    const at::Tensor & self,
    const at::Tensor & target,
    const c10::optional<at::Tensor> & weight,
    int64_t reduction, int64_t ignore_index) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

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

  if (self_bdim && target_bdim && (!weight || !weight->defined()) && ignore_index < 0) {
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    auto results = nll_loss_forward_self_target_batch_rule(
        self_value, self_bdim, target_value, target_bdim, reduction);
    return std::make_tuple(
      makeBatched(std::get<0>(results), std::get<1>(results), cur_level),
      makeBatched(std::get<2>(results), std::get<3>(results), cur_level)
    );
  }

  if ((!weight || !weight->defined()) && ignore_index < 0) {
    // Decomposition: gather to get unreduced loss. 1 is for the C dim, that's always 1.
    // gather can handle arbitrary strides so it's a good candidate for a decomposition.
    auto target_ = target.unsqueeze(1);
    auto result = at::gather(self, 1, target_).squeeze(1);
    auto total_weight = at::full(
        {}, result.numel(), self.scalar_type(),
        self.layout(), self.device(), nullopt);

    // Apply the reduction
    switch (reduction) {
      case Reduction::None:
        return std::make_tuple(-result, total_weight);
      case Reduction::Sum:
        return std::make_tuple(-result.sum(), total_weight);
      case Reduction::Mean:
        return std::make_tuple(-result.mean(), total_weight);
    }
  }

  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::nll_loss_forward", "");
  return slow_fallback<Tensor,Tensor>(op, {self, target, weight, reduction, ignore_index});
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
  m.impl("nll_loss_forward", nll_loss_forward_plumbing);
  m.impl("nll_loss_backward", nll_loss_backward_plumbing);
  VMAP_SUPPORT("mse_loss", mse_loss_batch_rule);
  VMAP_SUPPORT("mse_loss_backward", mse_loss_backward_batch_rule);
}

}}
