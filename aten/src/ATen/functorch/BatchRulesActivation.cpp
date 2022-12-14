// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>

// NB: most activation functions fit pointwise unary or binary rules.
// These are only the ones that have special batch rules to help with organization
namespace at { namespace functorch {
std::tuple<Tensor,optional<int64_t>>
glu_batch_rule(const Tensor& self, optional<int64_t> self_bdim, int64_t dim) {
  // repeated error message from glu because 0D -> 1D when batched
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 1, "glu does not support 0-dimensional tensors");

  const auto rank = rankWithoutBatchDim(self, self_bdim);
  const auto dim_ = maybe_wrap_dim(dim, rank) + 1;

  const auto self_ = moveBatchDimToFront(self, self_bdim);

  const auto res = at::glu(self_, dim_);
  return std::make_tuple(res, 0);
}

std::tuple<Tensor,optional<int64_t>> glu_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim) {
  if (self_bdim) {
    // repeated error message from glu because 0D -> 1D when batched
    // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
    // can't be evenly halved, but give a nicer error message here.
    TORCH_CHECK(self.dim() > 1, "glu does not support 0-dimensional tensors");
  }

  const auto rank = rankWithoutBatchDim(self, self_bdim);
  const auto dim_ = maybe_wrap_dim(dim, rank) + 1;

  const auto batch_size = get_bdim_size2(grad_output, grad_output_bdim, self, self_bdim);
  const auto grad_output_ = ensure_has_bdim(moveBatchDimToFront(grad_output, grad_output_bdim), grad_output_bdim.has_value(), batch_size);
  const auto self_ = ensure_has_bdim(moveBatchDimToFront(self, self_bdim), self_bdim.has_value(), batch_size);

  const auto res = at::glu_backward(grad_output_, self_, dim_);
  return std::make_tuple(res, 0);
}

std::tuple<Tensor,optional<int64_t>> prelu_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& weight, optional<int64_t> weight_bdim) {
  if (!weight_bdim && weight.dim() == 0) {
    return std::make_tuple(at::prelu(input, weight), input_bdim);
  }

  const auto input_ = moveBatchDimToFront(input, input_bdim);
  auto weight_flatten = moveBatchDimToFront(weight, weight_bdim);

  const auto weight_logical_dim = rankWithoutBatchDim(weight, weight_bdim);
  TORCH_CHECK(weight_logical_dim == 0 || weight_logical_dim == 1,
      "prelu: Expected `weight` to be a scalar or 1D tensor, but got ndim = ",
      weight_logical_dim);

  if (weight_flatten.dim() > 1) {
    // for an input [N, C, ...]
    // weight can be a non-vector but the total number of elements must be the same as C
    weight_flatten = at::flatten(weight_flatten, weight_bdim.has_value() ? 1 : 0, -1);
  }

  const int64_t input_logical_rank = rankWithoutBatchDim(input, input_bdim);
  VmapDimVector new_shape(weight_flatten.sizes().begin(), weight_flatten.sizes().end());
  const int64_t final_size = weight_bdim ? (input_logical_rank + 1) : input_logical_rank;
  new_shape.reserve(final_size);

  if (weight_flatten.dim() == 2 || !weight_bdim) {
    // if weight (without batching) is not a scalar, its size must match the "channel dimension" of input. To do the
    // decomposition, we pad the weight to

    // copies checks from prelu if the weight (without vmap) is not a scalar
    TORCH_CHECK(input_logical_rank > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    if (input_logical_rank > 1) {
      const auto channel_dim = input_bdim ? 2 : 1;
      channel_size = input_.size(channel_dim);
    }
    const auto weight_num = weight_flatten.size(-1);
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    // pads to the left so that the flattened shape matches up with the channel
    if (!weight_bdim) {
      new_shape.insert(new_shape.begin(), 1);
    } else {
      new_shape.insert(new_shape.begin() + 1, 1);
    }
  }

  for (int64_t i = new_shape.size(); i < final_size; i ++) {
    new_shape.push_back(1);
  }
  TORCH_INTERNAL_ASSERT((int64_t)new_shape.size() == final_size);
  const auto weight_padded = weight_flatten.view(new_shape);
  auto zero_tensor = at::zeros(1, input.options());

  // decomposes function,
  auto res = at::maximum(zero_tensor, input_) + weight_padded * at::minimum(zero_tensor, input_);
  return std::make_tuple(res, 0);
}

VmapDimVector ensure_shape_with_bdim(const Tensor& input, const bool has_bdim, const int64_t batch_size) {
  // helper function that get the size of input, ensuring that there's batch dim, without expanding input
  if (has_bdim) {
    // sad to have to copy but got garbage if tried to return an IntArrayRef and just do input.sizes()
    VmapDimVector new_shape(input.sizes().begin(), input.sizes().end());
    return new_shape;
  }
  VmapDimVector new_shape(1, batch_size);
  new_shape.reserve(input.dim() + 1);
  new_shape.insert(new_shape.end(), input.sizes().begin(), input.sizes().end());
  return new_shape;
}

VmapDimVector shape_maybe_with_bdim(const Tensor& input, const bool need_bdim, const bool has_bdim, const int64_t batch_size) {
  // if need_bdim, will return the input with a guaranteed bdim. If not, will return the input logical size (no batch dim)
  if (need_bdim) {
    return ensure_shape_with_bdim(input, has_bdim, batch_size);
  } else if (has_bdim) { // !need_bdim && has_bdim
    VmapDimVector new_shape(input.sizes().begin() + 1, input.sizes().end());
    return new_shape;
  } else { // !need_bdim && !has_bdim
    VmapDimVector new_shape(input.sizes().begin(), input.sizes().end());
    return new_shape;
  }
}

std::tuple<Tensor, Tensor> prelu_backward_batched(
    const Tensor& grad_out, const Tensor& self, const Tensor& weight,
    const VmapDimVector& self_grad_shape, const VmapDimVector& weight_grad_padded_shape, const VmapDimVector& weight_grad_shape) {
  // helper function that produces a batched gradient for prelu using a decomposition inspired by the AOTAutograd ones
  const auto input_grad_collector = at::where(self > 0, grad_out, weight * grad_out);
  const auto input_grad = native::sum_to_size(input_grad_collector, self_grad_shape);
  const auto weight_grad_collector = at::where(self > 0, at::zeros(1, self.options()), self * grad_out);
  const auto weight_grad_collector_2 = native::sum_to_size(weight_grad_collector, weight_grad_padded_shape);
  const auto weight_grad = weight_grad_collector_2.view(weight_grad_shape);
  return std::make_tuple(input_grad, weight_grad);
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>> prelu_backward_batch_rule(
    const Tensor& grad_out, optional<int64_t> grad_out_bdim,
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& weight, optional<int64_t> weight_bdim) {
  const auto batch_size = get_bdim_size3(grad_out, grad_out_bdim, self, self_bdim, weight, weight_bdim);
  const auto grad_out_ = moveBatchDimToFront(grad_out, grad_out_bdim);
  const auto self_ = moveBatchDimToFront(self, self_bdim);
  const auto self_size_with_bdim = ensure_shape_with_bdim(self_, self_bdim.has_value(), batch_size);
  if (!weight_bdim && weight.dim() == 0) {
    VmapDimVector weight_grad_shape(1, batch_size);
    VmapDimVector weight_grad_shape_padded(self_bdim.has_value() ? self.dim() : self.dim() + 1, 1);
    weight_grad_shape_padded[0] = batch_size;
    const auto grads = prelu_backward_batched(grad_out_, self_, weight, self_size_with_bdim, weight_grad_shape_padded, weight_grad_shape);
    return std::make_tuple(std::get<0>(grads), 0, std::get<1>(grads), 0);
  }
  const auto weight_ = moveBatchDimToFront(weight, weight_bdim);
  auto weight_flatten = weight_;
  if (weight_flatten.dim() > 1) {
    // for an input [N, C, ...]
    // weight can be a non-vector but the total number of elements must be the same as C
    weight_flatten = at::flatten(weight_flatten, weight_bdim.has_value() ? 1 : 0, -1);
  }

  const int64_t self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  VmapDimVector new_shape(weight_flatten.sizes().begin(), weight_flatten.sizes().end());
  const int64_t final_size = weight_bdim ? (self_logical_rank + 1) : self_logical_rank;
  new_shape.reserve(final_size);

  if (weight_flatten.dim() == 2 || !weight_bdim) {
    // if weight (without batching) is not a scalar, its size must match the "channel dimension" of input. To do the
    // decomposition, we pad the weight to

    // copies checks from prelu if the weight (without vmap) is not a scalar
    TORCH_CHECK(self_logical_rank > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    if (self_logical_rank > 1) {
      channel_size = self_.size(self_bdim.has_value() ? 2 : 1);
    }

    const auto weight_num = weight_flatten.size(-1);
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    // pads to the left so that the flattened shape matches up with the channel
    if (!weight_bdim) {
      new_shape.insert(new_shape.begin(), 1);
    } else {
      new_shape.insert(new_shape.begin() + 1, 1);
    }
  }

  for (int64_t i = new_shape.size(); i < final_size; i ++) {
    new_shape.push_back(1);
  }
  // weight grad does not depend on weight values. It is batched iff grad_out or self are batched
  const auto weight_grad_is_batched = grad_out_bdim.has_value() || self_bdim.has_value();

  const auto weight_padded = weight_flatten.view(new_shape);
  const auto weight_grad_shape = shape_maybe_with_bdim(weight_, weight_grad_is_batched, weight_bdim.has_value(), batch_size);
  const auto weight_padded_grad_shape = shape_maybe_with_bdim(weight_padded, weight_grad_is_batched, weight_bdim.has_value(), batch_size);

  const auto grads = prelu_backward_batched(grad_out_, self_, weight_padded, self_size_with_bdim, weight_padded_grad_shape, weight_grad_shape);
  return std::make_tuple(std::get<0>(grads), 0, std::get<1>(grads), (weight_grad_is_batched ? optional<int64_t>(0) : nullopt));
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(glu_backward, glu_backward_batch_rule);
  VMAP_SUPPORT(glu, glu_batch_rule);
  VMAP_SUPPORT(prelu, prelu_batch_rule)
  VMAP_SUPPORT(prelu_backward, prelu_backward_batch_rule)
}
}} // namespace at::functorch
