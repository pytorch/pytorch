// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <iostream>
#include <ATen/Operators.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>


namespace at { namespace functorch {

// Note [Adding vmap support for an operator]
// Hey there! So you have an operator and you want to get it to work with vmap.
// For example, let's say you just invented the `sum.int` operator and want to make
// it so that the following works.
// >>> tensor = torch.randn(B, 3)
// >>> vmap(torch.sum, (0, None))(tensor, 0)` works
// There are three main ways to do so.
//
// Note [Writing batch rule for out-of-place operators]
// If your operator is out-of-place, you can write a batch rule for it.
// The batch rule defines how to perform the operator on inputs where each
// Tensor input may have an additional dimension that is being vmapped over.
// We refer to this dimension as the *batch dimension* or bdim for short.
//
// For example, let's consider writing a batch rule for
// `Tensor sum(const Tensor& self, int64_t dim)`. The signature of the
// batch rule has an additional optional<int64_t> argument after each
// Tensor argument and return. So, in this case, the batch rule has signature
//   tuple<Tensor,optional<int64_t>> sum_batch_rule(
//       const Tensor& self, optional<int64_t> self_bdim, int64_t dim);
//
// The vmap call above invokes the batch rule with `self = tensor`,
// `self_bdim = 0`, and `dim = 0`. Note that there are **no BatchedTensors**
// involved in this case; there exists some plumbing that automatically unwraps
// BatchedTensors before calling the batch rule.
//
// To write the logic of the batch rule: think about the semantics of the
// `sum` operation if `self` had an additional dimension (indicated by self_bdim):
// - If `self_bdim` is null, then we just do `result = self.sum(dim)` as usual
// - If `self_bdim` is not-null, then we need to modify `dim`. `dim` is equal
//   to whatever the user passed in (0 in this case), but we should actually
//   perform the reduction over dimension 1 and do `result = self.sum(1)`
//   because dim 0 is being vmapped over.
// Finally, we return the result as well as a new bdim
// - If `self_bdim` is null, then there's no batch dim in the result.
// - If `self_bdim` is not-null, then we return where the bdim is.
//   Since we invoked `result = self.sum(1)`, the bdim is still at dim 0.
//
// Now that we have written `sum_batch_rule`, we have to register it inside a
// TORCH_LIBRARY_IMPL block:
//   TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
//     ...
//     VMAP_SUPPORT("sum.int", sum_batch_rule);
//     ...
//   }
//
// Note [Reusing batch rules to add vmap support for a complicated operator]
// Can't figure out how to write a batch rule for a big operation? If the
// operation can be expressed as a composition of other operations that do have
// batch rules, then that is another way to add vmap support. For example,
// consider the following schema
//   func: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1)
// and assume we already have batching rules for basic arithmetic operators.
//
// To add vmap support, define a decomposition using the same signature:
//   Tensor addcmul_decomp(const Tensor& self, const Tensor& tensor1,
//                         const Tensor& tensor2, const Scalar& value) {
//     auto product = torch.mul(tensor1, tensor2);
//     return torch.add(self, product, value);
//   }
// And register it inside a TORCH_LIBRARY_IMPL block:
//   TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
//     ...
//     m.impl("addcmul", addcmul_decomp);
//     ...
//   }
//
// Note [Writing batch rule for in-place operators]
// TODO: This is kinda complicated. Saving this for a future date.

std::tuple<Tensor,optional<int64_t>> unsqueeze_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, rank + 1) + 1;
  return std::make_tuple(self_.unsqueeze(dim), 0);
}

// NB: repeat is not actually a view, but it is in this file
std::tuple<Tensor,optional<int64_t>> repeat_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    IntArrayRef sizes) {

  VmapDimVector sizes_with_bdim = { sizes.begin(), sizes.end() };
  sizes_with_bdim.insert(sizes_with_bdim.begin(), 1);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  while (self_.dim() < sizes_with_bdim.size()) {
    self_ = self_.unsqueeze(1);
  }
  return std::make_tuple(self_.repeat(sizes_with_bdim), 0);
}


std::tuple<Tensor,optional<int64_t>> diag_batch_rule(
    const Tensor& input,
    optional<int64_t> input_bdim,
    int64_t diagonal) {
  if (!input_bdim) {
    return std::make_tuple(at::diag(input, diagonal), nullopt);
  }
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto rank = rankWithoutBatchDim(input, input_bdim);

  if (rank == 1) {
    return std::make_tuple(at::diag_embed(input_, diagonal), 0);
  } else if (rank == 2) {
    return std::make_tuple(at::diagonal(input_.movedim(0, -1), diagonal).clone(), rank - 2);
  } else {
    throw std::runtime_error("Passed in an invalid shape to at::diag");
  }
}

std::tuple<Tensor,optional<int64_t>> _unsafe_view_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    IntArrayRef size) {
  VmapDimVector view_size(size);
  view_size.insert(view_size.begin() + *self_bdim, self.size(*self_bdim));

  return std::make_tuple(at::_unsafe_view(self, view_size), self_bdim);
}

Tensor trace_decomp(const Tensor& self) {
  return at::sum(at::diagonal(self));
}

std::tuple<Tensor,optional<int64_t>> flip_batch_rule(const Tensor& self, optional<int64_t> self_bdim, IntArrayRef dims) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  VmapDimVector new_dims;
  for (auto i: dims) {
    new_dims.push_back(getPhysicalDim(self, true, i));
  }
  return std::make_tuple(at::flip(self_, new_dims), 0);
}

const Tensor& resize__plumbing(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value() ||
      optional_memory_format == c10::MemoryFormat::Contiguous,
      "resize_: batching rule only supports None or Contiguous MemoryFormat");
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  TORCH_INTERNAL_ASSERT(self_bdim.has_value());

  // TODO: The following algorithm only works for batch dim == 0.
  // To get it to work for something else we need the ability to modify
  // the BatchDims attribute of BatchedTensorImpl
  TORCH_INTERNAL_ASSERT(self_bdim.value() == 0, "NYI: resize_ batch rule for batch dim != 0");

  // Resize the wrapped tensor
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  self_value = moveBatchDimToFront(self_value, self_bdim);
  VmapDimVector new_size(size);
  new_size.insert(new_size.begin(), self_value.size(*self_bdim));
  self_value.resize_(new_size);

  // Update the sizes and strides of the wrapper
  auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched);
  batched->refreshSizesAndStrides();

  return self;
}

std::tuple<Tensor, optional<int64_t>> squeeze_batch_rule(const Tensor& self, optional<int64_t> bdim) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());
  // Special case for scalar arrays to replicate PyTorch behavior.
  if (self.dim() == 1) {
    return std::make_tuple(self.alias(), bdim);
  }

  // Manually calculate the output shape by eliding all dimensions of
  // size 1 keeping track of where the batch index started and where it
  // ended up moving to. We also ensure we do not drop the batch index.
  auto shape = self.sizes();
  DimVector squeezed_sizes;
  bool before_batch_idx = true;
  int64_t new_batch_idx = 0;
  int64_t original_idx = 0;

  for (auto it = shape.begin(); it != shape.end(); ++it) {
    // Keep only dimensions != 1 and the batch dimension (irrespective of size).
    if (*it != 1 || original_idx == bdim) {
      squeezed_sizes.push_back(*it);
      if (original_idx == bdim) {
        before_batch_idx = false;
      }
      // Only increment for the dimensions that will be kept in the output.
      if (before_batch_idx) {
        ++new_batch_idx;
      }
    }
    ++original_idx;
  }

  auto result = self.view(squeezed_sizes);
  return std::make_tuple(result, c10::optional<int64_t>(new_batch_idx));
}

std::tuple<Tensor, optional<int64_t>> squeeze_dim_batch_rule(const Tensor& self, optional<int64_t> bdim, int64_t dim) {
  TORCH_INTERNAL_ASSERT(bdim.has_value());
  // Special case for scalar arrays to replicate PyTorch behavior.
  if (self.dim() == 1) {
    TORCH_CHECK(dim == 0, "Dimension is out of range (expected to be in range of [-1, 0], but got ", dim);
    return std::make_tuple(self.alias(), bdim);
  }

  // Calculate the proper offset if dim is negative.
  auto actual_dim = dim;
  if (dim < 0) {
    actual_dim = self.dim() + dim - 1;
  }
  if (actual_dim < bdim) {
    // Since dimension to be squeezed is before the batch dimension pass as-is.
    auto original_size = self.dim();
    auto result = self.squeeze(actual_dim);
    auto updated_batch_idx = *bdim;
    if (result.dim() != original_size) {
      // A column before batch dimension has been dropped so adjust accordingly.
      --updated_batch_idx;
    }
    return std::make_tuple(result, optional<int64_t>(updated_batch_idx));
  } else {
    // Since dimension to be squeezed is after the batch dimension adjust by one to account
    // for the original batch dimension. In this case batch dimension won't move.
    return std::make_tuple(self.squeeze(actual_dim + 1), bdim);
  }
}

std::tuple<std::vector<Tensor>, optional<int64_t>> chunk_batching_rule(const Tensor& self, optional<int64_t> self_bdim, int64_t chunks, int64_t dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  int64_t new_dim = getPhysicalDim(self, self_bdim.has_value(), dim);
  return std::make_tuple(at::chunk(self_, chunks, new_dim), 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("diag", diag_batch_rule);
  VMAP_SUPPORT("chunk", chunk_batching_rule);
  m.impl("flatten.using_ints", static_cast<decltype(&ATEN_FN2(flatten, using_ints))>(native::flatten));
  VMAP_SUPPORT("flip", flip_batch_rule);
  m.impl("trace", trace_decomp);
  VMAP_SUPPORT("tril", VARIADIC_BDIMS_BATCH_RULE(ATEN_FN(tril)));
  VMAP_SUPPORT("triu", VARIADIC_BDIMS_BATCH_RULE(ATEN_FN(triu)));
  VMAP_SUPPORT("repeat", repeat_batch_rule);
  VMAP_SUPPORT("_unsafe_view", _unsafe_view_batch_rule);
  VMAP_SUPPORT("unsqueeze", unsqueeze_batch_rule);
  m.impl("resize_", resize__plumbing);
  VMAP_SUPPORT("squeeze", squeeze_batch_rule);
  VMAP_SUPPORT("squeeze.dim", squeeze_dim_batch_rule);
}

}}
