// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <iostream>
#include <ATen/Operators.h>


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
  dim = maybe_wrap_dim(dim, rank + 1);
  if (self_bdim) {
    dim += 1;
  }
  return std::make_tuple(self_.unsqueeze(dim), valIfNonempty(self_bdim, 0));
}

// NB: repeat is not actually a view, but it is in this file
std::tuple<Tensor,optional<int64_t>> repeat_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    IntArrayRef sizes) {
  if (!self_bdim) {
    return std::make_tuple(self.repeat(sizes), nullopt);
  }

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
  if (!self_bdim) {
    return std::make_tuple(at::_unsafe_view(self, size), nullopt);
  }
  VmapDimVector view_size(size);
  view_size.insert(view_size.begin() + *self_bdim, self.size(*self_bdim));

  return std::make_tuple(at::_unsafe_view(self, view_size), self_bdim);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  m.impl("flatten.using_ints", static_cast<decltype(&ATEN_FN2(flatten, using_ints))>(native::flatten));
  VMAP_SUPPORT("unsqueeze", unsqueeze_batch_rule);
  VMAP_SUPPORT("repeat", repeat_batch_rule);
  VMAP_SUPPORT("diag", diag_batch_rule);
  VMAP_SUPPORT("triu", SINGLE_ARG(variadic_bdims_batch_rule<decltype(&ATEN_FN(triu)), &at::triu, int64_t>));
  VMAP_SUPPORT("tril", SINGLE_ARG(variadic_bdims_batch_rule<decltype(&ATEN_FN(tril)), &at::tril, int64_t>));
  VMAP_SUPPORT("_unsafe_view", _unsafe_view_batch_rule);
}

}}
