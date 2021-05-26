// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/native/ResizeCommon.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>

#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/BatchingMetaprogramming.h>
#include <functorch/csrc/VmapTransforms.h>
#include <functorch/csrc/BatchedFallback.h>
#include <functorch/csrc/Constants.h>

namespace at { namespace functorch {

Tensor moveBatchDimToFront(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
int64_t rankWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
optional<int64_t> valIfNonempty(optional<int64_t> maybe_empty, int64_t new_val);
int64_t getPhysicalDim(const Tensor& tensor, bool has_batch_dim, int64_t logical_dim);

void vmapIncompatibleInplaceError(const char* schema_name);

Tensor maybePadToLogicalRank(const Tensor& tensor, optional<int64_t> has_bdim, int64_t logical_rank);

#define VMAP_SUPPORT(op, batch_rule) \
  m.impl(op, PrimBatchRule7< \
      decltype(&batch_rule), &batch_rule, to_operator_t<decltype(batch_rule)> \
      >::apply);

template <typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>> basic_unary_batch_rule(
    const Tensor& tensor, optional<int64_t> batch_dim, ExtraArgs... extra_args) {
  return {Func(tensor, std::forward<ExtraArgs>(extra_args)...), batch_dim};
}

#define INVOKE(object,ptrToMember)  ((object).*(ptrToMember))

template <typename F, F Method, typename... ExtraArgs>
Tensor& unary_inplace_batch_rule(Tensor& self, optional<int64_t>, ExtraArgs... extra_args) {
  INVOKE(self, Method)(std::forward<ExtraArgs>(extra_args)...);
  return self;
}

Tensor reshape_dim_into(int64_t src, int64_t dst, const Tensor& x);
Tensor reshape_dim_outof(int64_t src, int64_t size1, const Tensor& x);

}}

