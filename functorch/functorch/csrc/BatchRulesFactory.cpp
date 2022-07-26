// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include "c10/core/SymIntArrayRef.h"

namespace at { namespace functorch {

template <typename A, A a, typename C>
struct NewBlahBatchRuleHelper;

template <typename F, F Func, typename A, typename B, typename... T>
struct NewBlahBatchRuleHelper<F, Func, typelist<A, B, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      IntArrayRef shape,
      T... extra_args) {
    const auto bdim_size = tensor.size(batch_dim.value());
    VmapDimVector new_shape;
    new_shape.reserve(shape.size() + 1);
    new_shape.emplace_back(bdim_size);
    new_shape.insert(new_shape.end(), shape.begin(), shape.end());
    return std::make_tuple(Func(tensor, new_shape, std::forward<T>(extra_args)...), 0);
  }
};

// USAGE: NEW_BLAH_BATCH_RULE(at::new_zeros)
// INCORRECT USAGE: NEW_BLAH_BATCH_RULE(&at::new_zeros)
// It is important that this macro is not passed a function pointer!!
#define NEW_BLAH_BATCH_RULE(fn) SINGLE_ARG(\
    NewBlahBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

std::tuple<Tensor,optional<int64_t>> _new_zeros_with_same_feature_meta_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim,
    int64_t self_num_batch_dims) {
  // The "self, other" naming is too confusing
  // What this function really says is "create a new tangent for this base".
  const auto& base = other;
  const auto& base_bdim = other_bdim;
  const auto& tangent = self;
  const auto& tangent_bdim = self_bdim;

  // Three case:
  //          Case 1  Case 2  Case 3
  // base        [6]  [B, 6]  [B, 6]
  // tangent  [B, 5]     [5]  [B, 5]
  // result   [B, 6]  [B, 6]  [B, 6]

  // Case 2 & 3
  if (base_bdim) {
    auto base_ = moveBatchDimToFront(base, base_bdim);
    Tensor tangent_ = tangent;
    if (tangent_bdim.has_value()) {
      // tangent  [B, K0, K1, 5]
      // base_            [B, 6]
      // We want to move B to after the Ks, so that self_num_batch_dims
      // (which really means tangent_num_batch_dims) isn't interfered with.
      // [B, K0, K1, 6] -> [K0, K1, B, 6]
      //
      // [K0, K1, B, 6], [B, 5], 2 -> [K0, K1, B, 5]
      tangent_ = tangent.movedim(*tangent_bdim, self_num_batch_dims);
    }
    const auto result = at::_new_zeros_with_same_feature_meta(tangent_, base_, self_num_batch_dims);
    return std::make_tuple(result, self_num_batch_dims);
  }

  // Case 1:
  auto tangent_ = moveBatchDimToFront(tangent, tangent_bdim);
  auto result = at::_new_zeros_with_same_feature_meta(tangent_, base, self_num_batch_dims + 1);
  return std::make_tuple(result, 0);
}

bool _has_same_storage_numel_batch_rule(const Tensor& a, const Tensor& b) {
  return true;
}

Tensor new_empty_symint_decomp(
    const Tensor& self,
    SymIntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt
    ) {
  return self.new_empty(c10::asIntArrayRefSlow(size), dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  m.impl("_has_same_storage_numel", _has_same_storage_numel_batch_rule);
  VMAP_SUPPORT(ones_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(ones_like)));
  VMAP_SUPPORT(zeros_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(zeros_like)));
  VMAP_SUPPORT(empty_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(empty_like)));
  VMAP_SUPPORT(randn_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(randn_like)));
  VMAP_SUPPORT(rand_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(rand_like)));
  VMAP_SUPPORT(full_like, BASIC_UNARY_BATCH_RULE(ATEN_FN(full_like)));
  VMAP_SUPPORT(new_empty, NEW_BLAH_BATCH_RULE(ATEN_FN(new_empty)));
  m.impl("new_empty.SymInt", new_empty_symint_decomp);
  VMAP_SUPPORT(new_zeros, NEW_BLAH_BATCH_RULE(ATEN_FN(new_zeros)));
  VMAP_SUPPORT(new_ones, NEW_BLAH_BATCH_RULE(ATEN_FN(new_ones)));
  VMAP_SUPPORT(new_full, NEW_BLAH_BATCH_RULE(ATEN_FN(new_full)));
  VMAP_SUPPORT(_new_zeros_with_same_feature_meta, _new_zeros_with_same_feature_meta_batch_rule);
  // Not sure how to add the ones with irregular args to the mix cleanly (i.e. randint takes an extra int parameter)
}
}}
