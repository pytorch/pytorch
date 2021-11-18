// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>

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


TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("ones_like", BASIC_UNARY_BATCH_RULE(ATEN_FN(ones_like)));
  VMAP_SUPPORT("zeros_like", BASIC_UNARY_BATCH_RULE(ATEN_FN(zeros_like)));
  VMAP_SUPPORT("empty_like", BASIC_UNARY_BATCH_RULE(ATEN_FN(empty_like)));
  VMAP_SUPPORT("randn_like", BASIC_UNARY_BATCH_RULE(ATEN_FN(randn_like)));
  VMAP_SUPPORT("rand_like", BASIC_UNARY_BATCH_RULE(ATEN_FN(rand_like)));
  VMAP_SUPPORT("full_like", BASIC_UNARY_BATCH_RULE(ATEN_FN(full_like)));
  VMAP_SUPPORT("new_empty", NEW_BLAH_BATCH_RULE(ATEN_FN(new_empty)));
  VMAP_SUPPORT("new_zeros", NEW_BLAH_BATCH_RULE(ATEN_FN(new_zeros)));
  VMAP_SUPPORT("new_ones", NEW_BLAH_BATCH_RULE(ATEN_FN(new_ones)));
  VMAP_SUPPORT("new_full", NEW_BLAH_BATCH_RULE(ATEN_FN(new_full)));
  // Not sure how to add the ones with irregular args to the mix cleanly (i.e. randint takes an extra int parameter)
}
}}

