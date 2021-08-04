// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/native/ResizeCommon.h>
#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <torch/csrc/autograd/variable.h>

#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/BatchingMetaprogramming.h>
#include <functorch/csrc/VmapTransforms.h>
#include <functorch/csrc/BatchedFallback.h>
#include <functorch/csrc/Constants.h>

namespace at { namespace functorch {
Tensor reshape_dim_into(int64_t src, int64_t dst, const Tensor& x);
Tensor reshape_dim_outof(int64_t src, int64_t size1, const Tensor& x);

Tensor moveBatchDimToFront(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
int64_t rankWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
int64_t numelWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
optional<int64_t> valIfNonempty(optional<int64_t> maybe_empty, int64_t new_val);
int64_t getPhysicalDim(const Tensor& tensor, bool has_batch_dim, int64_t logical_dim);

void vmapIncompatibleInplaceError(const char* schema_name);

Tensor maybePadToLogicalRank(const Tensor& tensor, optional<int64_t> has_bdim, int64_t logical_rank);

#define VMAP_SUPPORT(op, batch_rule) \
  m.impl(op, PrimBatchRule7< \
      decltype(&batch_rule), &batch_rule, to_operator_t<decltype(batch_rule)> \
      >::apply);

// DO NOT USE ME DIRECTLY! Use BASIC_UNARY_BATCH_RULE to save yourself some pain
template <typename A, A a, typename C>
struct BasicUnaryBatchRuleHelper;

template <typename F, F Func, typename A, typename... T>
struct BasicUnaryBatchRuleHelper<F, Func, typelist<A, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    return std::make_tuple(Func(tensor, std::forward<T>(extra_args)...), batch_dim);
  }
};

template <typename A, A a, typename C>
struct ScalarUnaryBatchRuleHelper;

template <typename F, F Func, typename S, typename A, typename... T>
struct ScalarUnaryBatchRuleHelper<F, Func, typelist<S, A, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Scalar& self,
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    return std::make_tuple(Func(self, tensor, std::forward<T>(extra_args)...), batch_dim);
  }
};



// USAGE: BASIC_UNARY_BATCH_RULE(at::sin)
// INCORRECT USAGE: BASIC_UNARY_BATCH_RULE(&at::sin)
// It is important that this macro is not passed a function pointer!!
#define BASIC_UNARY_BATCH_RULE(fn) SINGLE_ARG(\
    BasicUnaryBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

#define SCALAR_UNARY_BATCH_RULE(fn) SINGLE_ARG(\
    ScalarUnaryBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

template <typename A, A a, typename C>
struct VariadicBdimsBatchRuleHelper;

template <typename F, F Func, typename A, typename... T>
struct VariadicBdimsBatchRuleHelper<F, Func, typelist<A, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    auto tensor_ = moveBatchDimToFront(tensor, batch_dim);
    return std::make_tuple(Func(tensor_, std::forward<T>(extra_args)...), 0);
  }
};

// USAGE: VARIADIC_BDIMS_BATCH_RULE(at::cholesky_inverse)
// INCORRECT USAGE: VARIADIC_BDIMS_BATCH_RULE(&at::cholesky_inverse)
// It is important that this macro is not passed a function pointer!!
#define VARIADIC_BDIMS_BATCH_RULE(fn) SINGLE_ARG(\
    VariadicBdimsBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)


template <typename A, A a, typename C>
struct ExistingBdimBatchRuleHelper;

template <typename F, F Func, typename A, typename... T>
struct ExistingBdimBatchRuleHelper<F, Func, typelist<A, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& self,
      optional<int64_t> self_bdim,
      T... extra_args) {
    auto self_ = reshape_dim_into(*self_bdim, 0, self);
    auto out = Func(self_, std::forward<T>(extra_args)...);
    return std::make_tuple(reshape_dim_outof(0, self.sizes()[*self_bdim], out), 0);
  }
};

// USAGE: EXISTING_BDIM_BATCH_RULE(at::cholesky_inverse)
// INCORRECT USAGE: EXISTING_BDIM_BATCH_RULE(&at::cholesky_inverse)
// It is important that this macro is not passed a function pointer!!
#define EXISTING_BDIM_BATCH_RULE(fn) SINGLE_ARG(\
    ExistingBdimBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)


#define INVOKE(object,ptrToMember)  ((object).*(ptrToMember))
#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));


template <typename F, F Method, typename... ExtraArgs>
Tensor& unary_inplace_batch_rule(Tensor& self, optional<int64_t>, ExtraArgs... extra_args) {
  INVOKE(self, Method)(std::forward<ExtraArgs>(extra_args)...);
  return self;
}

}}

