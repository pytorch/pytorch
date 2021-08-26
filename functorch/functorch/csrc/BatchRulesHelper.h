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
#include <functorch/csrc/PlumbingHelper.h>
#include <ATen/core/dispatch/Dispatcher.h>
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

// USAGE: BASIC_UNARY_BATCH_RULE(at::sin)
// INCORRECT USAGE: BASIC_UNARY_BATCH_RULE(&at::sin)
// It is important that this macro is not passed a function pointer!!
#define BASIC_UNARY_BATCH_RULE(fn) SINGLE_ARG(\
    BasicUnaryBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

#define UNARY_POINTWISE(op) \
  VMAP_SUPPORT(#op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));

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

#define VARIADIC_BDIMS(op) \
  VMAP_SUPPORT(#op, VARIADIC_BDIMS_BATCH_RULE(ATEN_FN(op)));

#define VARIADIC_BDIMS2(op, overload) \
  VMAP_SUPPORT(#op"."#overload, VARIADIC_BDIMS_BATCH_RULE(ATEN_FN2(op, overload)));

template<class F, F Func>
void boxed_tensor_inputs_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();
  auto arguments = torch::jit::pop(*stack, num_arguments);

  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();


  std::vector<std::pair<Tensor, optional<int64_t>>> tensor_inputs;
  std::vector<int64_t> tensor_pos;
  for (const auto idx : c10::irange(0, num_arguments)) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      Tensor tensor_value;
      optional<int64_t> tensor_bdim;
      std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(ivalue.toTensor(), cur_level);
      tensor_inputs.push_back(std::make_pair(tensor_value, tensor_bdim));
      tensor_pos.push_back(idx);
    }
  }
  Func(tensor_inputs);

  size_t tensor_idx = 0;
  TORCH_INTERNAL_ASSERT(tensor_pos.size() > 0);
  for (const auto arg_idx : c10::irange(0, num_arguments)) {
    if (tensor_idx >= tensor_pos.size() || (int64_t)arg_idx != tensor_pos[tensor_idx]) {
      torch::jit::push(stack, arguments[arg_idx]);
    } else {
      TORCH_INTERNAL_ASSERT(tensor_idx < tensor_inputs.size());
      torch::jit::push(stack, tensor_inputs[tensor_idx].first);
      tensor_idx++;
    }
  }

  op.callBoxed(stack);
  const auto returns = torch::jit::pop(*stack, num_returns);
  for (const auto& ret : returns) {
    if (ret.isTensor()) {
      torch::jit::push(stack, makeBatched(ret.toTensor(), 0, cur_level));
    } else {
      TORCH_INTERNAL_ASSERT(false, "This boxed batching rule does not currently support ops that return non-tensor values");
    }
  }
}

inline void handle_variadic_bdims(std::vector<std::pair<Tensor, optional<int64_t>>> &tensor_inputs) {
  for (auto & tensor_input : tensor_inputs) {
    tensor_input.first = moveBatchDimToFront(tensor_input.first, tensor_input.second);
  }
}

#define VARIADIC_BDIMS_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_tensor_inputs_batch_rule<decltype(&handle_variadic_bdims), &handle_variadic_bdims>>());

inline void boxed_existing_bdim_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();
  auto arguments = torch::jit::pop(*stack, num_arguments);

  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();


  std::vector<std::pair<Tensor, optional<int64_t>>> tensor_inputs;
  std::vector<int64_t> tensor_pos;
  for (const auto idx : c10::irange(0, num_arguments)) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      Tensor tensor_value;
      optional<int64_t> tensor_bdim;
      std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(ivalue.toTensor(), cur_level);
      tensor_inputs.push_back(std::make_pair(tensor_value, tensor_bdim));
      tensor_pos.push_back(idx);
    }
  }
  int64_t batch_size = -1;
  for (auto& tensor_input : tensor_inputs) {
    if (tensor_input.second) {
      if (batch_size == -1) {
        batch_size = tensor_input.first.size(*tensor_input.second);
      }
      TORCH_INTERNAL_ASSERT(batch_size == tensor_input.first.size(*tensor_input.second));
      tensor_input.first = reshape_dim_into(*tensor_input.second, 0, tensor_input.first);
    }
  }

  size_t tensor_idx = 0;
  TORCH_INTERNAL_ASSERT(tensor_pos.size() > 0);
  for (const auto arg_idx : c10::irange(0, num_arguments)) {
    if (tensor_idx >= tensor_pos.size() || (int64_t)arg_idx != tensor_pos[tensor_idx]) {
      torch::jit::push(stack, arguments[arg_idx]);
    } else {
      TORCH_INTERNAL_ASSERT(tensor_idx < tensor_inputs.size());
      torch::jit::push(stack, tensor_inputs[tensor_idx].first);
      tensor_idx++;
    }
  }

  op.callBoxed(stack);
  const auto returns = torch::jit::pop(*stack, num_returns);
  for (const auto& ret : returns) {
    if (ret.isTensor()) {
      torch::jit::push(stack, makeBatched(reshape_dim_outof(0, batch_size, ret.toTensor()), 0, cur_level));
    } else {
      TORCH_INTERNAL_ASSERT(false, "This boxed batching rule does not currently support ops that return non-tensor values");
    }
  }
}

#define EXISTING_BDIM_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_existing_bdim_batch_rule>());


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


#define EXISTING_BDIM(op) \
  VMAP_SUPPORT(#op, EXISTING_BDIM_BATCH_RULE(ATEN_FN(op)));

#define EXISTING_BDIM2(op, overload) \
  VMAP_SUPPORT(#op"."#overload, EXISTING_BDIM_BATCH_RULE(ATEN_FN2(op, overload)));

#define INVOKE(object,ptrToMember)  ((object).*(ptrToMember))


template <typename F, F Method, typename... ExtraArgs>
Tensor& unary_inplace_batch_rule(Tensor& self, optional<int64_t>, ExtraArgs... extra_args) {
  INVOKE(self, Method)(std::forward<ExtraArgs>(extra_args)...);
  return self;
}

}}

