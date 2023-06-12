// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <c10/util/TypeList.h>

#include <ATen/ATen.h>
#include <ATen/Operators.h>

#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/BatchingMetaprogramming.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/VmapGeneratedPlumbing.h>

#include <utility>

// This file contains helper functions for batching rules.

namespace at { namespace functorch {

TORCH_API Tensor reshape_dim_into(int64_t src, int64_t dst, const Tensor& x);
TORCH_API Tensor reshape_dim_outof(int64_t src, int64_t size1, const Tensor& x);

TORCH_API Tensor reshape_dim_outof_symint(int64_t src, c10::SymInt size1, const Tensor& x);

Tensor moveBatchDimToFront(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
int64_t rankWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
int64_t numelWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim);
optional<int64_t> valIfNonempty(optional<int64_t> maybe_empty, int64_t new_val);
int64_t getPhysicalDim(const Tensor& tensor, bool has_batch_dim, int64_t logical_dim);
VmapDimVector getPhysicalDims(const Tensor& tensor, bool has_batch_dim, IntArrayRef logical_dims);

void vmapIncompatibleInplaceError(const char* schema_name);

Tensor maybePadToLogicalRank(const Tensor& tensor, optional<int64_t> has_bdim, int64_t logical_rank);

void check_randomness(RandomnessType randomness);
void check_randomness(RandomnessType randomness, bool any_tensor_bdim);

inline Tensor ensure_has_bdim(const Tensor& tensor, bool has_bdim, c10::SymInt batch_size) {
  if (has_bdim) {
    return tensor;
  }
  const auto sizes = tensor.sym_sizes();
  SymDimVector expanded_shape;
  expanded_shape.reserve(sizes.size());
  expanded_shape.emplace_back(std::move(batch_size));
  expanded_shape.insert(expanded_shape.end(), sizes.begin(), sizes.end());
  return tensor.expand_symint(expanded_shape);
}

#define VMAP_SUPPORT(op, batch_rule) \
  m.impl(#op, op ## _generated_plumbing<decltype(&batch_rule), &batch_rule>);

#define VMAP_SUPPORT2(op, overload, batch_rule) \
  m.impl(#op "." #overload, op ## _ ## overload ## _generated_plumbing<decltype(&batch_rule), &batch_rule>);

#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

// DO NOT USE ME DIRECTLY! Use BASIC_UNARY_BATCH_RULE to save yourself some pain
template <typename A, A a, typename C>
struct BasicUnaryBatchRuleHelper;

template <typename F, F Func, typename A, typename... T>
struct BasicUnaryBatchRuleHelper<F, Func, c10::guts::typelist::typelist<A, T...>> {
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
  VMAP_SUPPORT(op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));

template <typename A, A a, typename C>
struct VariadicBdimsBatchRuleHelper;

template <typename F, F Func, typename A, typename... T>
struct VariadicBdimsBatchRuleHelper<F, Func, c10::guts::typelist::typelist<A, T...>> {
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
  VMAP_SUPPORT(op, VARIADIC_BDIMS_BATCH_RULE(ATEN_FN(op)));

#define VARIADIC_BDIMS2(op, overload) \
  VMAP_SUPPORT2(op, overload, VARIADIC_BDIMS_BATCH_RULE(ATEN_FN2(op, overload)));

template<class F, F Func>
void boxed_tensor_inputs_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();

  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "boxed_tensor_inputs_batch_rule");

  int64_t cur_level = maybe_layer->layerId();

  auto orig_arguments = torch::jit::last(*stack, num_arguments);
  if (std::none_of(orig_arguments.begin(), orig_arguments.end(), ivalueParticipatesInCurrentLevel)) {
    op.callBoxed(stack);
    return;
  }

  auto arguments = torch::jit::pop(*stack, num_arguments);
  std::vector<std::pair<Tensor, optional<int64_t>>> tensor_inputs;
  std::vector<int64_t> tensor_pos;
  for (const auto idx : c10::irange(0, num_arguments)) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      Tensor tensor_value;
      optional<int64_t> tensor_bdim;
      std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(ivalue.toTensor(), cur_level);
      tensor_inputs.emplace_back(tensor_value, tensor_bdim);
      tensor_pos.push_back(idx);
    }
  }
  Func(tensor_inputs);

  size_t tensor_idx = 0;
  TORCH_INTERNAL_ASSERT(!tensor_pos.empty());
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

inline void handle_pointwise_ops(std::vector<std::pair<Tensor, optional<int64_t>>> &tensor_inputs) {
  int64_t out_logical_rank = 0;
  for (auto& tensor_input : tensor_inputs) {
    int64_t cur_logical_rank = rankWithoutBatchDim(tensor_input.first, tensor_input.second);
    out_logical_rank = std::max(out_logical_rank, cur_logical_rank);
  }
  for (auto& tensor_input: tensor_inputs) {
    tensor_input.first = moveBatchDimToFront(tensor_input.first, tensor_input.second);
    tensor_input.first = maybePadToLogicalRank(tensor_input.first, tensor_input.second, out_logical_rank);
  }
}

#define POINTWISE_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_tensor_inputs_batch_rule<decltype(&handle_pointwise_ops), &handle_pointwise_ops>>());

#define POINTWISE_BOXED2(op, overload) \
  m.impl(#op "." #overload, torch::CppFunction::makeFromBoxedFunction<boxed_tensor_inputs_batch_rule<decltype(&handle_pointwise_ops), &handle_pointwise_ops>>());

inline void handle_variadic_bdims(std::vector<std::pair<Tensor, optional<int64_t>>> &tensor_inputs) {
  for (auto & tensor_input : tensor_inputs) {
    tensor_input.first = moveBatchDimToFront(tensor_input.first, tensor_input.second);
  }
}

#define VARIADIC_BDIMS_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_tensor_inputs_batch_rule<decltype(&handle_variadic_bdims), &handle_variadic_bdims>>());

using UnpackedBatchedTensor = std::tuple<Tensor,optional<int64_t>>;

inline void find_and_unpack_tensors(
    const torch::jit::Stack* stack,
    int64_t num_args,
    int64_t cur_level,
    SmallVector<UnpackedBatchedTensor, 5>* tensors,
    SmallVector<int64_t, 5>* tensors_pos,
    int64_t* batch_size) {

  int64_t computed_batch_size = -1;
  int64_t args_begin = stack->size() - num_args;

  for (const auto idx : c10::irange(0, num_args)) {
    const auto& ivalue = (*stack)[args_begin + idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    auto unpacked = unwrapTensorAtLevel(ivalue.toTensor(), cur_level);
    const auto& tensor_value = std::get<0>(unpacked);
    const auto tensor_bdim = std::get<1>(unpacked);
    if (tensor_bdim.has_value()) {
      auto candidate_batch_size = tensor_value.size(*tensor_bdim);
      if (computed_batch_size == -1) {
        computed_batch_size = candidate_batch_size;
      }
      TORCH_INTERNAL_ASSERT(candidate_batch_size == computed_batch_size);
    }

    tensors->push_back(std::move(unpacked));
    tensors_pos->push_back(idx);
  }
  TORCH_INTERNAL_ASSERT(computed_batch_size > -1);
  *batch_size = computed_batch_size;
}

inline void boxed_existing_bdim_all_batch_rule(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();

  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "boxed_existing_bdim_all_batch_rule");
  int64_t cur_level = maybe_layer->layerId();

  const auto arguments = torch::jit::last(stack, num_arguments);
  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    op.callBoxed(stack);
    return;
  }

  int64_t args_begin = stack->size() - num_arguments;
  SmallVector<UnpackedBatchedTensor, 5> tensor_inputs;
  SmallVector<int64_t, 5> tensor_pos;
  int64_t batch_size;

  find_and_unpack_tensors(
      stack, num_arguments, cur_level,
      &tensor_inputs, &tensor_pos, &batch_size);

  // for each tensor, ensure it has a bdim and reshape it.
  for (const auto tensor_idx : c10::irange(0, tensor_inputs.size())) {
    const auto& value = std::get<0>(tensor_inputs[tensor_idx]);
    auto bdim = std::get<1>(tensor_inputs[tensor_idx]);
    auto value_ = ensure_has_bdim(value, bdim.has_value(), batch_size);
    if (!bdim.has_value()) {
      bdim = 0;
    }
    (*stack)[args_begin + tensor_pos[tensor_idx]] = reshape_dim_into(*bdim, 0, value_);
  }

  op.callBoxed(stack);

  for (const auto idx : c10::irange(args_begin, args_begin + num_returns)) {
    const auto& ret = (*stack)[idx];
    TORCH_INTERNAL_ASSERT(ret.isTensor(),
        "This boxed batching rule does not currently support ops that return non-tensor values");
    (*stack)[idx] = makeBatched(reshape_dim_outof(0, batch_size, ret.toTensor()), 0, cur_level);
  }
}

// Use when all tensors arguments accept one (normal) batch dim.
// This batching rule expands the batch dim on all Tensors, reshapes it into
// dim 0, calls the op, and then reshapes the batch dim out of dim 0.
// This is not the most efficient thing; if there are alternatives, plese try
// to use them. Use this only as a last resort.
#define EXISTING_BDIM_ALL_BOXED(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_existing_bdim_all_batch_rule>());

template <int64_t feature_rank, int64_t contig_tensor_index=-1>
inline void boxed_all_tensors_have_optional_bdim(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();

  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "boxed_all_tensors_have_optional_bdim");
  int64_t cur_level = maybe_layer->layerId();

  const auto arguments = torch::jit::last(stack, num_arguments);
  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    op.callBoxed(stack);
    return;
  }

  int64_t args_begin = stack->size() - num_arguments;
  SmallVector<UnpackedBatchedTensor, 5> tensor_inputs;
  SmallVector<int64_t, 5> tensor_pos;
  int64_t batch_size;

  find_and_unpack_tensors(
      stack, num_arguments, cur_level,
      &tensor_inputs, &tensor_pos, &batch_size);

  optional<bool> is_no_batch_dim_case;

  for (const auto tensor_idx : c10::irange(0, tensor_inputs.size())) {
    const auto& value = std::get<0>(tensor_inputs[tensor_idx]);
    auto bdim = std::get<1>(tensor_inputs[tensor_idx]);
    const auto logical_rank = rankWithoutBatchDim(value, bdim);

    if (!is_no_batch_dim_case.has_value()) {
      is_no_batch_dim_case = (logical_rank == feature_rank);
    }
    auto value_ = ensure_has_bdim(value, bdim.has_value(), batch_size);
    if (!bdim.has_value()) {
      bdim = 0;
    }
    if (*is_no_batch_dim_case) {
      TORCH_INTERNAL_ASSERT(logical_rank == feature_rank);
      value_ = moveBatchDimToFront(value_, bdim);
      if (tensor_idx == contig_tensor_index) {
        value_ = value_.contiguous();
      }
      (*stack)[args_begin + tensor_pos[tensor_idx]] = std::move(value_);
      continue;
    }
    TORCH_INTERNAL_ASSERT(logical_rank == feature_rank + 1);
    value_ = reshape_dim_into(*bdim, 0, value_);
    if (tensor_idx == contig_tensor_index) {
      value_ = value_.contiguous();
    }
    (*stack)[args_begin + tensor_pos[tensor_idx]] = std::move(value_);
  }

  op.callBoxed(stack);

  for (const auto idx : c10::irange(args_begin, args_begin + num_returns)) {
    const auto& ret = (*stack)[idx];
    TORCH_INTERNAL_ASSERT(ret.isTensor(),
        "This boxed batching rule does not currently support ops that return non-tensor values");
    if (*is_no_batch_dim_case) {
      (*stack)[idx] = makeBatched(ret.toTensor(), 0, cur_level);
    } else {
      (*stack)[idx] = makeBatched(reshape_dim_outof(0, batch_size, ret.toTensor()), 0, cur_level);
    }
  }
}

// Useful for many NN operators.
// The operator must satisfy the following:
// - All arguments must accept an optional batch dim.
// - All arguments must be the same rank
#define ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED(feature_rank, op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<boxed_all_tensors_have_optional_bdim<feature_rank>>());

#define ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(feature_rank, op, contig_tensor_index) \
  m.impl(#op, \
         torch::CppFunction::makeFromBoxedFunction<\
             boxed_all_tensors_have_optional_bdim<\
                 feature_rank, \
                 contig_tensor_index>\
             >());

template <typename A, A a, typename C>
struct ExistingBdimBatchRuleHelper;

template <typename F, F Func, typename A, typename... T>
struct ExistingBdimBatchRuleHelper<F, Func, c10::guts::typelist::typelist<A, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& self,
      optional<int64_t> self_bdim,
      T... extra_args) {
    auto self_ = reshape_dim_into(*self_bdim, 0, self);
    auto out = Func(self_, std::forward<T>(extra_args)...);
    return std::make_tuple(reshape_dim_outof_symint(0, self.sym_sizes()[*self_bdim], out), 0);
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
  VMAP_SUPPORT(op, EXISTING_BDIM_BATCH_RULE(ATEN_FN(op)));

#define EXISTING_BDIM2(op, overload) \
  VMAP_SUPPORT2(op, overload, EXISTING_BDIM_BATCH_RULE(ATEN_FN2(op, overload)));

#define INVOKE(object,ptrToMember)  ((object).*(ptrToMember))


template <typename F, F Method, typename... ExtraArgs>
Tensor& unary_inplace_batch_rule(Tensor& self, optional<int64_t>, ExtraArgs... extra_args) {
  INVOKE(self, Method)(std::forward<ExtraArgs>(extra_args)...);
  return self;
}

inline int64_t get_bdim_size4(
    const Tensor& a_value, optional<int64_t> a_bdim,
    const Tensor& b_value, optional<int64_t> b_bdim,
    const Tensor& c_value, optional<int64_t> c_bdim,
    const Tensor& d_value, optional<int64_t> d_bdim) {
  if (a_bdim)
    return a_value.size(*a_bdim);
  if (b_bdim)
    return b_value.size(*b_bdim);
  if (c_bdim)
    return c_value.size(*c_bdim);
  if (d_bdim)
    return d_value.size(*d_bdim);
  TORCH_INTERNAL_ASSERT(false);
}

inline int64_t get_bdim_size3(
    const Tensor& a_value, optional<int64_t> a_bdim,
    const Tensor& b_value, optional<int64_t> b_bdim,
    const Tensor& c_value, optional<int64_t> c_bdim) {
  if (a_bdim)
    return a_value.size(*a_bdim);
  if (b_bdim)
    return b_value.size(*b_bdim);
  if (c_bdim)
    return c_value.size(*c_bdim);
  TORCH_INTERNAL_ASSERT(false);
}

inline int64_t get_bdim_size2(
    const Tensor& a_value, optional<int64_t> a_bdim,
    const Tensor& b_value, optional<int64_t> b_bdim) {
  if (a_bdim)
    return a_value.size(*a_bdim);
  if (b_bdim)
    return b_value.size(*b_bdim);
  TORCH_INTERNAL_ASSERT(false);
}

// [start, start + 1, ..., stop - 1]
inline VmapDimVector range(int64_t start, int64_t stop) {
  TORCH_INTERNAL_ASSERT(stop >= start);
  VmapDimVector dims;
  dims.reserve(stop - start);
  for (int64_t i = start; i < stop; i++) {
    dims.emplace_back(i);
  }
  return dims;
}
std::tuple<Tensor, Tensor> _binary_pointwise_helper(
    const Tensor& tensor, optional<int64_t> tensor_batch_dim, const Tensor& other, optional<int64_t> other_batch_dim,
    bool do_type_promotion=true);

}}
