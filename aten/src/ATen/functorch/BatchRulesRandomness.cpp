// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/BatchRulesHelper.h>

#include <utility>

// This file contains batching rules for random operations. These are different
// from our regular batching rules: regular batching rules get registered to the
// FuncTorchBatched key, but batching rules for random operations get
// registered to FuncTorchVmapMode. This is because we need to interpose on
// random operations even if they're not on a BatchedTensor.

namespace at::functorch {

template <typename F, F Func, typename... ExtraArgs>
Tensor random_batching_rule(SymIntArrayRef shape, ExtraArgs... extra_args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  c10::SmallVector<SymInt> shapeVec(1, maybe_layer->batchSize());
  shapeVec.reserve(shape.size() + 1);
  shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());
  RandomnessType randomness = maybe_layer->randomness();
  check_randomness(randomness);
  if (randomness == RandomnessType::Different) {
    return makeBatched(Func(shapeVec, std::forward<ExtraArgs>(extra_args)...), 0, maybe_layer->layerId());
  } else {
    return Func(shape, std::forward<ExtraArgs>(extra_args)...);
  }
}

template <typename F, F Func, typename... ExtraArgs>
Tensor& random_inplace_batching_rule(Tensor& self, ExtraArgs... extra_args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  const auto cur_level = maybe_layer->layerId();
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  self_value = moveBatchDimToFront(self_value, self_bdim);
  RandomnessType randomness = maybe_layer->randomness();
  check_randomness(randomness);
  TORCH_CHECK(
    !(randomness == RandomnessType::Different && !self_bdim),
    "vmap: Cannot ask for different inplace randomness on an unbatched tensor. This will appear like same randomness. ",
    "If this is necessary for your usage, please file an issue with functorch.");
  if (randomness == RandomnessType::Same && self_bdim) {
    auto intermediate = empty(self.sizes(), self.options());
    Func(intermediate, std::forward<ExtraArgs>(extra_args)...);
    self.copy_(intermediate); // batching should make this just work out...
    return self;
  } else {
    Func(self_value, std::forward<ExtraArgs>(extra_args)...);
    return self;
  }
}

static Tensor& bernoulli_inplace_Tensor_batching_rule(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  auto cur_level = maybe_layer->layerId();
  RandomnessType randomness = maybe_layer->randomness();

  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);

  auto [other_value, other_bdim] = unwrapTensorAtLevel(p_, cur_level);

  check_randomness(randomness, other_bdim.has_value());

  if (!self_bdim && other_bdim) {
    vmapIncompatibleInplaceError("inplace bernoulli");
  }

  // compute max logical rank
  auto self_logical_rank = rankWithoutBatchDim(self_value, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other_value, other_bdim);
  auto max_logical_rank = std::max(self_logical_rank, other_logical_rank);

  auto self_ = moveBatchDimToFront(self_value, self_bdim);
  auto other_ = moveBatchDimToFront(other_value, other_bdim);

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_bdim, max_logical_rank);
  TORCH_CHECK(
    !(randomness == RandomnessType::Different && !self_bdim),
    "vmap: Cannot ask for different inplace randomness on an unbatched tensor. This will appear like same randomness. ",
    "If this is necessary for your usage, please file an issue with functorch.");
  if (randomness == RandomnessType::Same && self_bdim) {
    auto intermediate = empty(self.sizes(), self.options());
    intermediate.bernoulli_(other_, std::move(gen));
    self.copy_(intermediate); // batching should make this just work out...
    return self;
  } else {
    self_.bernoulli_(other_, std::move(gen));
    return self;
  }
}

template <typename F, F Func, typename... ExtraArgs>
Tensor randperm_batching_rule(int64_t n, ExtraArgs... extra_args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  auto const batch_size = maybe_layer->batchSize();
  RandomnessType randomness = maybe_layer->randomness();
  check_randomness(randomness);
  if (randomness == RandomnessType::Different) {
    std::vector<at::Tensor> stackedList(batch_size.guard_int(__FILE__, __LINE__));
    for (int64_t idx = 0; idx < batch_size; ++idx) {
      // since this is done in a loop, need to pass by reference for generator to update
      stackedList[idx] = Func(n, extra_args...);
    }
    return makeBatched(at::stack(stackedList), 0, maybe_layer->layerId());
  } else {
    return Func(n, std::forward<ExtraArgs>(extra_args)...);
  }
}

template <typename F, F Func, typename... ExtraArgs>
Tensor unary_pointwise_random_batch_rule(const Tensor& tensor, ExtraArgs... extra_args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  const auto cur_level = maybe_layer->layerId();

  auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(tensor, cur_level);
  tensor_value = moveBatchDimToFront(tensor_value, tensor_bdim);

  RandomnessType randomness = maybe_layer->randomness();
  check_randomness(randomness, tensor_bdim.has_value());
  auto shape = tensor_value.sizes();
  VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
  shapeVec.reserve(shape.size() + 1);
  shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());

  if (randomness == RandomnessType::Different && !tensor_bdim) {
    tensor_value = tensor_value.expand_symint(shapeVec);
  }
  auto out = Func(tensor_value, std::forward<ExtraArgs>(extra_args)...);
  if (randomness == RandomnessType::Same && !tensor_bdim) {
    return out;
  }
  return makeBatched(out, 0, cur_level);
}

template<typename F, F Func, typename... ExtraArgs>
Tensor tensor_like_random_batch_rule(const Tensor& self, ExtraArgs... extra_args) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  const auto cur_level = maybe_layer->layerId();
  RandomnessType randomness = maybe_layer->randomness();
  check_randomness(randomness);

  auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(self, cur_level);
  tensor_value = moveBatchDimToFront(tensor_value, tensor_bdim);

  if (randomness == RandomnessType::Same && tensor_bdim) {
    tensor_value = tensor_value[0];
  } else if (randomness == RandomnessType::Different && !tensor_bdim) {
    auto shape = tensor_value.sizes();
    VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
    shapeVec.reserve(shape.size() + 1);
    shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());
    tensor_value = tensor_value.expand_symint(shapeVec);
  }

  auto res = Func(tensor_value, std::forward<ExtraArgs>(extra_args)...);
  return (randomness == RandomnessType::Same) ? res : makeBatched(res, 0, cur_level);
}

static std::tuple<Tensor,Tensor> native_dropout_batching_rule(const Tensor& tensor, double p, std::optional<bool> train) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  const auto cur_level = maybe_layer->layerId();
  RandomnessType randomness = maybe_layer->randomness();

  auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(tensor, cur_level);
  tensor_value = moveBatchDimToFront(tensor_value, tensor_bdim);

  if (!train.has_value() || *train) {
    check_randomness(randomness); // if we are in eval mode, we don't use about randomness
  }

  if ((train.has_value() && !*train) ||
      randomness == RandomnessType::Different) {
    if (!tensor_bdim) {
      // if tensor is unbatched, add batch dim before
      // calling dropout.
      auto shape = tensor_value.sizes();
      VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
      shapeVec.reserve(shape.size() + 1);
      shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());
      tensor_value = tensor_value.expand_symint(shapeVec);
    }
    auto [output, mask] = at::native_dropout(tensor_value, p, train);
    return std::make_tuple(
        makeBatched(output, 0, cur_level),
        makeBatched(mask, 0, cur_level));
  }

  // repeated code from the CPU kernel since the CUDA one doesn't call bernoulli_ explicitly
  double p1m = 1. - p;
  // Check for probability of zero to avoid divide by zero and NaN results
  double scale = p1m == 0 ? 0. : 1. / p1m;
  Tensor mask = at::empty_like(tensor, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  mask.bernoulli_(p1m);
  const auto output = tensor.mul(mask).mul_(scale);
  return std::make_tuple(output, mask);
}

static Tensor multinomial_batching_rule(const Tensor& self, const int64_t num_samples, const bool replacement, const std::optional<Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
  auto maybe_layer = maybeCurrentDynamicLayer();
  const auto cur_level = maybe_layer->layerId();

  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  self_value = moveBatchDimToFront(self_value, self_bdim);

  RandomnessType randomness = maybe_layer->randomness();
  check_randomness(randomness, self_bdim.has_value());

  if (randomness == RandomnessType::Different) {
    // 1D cases: S -> BS -> multinomial(BS)
    //           BS -> multinomial(BS)
    //
    // 2D cases: MS -> BMS -> (BM)S -> multinomial((BM)S) -> (BM)S -> BMS
    //           BMS -> (BM)S -> multinomial((BM)S) -> (BM)S -> BMS
    const auto is_2D_case = rankWithoutBatchDim(self_value, self_bdim) == 2;
    if (!self_bdim.has_value()) {
      self_value = ensure_has_bdim(self_value, self_bdim.has_value(), maybe_layer->batchSize());
    }
    if (is_2D_case) {
      self_value = reshape_dim_into(0, 0, self_value);
    }
    auto out = multinomial(self_value, num_samples, replacement, generator);
    if (is_2D_case) {
      out = reshape_dim_outof_symint(0, maybe_layer->batchSize(), out);
    }
    return makeBatched(out, 0, cur_level);;
  }

  TORCH_INTERNAL_ASSERT(randomness == RandomnessType::Same); // check_randomness eliminates error randomness
  TORCH_INTERNAL_ASSERT(!self_bdim.has_value()); // check_randomness eliminates same randomness with batched input
  // Must be same randomness with unbatched input
  // 1D case: S -> multinomial(S) -> S
  // 2D case: MS -> multinomial(MS) -> MS
  return multinomial(self_value, num_samples, replacement, generator);
}

template <typename A, A a, typename C>
struct RandomBatchRuleHelper;

template <typename F, F Func, typename T1, typename... T>
struct RandomBatchRuleHelper<F, Func, typelist<T1, T...>> {
  static Tensor apply(SymIntArrayRef shape, T... extra_args) {
    return random_batching_rule<F, Func, T...>(shape, std::forward<T>(extra_args)...);
  }
};

template <typename F, F Func, typename... T>
Tensor rand_int_wrapper(SymIntArrayRef shape, c10::SymInt high, T... extra_args) {
  return Func(high, shape, std::forward<T>(extra_args)...);
}

template <typename A, A a, typename C>
struct RandomInplaceBatchRuleHelper;

template <typename F, F Func, typename T1, typename... T>
struct RandomInplaceBatchRuleHelper<F, Func, typelist<T1, T...>> {
  static Tensor& apply(Tensor& self, T... extra_args) {
    return random_inplace_batching_rule<F, Func, T...>(self, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct RandIntBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename... T>
struct RandIntBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  static Tensor apply(c10::SymInt high, SymIntArrayRef shape, T... extra_args) {
    return random_batching_rule<decltype(&rand_int_wrapper<F, Func, T...>),
                                &rand_int_wrapper<F, Func, T...>,
                                c10::SymInt, T...>(shape, std::move(high), std::forward<T>(extra_args)...);
  }
};

template <typename F, F Func, typename T0, typename T1, typename... T>
Tensor rand_int_low_wrapper(SymIntArrayRef shape, T0 scalar0, T1 scalar1, T... extra_args) {
  return Func(scalar0, scalar1, shape, std::forward<T>(extra_args)...);
}

template <typename A, A a, typename C>
struct RandTwoLeadingScalarsBatchRuleHelper;

template <typename F, F Func, typename T0, typename T1, typename T2, typename... T>
struct RandTwoLeadingScalarsBatchRuleHelper<F, Func, typelist<T0, T1, T2, T...>> {
  static Tensor apply(T0 scalar0, T1 scalar1, SymIntArrayRef shape, T... extra_args) {
    return random_batching_rule<decltype(&rand_int_low_wrapper<F, Func, T0, T1, T...>),
                                &rand_int_low_wrapper<F, Func, T0, T1, T...>,
                                T0, T1, T...>(shape, scalar0, scalar1, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct RandpermBatchRuleHelper;

template <typename F, F Func, typename T1, typename... T>
struct RandpermBatchRuleHelper<F, Func, typelist<T1, T...>> {
  static Tensor apply(int64_t n, T... extra_args) {
    return randperm_batching_rule<F, Func, T...>(n, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct UnaryPointwiseRandomBatchRule;

template <typename F, F Func, typename A0, typename... T>
struct UnaryPointwiseRandomBatchRule<F, Func, typelist<A0, T...>> {
  static Tensor apply(const Tensor& tensor, T... extra_args) {
    return unary_pointwise_random_batch_rule<F, Func, T...>(tensor, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct NormalPointwiseBatchRule;

template <typename F, F Func, typename A0, typename... T>
struct NormalPointwiseBatchRule<F, Func, typelist<A0, T...>> {
  static Tensor apply(const Tensor& tensor, T... extra_args) {
    return unary_pointwise_random_batch_rule<F, Func, T...>(tensor, std::forward<T>(extra_args)...);
  }
};

template<typename F, F Func, typename... T>
Tensor normal_wrapper(const Tensor& tensor, double scalar, T... extra_args) {
  return Func(scalar, tensor, extra_args...);
}

template <typename A, A a, typename C>
struct UnaryPointwiseRandomLeadingFloatBatchRule;

template <typename F, F Func, typename A0, typename A1, typename... T>
struct UnaryPointwiseRandomLeadingFloatBatchRule<F, Func, typelist<A0, A1, T...>> {
  static Tensor apply(double scalar, const Tensor& tensor, T... extra_args) {
    return unary_pointwise_random_batch_rule<decltype(&normal_wrapper<F, Func, T...>),
                                         &normal_wrapper<F, Func, T...>, double,
                                         T...>(tensor, scalar, std::forward<T>(extra_args)...);
  }
};

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  #define RANDOM_INPLACE_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  RANDOM_INPLACE_BATCH_RULE2(bernoulli_, float);

  #undef RANDOM_INPLACE_BATCH_RULE2
}

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  #define RANDOM_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandomBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDOM_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define RANDOM_INPLACE_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDOM_INPLACE_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandomInplaceBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define RANDINT_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandIntBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                             c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDINT_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandIntBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define RAND_TWO_LEADING_SCALARS_BATCH_RULE(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandTwoLeadingScalarsBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                                c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))
  #define RANDPERM_BATCH_RULE(op) \
    m.impl(#op, SINGLE_ARG(\
      RandpermBatchRuleHelper<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                            c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define RANDPERM_BATCH_RULE2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      RandpermBatchRuleHelper<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                            c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define UNARY_POINTWISE_RANDOM(op) \
    m.impl(#op, SINGLE_ARG(\
      UnaryPointwiseRandomBatchRule<decltype(&ATEN_FN(op)), &ATEN_FN(op), \
                                    c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

  #define UNARY_POINTWISE_RANDOM2(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      UnaryPointwiseRandomBatchRule<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                                    c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  #define UNARY_POINTWISE_RANDOM_LEADING_FLOAT(op, overload) \
    m.impl(#op"."#overload, SINGLE_ARG(\
      UnaryPointwiseRandomLeadingFloatBatchRule<decltype(&ATEN_FN2(op, overload)), &ATEN_FN2(op, overload), \
                                                c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

  RANDOM_BATCH_RULE(randn);
  RANDOM_BATCH_RULE2(randn, generator);
  RANDOM_BATCH_RULE2(randn, generator_with_names);
  RANDOM_BATCH_RULE2(randn, names);

  RANDOM_BATCH_RULE(rand);
  RANDOM_BATCH_RULE2(rand, generator);
  RANDOM_BATCH_RULE2(rand, generator_with_names);
  RANDOM_BATCH_RULE2(rand, names);

  RANDOM_INPLACE_BATCH_RULE(random_);
  RANDOM_INPLACE_BATCH_RULE2(random_, from);
  RANDOM_INPLACE_BATCH_RULE2(random_, to);

  RANDOM_INPLACE_BATCH_RULE(cauchy_);
  RANDOM_INPLACE_BATCH_RULE(exponential_);
  RANDOM_INPLACE_BATCH_RULE(geometric_);
  RANDOM_INPLACE_BATCH_RULE(log_normal_);
  RANDOM_INPLACE_BATCH_RULE(normal_);
  RANDOM_INPLACE_BATCH_RULE(uniform_);

  RANDINT_BATCH_RULE(randint);
  RANDINT_BATCH_RULE2(randint, generator);
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(randint, low);
  RAND_TWO_LEADING_SCALARS_BATCH_RULE(randint, low_generator);

  m.impl("bernoulli_.Tensor", at::functorch::bernoulli_inplace_Tensor_batching_rule);
  RANDOM_INPLACE_BATCH_RULE2(bernoulli_, float);
  UNARY_POINTWISE_RANDOM2(bernoulli, p);

  RANDPERM_BATCH_RULE(randperm);
  RANDPERM_BATCH_RULE2(randperm, generator);

  RAND_TWO_LEADING_SCALARS_BATCH_RULE(normal, float_float);
  UNARY_POINTWISE_RANDOM2(normal, Tensor_float);
  UNARY_POINTWISE_RANDOM_LEADING_FLOAT(normal, float_Tensor);

  m.impl("native_dropout", native_dropout_batching_rule); // needs special casing because cuda version doesn't call bernoulli

  UNARY_POINTWISE_RANDOM(_standard_gamma);
  UNARY_POINTWISE_RANDOM(_sample_dirichlet);
  m.impl("multinomial", multinomial_batching_rule);
  UNARY_POINTWISE_RANDOM(poisson);
  UNARY_POINTWISE_RANDOM(bernoulli);

  #define TENSOR_LIKE_COMMON_ARG_TYPES std::optional<ScalarType>, std::optional<Layout>, std::optional<Device>, std::optional<bool>, std::optional<MemoryFormat>
  m.impl("randint_like", tensor_like_random_batch_rule<decltype(&ATEN_FN(randint_like)), &ATEN_FN(randint_like), int64_t, TENSOR_LIKE_COMMON_ARG_TYPES>);
  m.impl("randint_like.low_dtype", tensor_like_random_batch_rule<\
    decltype(&ATEN_FN2(randint_like, low_dtype)), &ATEN_FN2(randint_like, low_dtype), int64_t, int64_t, TENSOR_LIKE_COMMON_ARG_TYPES>);
  m.impl("rand_like", tensor_like_random_batch_rule<decltype(&ATEN_FN(rand_like)), &ATEN_FN(rand_like), TENSOR_LIKE_COMMON_ARG_TYPES>);
  m.impl("randn_like", tensor_like_random_batch_rule<decltype(&ATEN_FN(randn_like)), &ATEN_FN(randn_like), TENSOR_LIKE_COMMON_ARG_TYPES>);

  #undef RANDOM_BATCH_RULE
  #undef RANDOM_BATCH_RULE2
  #undef RANDOM_INPLACE_BATCH_RULE
  #undef RANDOM_INPLACE_BATCH_RULE2
  #undef RANDINT_BATCH_RULE
  #undef RANDINT_BATCH_RULE2
  #undef RAND_TWO_LEADING_SCALARS_BATCH_RULE
  #undef RANDPERM_BATCH_RULE
  #undef RANDPERM_BATCH_RULE2
  #undef UNARY_POINTWISE_RANDOM
  #undef UNARY_POINTWISE_RANDOM2
  #undef UNARY_POINTWISE_RANDOM_LEADING_FLOAT
  #undef TENSOR_LIKE_COMMON_ARG_TYPES
}

} // namespace at::functorch
