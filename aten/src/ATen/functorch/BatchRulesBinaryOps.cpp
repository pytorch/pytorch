// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <utility>

namespace at::functorch {

template <typename F, F Func, typename... ExtraArgs>
static Tensor _binary_pointwise_batch_rule(
    const Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
    const Tensor& other, std::optional<int64_t> other_batch_dim,
    ExtraArgs... extra_args) {

  auto [tensor_, other_]= _binary_pointwise_helper(
      tensor, tensor_batch_dim, other, other_batch_dim);

  return Func(tensor_, std::move(other_), std::forward<ExtraArgs>(extra_args)...);
}

template <typename A, A a, typename C>
struct BinaryPointwiseBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename... T>
struct BinaryPointwiseBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  static std::tuple<Tensor, std::optional<int64_t>> apply(
      const Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
      const Tensor& other, std::optional<int64_t> other_batch_dim,
      T... extra_args) {
    return std::tuple(_binary_pointwise_batch_rule<F, Func, T...>(
        tensor, tensor_batch_dim, other, other_batch_dim,
        std::forward<T>(extra_args)...), 0);
  }
};

#define BINARY_POINTWISE_BATCH_RULE(fn) SINGLE_ARG(\
    BinaryPointwiseBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

template <typename A, A a, typename C>
struct BinaryRandomPointwiseBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename... T>
struct BinaryRandomPointwiseBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  static Tensor apply(const Tensor& tensor, const Tensor& other, T... extra_args) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
    auto maybe_layer = maybeCurrentDynamicLayer();
    TORCH_INTERNAL_ASSERT(maybe_layer.has_value())
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto cur_level = maybe_layer->layerId();
    RandomnessType randomness = maybe_layer->randomness();

    auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(tensor, cur_level);

    auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);

    check_randomness(randomness, (tensor_bdim || other_bdim));
    if (randomness == RandomnessType::Different && !tensor_bdim && !other_bdim) {
      auto shape = tensor_value.sizes();
      VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
      shapeVec.reserve(shape.size() + 1);
      shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());

      // not taken care of with binary batch rule, which assumes at least one input is batched
      tensor_value = tensor_value.expand_symint(shapeVec);
      tensor_bdim = 0;
    } else if (randomness == RandomnessType::Same && !tensor_bdim && !other_bdim) {

      // avoids unnecessary checks and batch rule assuming output is batched
      return Func(tensor_value, other_value, std::forward<T>(extra_args)...);
    }
    auto res = _binary_pointwise_batch_rule<F, Func, T...>(
      tensor_value, tensor_bdim, other_value, other_bdim,
      std::forward<T>(extra_args)...);
    return makeBatched(std::move(res), 0, cur_level);
  }
};

#define BINARY_RANDOM_POINTWISE_BATCH_RULE(fn) SINGLE_ARG(\
    BinaryRandomPointwiseBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

template <typename M, M Meth, typename... ExtraArgs>
static void binary_pointwise_inplace_batch_rule(
    Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
    const Tensor& other, std::optional<int64_t> other_batch_dim,
    ExtraArgs... extra_args) {
  if (!tensor_batch_dim && other_batch_dim) {
    vmapIncompatibleInplaceError("inplace arithmetic");
  }

  // compute max logical rank
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  (tensor_.*Meth)(other_, std::forward<ExtraArgs>(extra_args)...);
}

template <typename F, F Func>
static std::tuple<Tensor, std::optional<int64_t>> comparison_pointwise_batch_rule(
    const Tensor& tensor, std::optional<int64_t> tensor_batch_dim,
    const Tensor& other, std::optional<int64_t> other_batch_dim) {
  // compute max logical rank
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  auto result = Func(tensor_, other_);
  return std::make_tuple( std::move(result), 0 );
}

static std::tuple<Tensor, std::optional<int64_t>> where_self_batch_rule(
    const Tensor& condition, std::optional<int64_t> condition_bdim,
    const Tensor& self, std::optional<int64_t> self_bdim, const Tensor& other, std::optional<int64_t> other_bdim) {
  auto condition_logical_rank = rankWithoutBatchDim(condition, condition_bdim);
  auto tensor_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  auto max_logical_rank = std::max({tensor_logical_rank, other_logical_rank, condition_logical_rank});

  auto condition_ = moveBatchDimToFront(condition, condition_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(other, other_bdim);

  condition_ = maybePadToLogicalRank(condition_, condition_bdim, max_logical_rank);
  self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_bdim, max_logical_rank);
  return std::make_tuple(at::where(condition_, self_, other_), 0);
}

static std::tuple<Tensor, std::optional<int64_t>> gelu_backward_batch_rule(
    const Tensor& grad_out, std::optional<int64_t> grad_out_bdim, const Tensor& input, std::optional<int64_t> input_bdim,
    std::string_view approximate) {

  // repeat the preprocessing from _binary_pointwise_batch_rule
  auto [grad_out_, input_]= _binary_pointwise_helper(grad_out, grad_out_bdim, input, input_bdim);

  // gelu_backward doesn't broadcast well so we need to insist all inputs have a bdim
  const auto batch_size = get_bdim_size2(grad_out, grad_out_bdim, input, input_bdim);
  grad_out_ = ensure_has_bdim(grad_out_, grad_out_bdim.has_value(), batch_size);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), batch_size);

  return std::make_tuple(at::gelu_backward(grad_out_, input_, approximate), 0);
}

static std::tuple<Tensor, std::optional<int64_t>> masked_select_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim,
    const Tensor& mask, std::optional<int64_t> mask_bdim) {
  TORCH_CHECK(!mask_bdim.has_value(),
      "vmap: Attempted to vmap over `mask` in torch.masked_select(self, mask) ",
      "We cannot support this because for each batch this would return a ",
      "differently shaped Tensor. "
      "Please voice your support in https://github.com/pytorch/functorch/issues/256");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  const auto batch_size = self_.size(0);
  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  const auto max_logical_rank = std::max(self_logical_rank, mask.dim());
  self_ = maybePadToLogicalRank(self_, 0, max_logical_rank);

  // masked_select returns a 1D tensor, so we have to reshape it into 2D
  const auto result = at::masked_select(self_, mask).view({ batch_size, -1 });
  return std::make_tuple(result, 0);
}

static std::tuple<Tensor, std::optional<int64_t>> masked_select_backward_batch_rule(
    const Tensor& grad, std::optional<int64_t> grad_bdim,
    const Tensor& self, std::optional<int64_t> self_bdim,
    const Tensor& mask, std::optional<int64_t> mask_bdim) {
  TORCH_CHECK(!mask_bdim.has_value(),
      "vmap: Attempted to vmap over `mask` in torch.masked_select_backward(grad, self, mask) ",
      "We cannot support this because for each batch this would return a ",
      "differently shaped Tensor. "
      "Please voice your support in https://github.com/pytorch/functorch/issues/256");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto grad_ = moveBatchDimToFront(grad, grad_bdim);

  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  const auto max_logical_rank = std::max(self_logical_rank, mask.dim());

  self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);

  const auto batch_size = get_bdim_size2(grad, grad_bdim, self, self_bdim);
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), batch_size);

  const auto result = at::masked_select_backward(grad_, self_.contiguous(), mask);
  return std::make_tuple(result, 0);
}

static std::tuple<Tensor, std::optional<int64_t>> cdist_backward_batch_rule(
    const Tensor& grad, std::optional<int64_t> grad_bdim,
    Tensor x1, std::optional<int64_t> x1_bdim,
    Tensor x2, std::optional<int64_t> x2_bdim,
    const double p,
    const Tensor& cdist, std::optional<int64_t> cdist_bdim) {

  if (cdist_bdim && !x1_bdim) {
    // We need to make sure that x1 has batch dim if cdist has one
    // otherwise, we get
    // RuntimeError: Function CdistBackward0 returned an invalid gradient at index 1 - got [5]
    // but expected shape compatible with [4, 5]
    auto bs = cdist.size(*cdist_bdim);
    x1 = ensure_has_bdim(x1, false, bs).contiguous();
    x1_bdim = 0;
  }

  // We need to apply the same preprocessing on x1 and x2 as in the forward pass
  // _binary_pointwise_batch_rule
  std::tie(x1, x2)= _binary_pointwise_helper(x1, x1_bdim, x2, x2_bdim);

  auto grad_ = moveBatchDimToFront(grad, grad_bdim);
  if ((x1_bdim || x2_bdim) && !grad_bdim) {
    // We need to make sure that grad has batch dim if x1 or x2 have one
    // Probably, there is an assumption on the strides.
    // Otherwise grad input contains thrash values, e.g. -7.0816e+29, 7.0816e+29
    auto bs = get_bdim_size2(x1, 0, x2, 0);
    grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), bs);
    grad_ = grad_.contiguous();
  }

  auto out = at::_cdist_backward(grad_, x1, x2, p, cdist);

  std::optional<int64_t> out_bdim = std::nullopt;
  if (x1_bdim || x2_bdim) {
    out_bdim = 0;
  }

  return std::make_tuple(std::move(out), out_bdim);
}

static void fill__Tensor_batch_rule(
    Tensor& self,
    std::optional<int64_t> self_bdim,
    const Tensor& other,
    std::optional<int64_t> other_bdim) {
  if (!other_bdim.has_value()) {
    // Optimization: fill_ is faster than the other path which does
    // reshaping + copy_
    self.fill_(other);
    return;
  }
  if (!self_bdim) {
    vmapIncompatibleInplaceError("fill_");
  }
  auto self_and_other = _binary_pointwise_helper(
      self, self_bdim, other, other_bdim, /*do_type_promotion*/false);
  std::get<0>(self_and_other).copy_(std::get<1>(self_and_other));
}

static
std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>>
rrelu_with_noise_batch_rule(
    const Tensor& self,
    std::optional<int64_t> self_bdim,
    Tensor& noise,
    std::optional<int64_t> noise_bdim,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    std::optional<at::Generator> generator) {

  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto noise_ = moveBatchDimToFront(self, noise_bdim);

  auto ret = at::rrelu_with_noise(self_, noise_, lower, upper, training, std::move(generator));

  return std::make_tuple(ret, 0, noise_, 0);
}

static Tensor rrelu_with_noise_batch(
    const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  int64_t cur_level = maybe_layer->layerId();
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [noise_value, noise_bdim] = unwrapTensorAtLevel(noise, cur_level);
  TORCH_CHECK(!noise_bdim.has_value(), "vmap: Attempted to vmap over 'noise' in torch.rrelu_with_noise. This is not supported.");
  auto res = rrelu_with_noise_batch_rule(self_value, self_bdim, noise_value, noise_bdim, lower, upper, training, std::move(generator));
  return makeBatched(std::move(std::get<0>(res)), std::get<1>(res), cur_level);
}

static std::tuple<Tensor, std::optional<int64_t>> log_sigmoid_backward_batch_rule(
  Tensor& grad, std::optional<int64_t> grad_bdim,
  Tensor& self, std::optional<int64_t> self_bdim,
  Tensor& buffer, std::optional<int64_t> buffer_bdim) {
  // NB: This emulates handle_pointwise_ops except we ignore the last argument, buffer
  // when any of the inputs are on cuda.
  // We do this because on cuda, buffer is a dummy tensor always of logical rank 1 and
  // it becomes an issue when the rest of the inputs are scalar
  int64_t out_logical_rank = std::max(rankWithoutBatchDim(grad, grad_bdim), rankWithoutBatchDim(self, self_bdim));
  if (!grad.is_cuda() && !self.is_cuda() && !buffer.is_cuda()) {
    out_logical_rank = std::max(out_logical_rank, rankWithoutBatchDim(buffer, buffer_bdim));
  }
  Tensor out_grad = maybePadToLogicalRank(moveBatchDimToFront(grad, grad_bdim), grad_bdim, out_logical_rank);
  Tensor out_self = maybePadToLogicalRank(moveBatchDimToFront(self, self_bdim), self_bdim, out_logical_rank);
  Tensor out_buffer = maybePadToLogicalRank(moveBatchDimToFront(buffer, buffer_bdim), buffer_bdim, out_logical_rank);
  return std::make_tuple(at::log_sigmoid_backward(out_grad, out_self, out_buffer), 0);
}

static Tensor binomial_wrapper(const Tensor& count, const Tensor& prob, std::optional<Generator> gen) {
  return at::binomial(count, prob.contiguous(), std::move(gen)); // Bug in PyTorch, prob shouldn't need to be contiguous
}

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  #define BINARY_RANDOM_POINTWISE(op) \
    m.impl(#op, BINARY_RANDOM_POINTWISE_BATCH_RULE(ATEN_FN(op)));
  #define BINARY_RANDOM_POINTWISE2(op, overload) \
    m.impl(#op"."#overload, BINARY_RANDOM_POINTWISE_BATCH_RULE(ATEN_FN2(op, overload)));

  BINARY_RANDOM_POINTWISE2(normal, Tensor_Tensor);
  m.impl("binomial", BINARY_RANDOM_POINTWISE_BATCH_RULE(at::functorch::binomial_wrapper));
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
#define BINARY_POINTWISE2(op, overload) \
  VMAP_SUPPORT2(op, overload, BINARY_POINTWISE_BATCH_RULE(ATEN_FN2(op, overload)));
#define BINARY_POINTWISE(op) \
  VMAP_SUPPORT(op, BINARY_POINTWISE_BATCH_RULE(ATEN_FN(op)));
#define UNARY_POINTWISE2(op, overload) \
  VMAP_SUPPORT2(op, overload, BASIC_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));
#define UNARY_POINTWISE(op) \
  VMAP_SUPPORT(op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));
#define UNARY_SCALAR_POINTWISE2(op, overload) \
  VMAP_SUPPORT(op, overload, SCALAR_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));

#define BINARY_SCALAR_2(op, tensor_tensor, tensor_scalar) \
  BINARY_POINTWISE2(op, tensor_tensor);\
  UNARY_POINTWISE2(op, tensor_scalar);

// For all 3 combinations of Tensor x Tensor, Tensor x Scalar, Scalar x Tensor
#define BINARY_SCALAR_3(op, tensor_tensor, tensor_scalar, scalar_tensor) \
  BINARY_POINTWISE2(op, tensor_tensor);\
  UNARY_POINTWISE2(op, tensor_scalar);\
  POINTWISE_BOXED(op.scalar_tensor);

#define BINARY_SCALAR_3_Tensor(op, tensor_scalar, scalar_tensor) \
  BINARY_POINTWISE(op);\
  UNARY_POINTWISE2(op, tensor_scalar);\
  POINTWISE_BOXED(op.scalar_tensor);

  // Batching rule registrations start
  POINTWISE_BOXED(__ilshift__.Tensor);
  POINTWISE_BOXED(__ilshift__.Scalar);
  POINTWISE_BOXED(__irshift__.Tensor)
  POINTWISE_BOXED(__irshift__.Scalar)
  BINARY_SCALAR_2(__lshift__, Tensor, Scalar);
  BINARY_SCALAR_2(__rshift__, Tensor, Scalar);

  BINARY_SCALAR_2(add, Tensor, Scalar);
  POINTWISE_BOXED(addcdiv);
  POINTWISE_BOXED(addcmul);
  BINARY_POINTWISE(atan2);
  BINARY_SCALAR_2(bitwise_and, Tensor, Scalar);
  BINARY_POINTWISE2(bitwise_and_, Tensor);
  POINTWISE_BOXED(bitwise_and_.Scalar);
  POINTWISE_BOXED(bitwise_and.Scalar_Tensor);
  BINARY_SCALAR_2(bitwise_or, Tensor, Scalar);
  BINARY_POINTWISE2(bitwise_or_, Tensor);
  POINTWISE_BOXED(bitwise_or_.Scalar);
  POINTWISE_BOXED(bitwise_or.Scalar_Tensor);
  BINARY_SCALAR_2(bitwise_xor, Tensor, Scalar);
  BINARY_POINTWISE2(bitwise_xor_, Tensor);
  POINTWISE_BOXED(bitwise_xor_.Scalar);
  POINTWISE_BOXED(bitwise_xor.Scalar_Tensor);
  BINARY_SCALAR_3(bitwise_left_shift, Tensor, Tensor_Scalar, Scalar_Tensor);
  POINTWISE_BOXED(bitwise_left_shift_.Tensor_Scalar);
  POINTWISE_BOXED(bitwise_left_shift_.Tensor);
  BINARY_SCALAR_3(bitwise_right_shift, Tensor, Tensor_Scalar, Scalar_Tensor);
  POINTWISE_BOXED(bitwise_right_shift_.Tensor_Scalar);
  POINTWISE_BOXED(bitwise_right_shift_.Tensor);

  UNARY_POINTWISE(clamp);
  POINTWISE_BOXED(clamp.Tensor);
  BINARY_POINTWISE2(clamp_min, Tensor);
  UNARY_POINTWISE(clamp_min);
  POINTWISE_BOXED(clamp_min_);
  BINARY_POINTWISE2(clamp_max, Tensor);
  UNARY_POINTWISE(clamp_max);
  POINTWISE_BOXED(clamp_max_);
  BINARY_POINTWISE(complex);

  VARIADIC_BDIMS_BOXED(_euclidean_dist);
  // Implementation note: _binary_pointwise_helper performs a dtype promotion if args are scalars,
  // but cdist can't work with scalars, at least 2d tensors.
  BINARY_POINTWISE(_cdist_forward);
  VMAP_SUPPORT(_cdist_backward, cdist_backward_batch_rule);

  BINARY_SCALAR_2(copysign, Tensor, Scalar);
  POINTWISE_BOXED(copysign_.Tensor);
  POINTWISE_BOXED(copysign_.Scalar);
  BINARY_SCALAR_2(div, Tensor, Scalar);
  BINARY_SCALAR_2(div, Tensor_mode, Scalar_mode);

  BINARY_POINTWISE(floor_divide);
  UNARY_POINTWISE2(floor_divide, Scalar);

  BINARY_POINTWISE(fmax);
  BINARY_POINTWISE(fmin);
  BINARY_SCALAR_2(fmod, Tensor, Scalar);
  POINTWISE_BOXED(frexp.Tensor);
  BINARY_POINTWISE(heaviside);
  BINARY_POINTWISE(hypot);
  BINARY_POINTWISE(gcd);
  BINARY_POINTWISE(igamma);
  BINARY_POINTWISE(igammac);
  BINARY_POINTWISE(logaddexp);
  BINARY_POINTWISE(logaddexp2);
  POINTWISE_BOXED(lerp.Scalar);
  POINTWISE_BOXED(lerp.Tensor);
  BINARY_POINTWISE(lcm);
  POINTWISE_BOXED(log_sigmoid_forward);
  BINARY_POINTWISE(maximum);
  BINARY_POINTWISE(minimum);

  BINARY_SCALAR_2(mul, Tensor, Scalar);
  BINARY_POINTWISE(nextafter);
  BINARY_SCALAR_3(pow, Tensor_Tensor, Tensor_Scalar, Scalar);
  POINTWISE_BOXED2(pow_, Scalar);
  BINARY_POINTWISE(polar);
  POINTWISE_BOXED(polygamma);
  BINARY_SCALAR_2(sub, Tensor, Scalar);
  BINARY_SCALAR_3(remainder, Tensor, Scalar, Scalar_Tensor);
  BINARY_SCALAR_2(rsub, Tensor, Scalar);

  BINARY_SCALAR_3_Tensor(special_xlog1py, other_scalar, self_scalar);
  BINARY_SCALAR_3_Tensor(special_zeta, other_scalar, self_scalar);

  VMAP_SUPPORT2(where, self, where_self_batch_rule);

  BINARY_SCALAR_3(xlogy, Tensor, Scalar_Other, Scalar_Self);

  POINTWISE_BOXED(elu_backward);
  BINARY_POINTWISE(hardsigmoid_backward);
  BINARY_POINTWISE(hardtanh_backward);
  BINARY_POINTWISE(hardshrink_backward);
  BINARY_POINTWISE(hardswish_backward);
  BINARY_POINTWISE(_prelu_kernel);
  VARIADIC_BDIMS_BOXED(_prelu_kernel_backward);
  BINARY_POINTWISE(leaky_relu_backward);
  BINARY_POINTWISE(logit_backward);
  VMAP_SUPPORT(log_sigmoid_backward, log_sigmoid_backward_batch_rule);
  VMAP_SUPPORT(gelu_backward, gelu_backward_batch_rule);
  BINARY_POINTWISE(sigmoid_backward);
  POINTWISE_BOXED(softplus_backward);
  BINARY_POINTWISE(softshrink_backward);
  BINARY_POINTWISE(tanh_backward);
  BINARY_POINTWISE(threshold_backward);
  BINARY_POINTWISE(silu_backward);

  using TensorScalarInplaceT = Tensor& (Tensor::*)(const Tensor&, const Scalar&) const;
  using ScalarScalarInplaceT = Tensor& (Tensor::*)(const Scalar&, const Scalar&) const;
  using TensorInplaceT = Tensor& (Tensor::*)(const Tensor&) const;
  using TensorInplaceModeT = Tensor& (Tensor::*)(const Tensor&, std::optional<std::string_view>) const;
  using ScalarInplaceT = Tensor& (Tensor::*)(const Scalar&) const;
  using CopyT = Tensor& (Tensor::*)(const Tensor&, bool) const;

  POINTWISE_BOXED(add_.Tensor); // just testing
  POINTWISE_BOXED(atan2_);
  POINTWISE_BOXED(gcd_);
  POINTWISE_BOXED(lcm_);
  VMAP_SUPPORT2(add_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarScalarInplaceT, &Tensor::add_, const Scalar&, const Scalar&>));
  VMAP_SUPPORT2(sub_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorScalarInplaceT, &Tensor::sub_, const Scalar&>));
  VMAP_SUPPORT2(sub_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarScalarInplaceT, &Tensor::sub_, const Scalar&, const Scalar&>));
  VMAP_SUPPORT2(mul_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::mul_>));
  VMAP_SUPPORT2(mul_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarInplaceT, &Tensor::mul_, const Scalar&>));
  VMAP_SUPPORT2(div_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::div_>));
  VMAP_SUPPORT2(div_, Tensor_mode, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceModeT, &Tensor::div_, std::optional<std::string_view>>));
  VMAP_SUPPORT2(div_, Scalar, SINGLE_ARG(unary_inplace_batch_rule<ScalarInplaceT, &Tensor::div_, const Scalar&>));
  VMAP_SUPPORT2(clamp_min_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::clamp_min_>));
  VMAP_SUPPORT2(clamp_max_, Tensor, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor::clamp_max_>));
  VMAP_SUPPORT2(masked_fill_, Scalar, SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorScalarInplaceT, &Tensor::masked_fill_, const Scalar&>));
  VMAP_SUPPORT(copy_, SINGLE_ARG(binary_pointwise_inplace_batch_rule<CopyT, &Tensor::copy_, bool>));

#define COMPARISON_POINTWISE(op) \
  VMAP_SUPPORT2(op, Tensor, \
      SINGLE_ARG(comparison_pointwise_batch_rule<decltype(&ATEN_FN2(op, Tensor)), &at::op>)); \
  UNARY_POINTWISE2(op, Scalar)

  COMPARISON_POINTWISE(eq);
  COMPARISON_POINTWISE(gt);
  COMPARISON_POINTWISE(ge);
  COMPARISON_POINTWISE(le);
  COMPARISON_POINTWISE(lt);
  COMPARISON_POINTWISE(ne);

#undef COMPARISON_POINTWISE
#undef BINARY_POINTWISE2
#undef BINARY_POINTWISE
#undef UNARY_POINTWISE2
#undef UNARY_POINTWISE
#undef UNARY_SCALAR_POINTWISE2
#undef BINARY_SCALAR_3

#define LOGICAL_COMPARISON_POINTWISE(op) \
  VMAP_SUPPORT(op, \
      SINGLE_ARG(comparison_pointwise_batch_rule<decltype(&ATEN_FN(op)), &ATEN_FN(op)>)); \
  VMAP_SUPPORT(op ## _, \
      SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor:: op ## _ >));

  LOGICAL_COMPARISON_POINTWISE(logical_and);
  LOGICAL_COMPARISON_POINTWISE(logical_or);
  LOGICAL_COMPARISON_POINTWISE(logical_xor);

#undef SINGLE_ARG
#undef LOGICAL_COMPARISON_POINTWISE
  VMAP_SUPPORT(masked_select, masked_select_batch_rule);
  VMAP_SUPPORT(masked_select_backward, masked_select_backward_batch_rule);

  VMAP_SUPPORT2(fill_, Tensor, fill__Tensor_batch_rule);
  m.impl("rrelu_with_noise", rrelu_with_noise_batch);
}

} // namespace at::functorch
