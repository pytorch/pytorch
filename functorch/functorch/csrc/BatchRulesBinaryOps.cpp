#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

static void handleScalarTypePromotion(Tensor& logical_scalar_tensor, Tensor& second) {
  auto result_type = at::native::result_type(logical_scalar_tensor[0], second);
  if (logical_scalar_tensor.scalar_type() != result_type) {
    logical_scalar_tensor = logical_scalar_tensor.to(result_type);
  }
  if (second.scalar_type() != result_type) {
    second = second.to(result_type);
  }
}

static Tensor maybePadToLogicalRank(const Tensor& tensor, optional<int64_t> has_bdim, int64_t logical_rank) {
  if (!has_bdim) {
    return tensor;
  }
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, has_bdim);
  if (tensor_logical_rank >= logical_rank) {
    return tensor;
  }
  VmapDimVector new_sizes(tensor.sizes().begin(), tensor.sizes().end());
  for (int64_t i = 0; i < logical_rank - tensor_logical_rank; i++) {
    new_sizes.insert(new_sizes.begin() + 1, 1);
  }
  return tensor.view(new_sizes);
}

template <typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>> binary_pointwise_batch_rule(
    const Tensor& tensor, optional<int64_t> tensor_batch_dim,
    const Tensor& other, optional<int64_t> other_batch_dim,
    ExtraArgs... extra_args) {
  // compute max logical rank
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // In the (0D, ND) case, type promotion semantics are different :/
  auto tensor_is_logical_scalar = (tensor_logical_rank == 0 && tensor_batch_dim.has_value());
  auto other_is_logical_scalar = (other_logical_rank == 0 && other_batch_dim.has_value());
  if (tensor_is_logical_scalar && !other_is_logical_scalar) {
    handleScalarTypePromotion(tensor_, other_);
  }
  if (other_is_logical_scalar && !tensor_is_logical_scalar) {
    handleScalarTypePromotion(other_, tensor_);
  }

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  auto result = Func(tensor_, other_, std::forward<ExtraArgs>(extra_args)...);
  auto result_batch_dim = tensor_batch_dim.has_value() || other_batch_dim.has_value()
    ? optional<int64_t>{0} : nullopt;
  return { std::move(result), std::move(result_batch_dim) };
}

template <typename F, F Func>
std::tuple<Tensor,optional<int64_t>> comparison_pointwise_batch_rule(
    const Tensor& tensor, optional<int64_t> tensor_batch_dim,
    const Tensor& other, optional<int64_t> other_batch_dim) {
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
  auto result_batch_dim = tensor_batch_dim.has_value() || other_batch_dim.has_value()
    ? optional<int64_t>{0} : nullopt;
  return { std::move(result), std::move(result_batch_dim) };
}

std::tuple<Tensor,optional<int64_t>> pow_scalar_tensor_batch_rule(
    const Scalar& other,
    const Tensor& tensor, optional<int64_t> tensor_batch_dim) {
  return { at::pow(other, tensor), tensor_batch_dim };
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  using TensorTensorScalarType = Tensor (*)(const Tensor&, const Tensor&, const Scalar&);
  using TensorTensorType = Tensor (*)(const Tensor&, const Tensor&);
  using TensorScalarType = Tensor (*)(const Tensor&, const Scalar&);
  using TensorScalarScalarType = Tensor (*)(const Tensor&, const Scalar&, const Scalar&);

#define BINARY_POINTWISE_BATCH_RULE_SCALAR(op) \
  binary_pointwise_batch_rule<TensorTensorScalarType, &op, const Scalar&>
#define BINARY_POINTWISE_WITH_SCALAR(op) \
  VMAP_SUPPORT(#op".Tensor", BINARY_POINTWISE_BATCH_RULE_SCALAR(at::op));

#define BINARY_POINTWISE_BATCH_RULE(op) binary_pointwise_batch_rule<TensorTensorType, &op>
#define BINARY_POINTWISE(op) VMAP_SUPPORT(#op".Tensor", BINARY_POINTWISE_BATCH_RULE(at::op));
#define SINGLE_ARG(...) __VA_ARGS__

  BINARY_POINTWISE_WITH_SCALAR(add);
  BINARY_POINTWISE_WITH_SCALAR(sub);
  BINARY_POINTWISE_WITH_SCALAR(rsub);
  BINARY_POINTWISE(mul);
  VMAP_SUPPORT("add.Scalar", SINGLE_ARG(basic_unary_batch_rule<TensorScalarScalarType, &at::add, const Scalar&, const Scalar&>));
  VMAP_SUPPORT("sub.Scalar", SINGLE_ARG(basic_unary_batch_rule<TensorScalarScalarType, &at::sub, const Scalar&, const Scalar&>));
  VMAP_SUPPORT("rsub.Scalar", SINGLE_ARG(basic_unary_batch_rule<TensorScalarScalarType, &at::rsub, const Scalar&, const Scalar&>));
  VMAP_SUPPORT("mul.Scalar", SINGLE_ARG(basic_unary_batch_rule<TensorScalarType, &at::mul, const Scalar&>));
  VMAP_SUPPORT("div.Scalar", SINGLE_ARG(basic_unary_batch_rule<TensorScalarType, &at::div, const Scalar&>));
  BINARY_POINTWISE(div);
  VMAP_SUPPORT("tanh_backward", BINARY_POINTWISE_BATCH_RULE(at::tanh_backward));
  VMAP_SUPPORT("threshold_backward", SINGLE_ARG(
        binary_pointwise_batch_rule<decltype(&at::threshold_backward), &at::threshold_backward, const Scalar&>));

  // at::pow has three out-of-place overloads
  VMAP_SUPPORT("pow.Tensor_Tensor", SINGLE_ARG(binary_pointwise_batch_rule<TensorTensorType, &at::pow>));
  VMAP_SUPPORT("pow.Tensor_Scalar", SINGLE_ARG(basic_unary_batch_rule<TensorScalarType, &at::pow, const Scalar&>));
  VMAP_SUPPORT("pow.Scalar", pow_scalar_tensor_batch_rule);

  VMAP_SUPPORT("clamp_min.Tensor",
      SINGLE_ARG(binary_pointwise_batch_rule<TensorTensorType, &at::clamp_min>));
  VMAP_SUPPORT("clamp_min",
      SINGLE_ARG(basic_unary_batch_rule<TensorScalarType, &at::clamp_min, const Scalar&>));
  VMAP_SUPPORT("clamp_max.Tensor",
      SINGLE_ARG(binary_pointwise_batch_rule<TensorTensorType, &at::clamp_max>));
  VMAP_SUPPORT("clamp_max",
      SINGLE_ARG(basic_unary_batch_rule<TensorScalarType, &at::clamp_max, const Scalar&>));


#define COMPARISON_POINTWISE(op) \
  VMAP_SUPPORT(#op".Tensor", \
      SINGLE_ARG(comparison_pointwise_batch_rule<TensorTensorType, &at::op>)); \
  VMAP_SUPPORT(#op".Scalar", \
      SINGLE_ARG(basic_unary_batch_rule<TensorScalarType, &at::op, const Scalar&>));

  COMPARISON_POINTWISE(eq);
  COMPARISON_POINTWISE(gt);
  COMPARISON_POINTWISE(ge);
  COMPARISON_POINTWISE(le);
  COMPARISON_POINTWISE(lt);
  COMPARISON_POINTWISE(ne);

#undef COMPARISON_POINTWISE
#undef SINGLE_ARG
#undef BINARY_POINTWISE_BATCH_RULE_SCALAR
#undef BINARY_POINTWISE_BATCH_RULE
#undef BINARY_POINTWISE_WITH_SCALAR
#undef BINARY_POINTWISE
}

}}
