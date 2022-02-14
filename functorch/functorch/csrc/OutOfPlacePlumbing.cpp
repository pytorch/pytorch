#include <functorch/csrc/OutOfPlacePlumbing.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>

namespace at { namespace functorch {
// ['_cast_Byte', '_cast_Char', '_cast_Double', '_cast_Float', '_cast_Int', '_cast_Long', '_cast_Short', '_cast_Half', 'matrix_rank', 'std', 'var', 'nuclear_norm', 'cholesky', 'cholesky_inverse', 'linalg_cholesky']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_0_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_0_t,Tensor,const Tensor &, bool>(
  batch_rule_0_t batch_rule,
  const Tensor & self, bool non_blocking
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['data', '_shape_as_tensor', 'abs', 'absolute', 'angle', 'view_as_real', 'view_as_complex', 'sgn', 'real', 'imag', '_conj', 'conj', '_conj_physical', 'conj_physical', 'resolve_conj', 'resolve_neg', '_neg_view', 'acos', 'arccos', 'acosh', 'arccosh', 'asinh', 'arcsinh', 'atanh', 'arctanh', 'asin', 'arcsin', 'atan', 'arctan', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'bitwise_not', 'logical_not', 'ceil', 'cos', 'cosh', 'corrcoef', 'erf', 'erfc', 'exp', 'exp2', 'expm1', 'floor', 'frac', 'inverse', 'isnan', 'isreal', 'fbgemm_pack_gemm_matrix_fp16', 'fbgemm_pack_quantized_matrix', 'log', 'log10', 'log1p', 'log2', 'logdet', 'matrix_exp', 'median', 'nanmedian', 'numpy_T', 'matrix_H', 'mT', 'mH', 'adjoint', 'rad2deg', 'deg2rad', 'ravel', 'reciprocal', 'neg', 'negative', 'round', 'relu', 'relu6', 'rsqrt', 'selu', 'silu', 'mish', 'sigmoid', 'sin', 'sinc', 'sinh', 'detach', 'squeeze', 'sqrt', 'square', 't', 'tan', 'tanh', 'fliplr', 'flipud', 'trunc', 'fix', '_sparse_sum', 'frobenius_norm', 'positive', 'coalesce', '_coalesce', '_indices', '_values', 'indices', 'values', 'crow_indices', 'col_indices', 'to_sparse', 'dequantize.self', 'q_per_channel_scales', 'q_per_channel_zero_points', 'int_repr', '_saturate_weight_to_fp16', 'trace', 'nonzero', 'argwhere', 'lgamma', 'digamma', 'erfinv', 'i0', 'sign', 'signbit', 'min', 'max', 'msort', 'all', 'any', 'alias', '_torch_cuda_cu_linker_symbol_op', 'hardsigmoid', 'hardswish', 'log_sigmoid', 'isfinite', 'isinf', 'isposinf', 'isneginf', 'special_entr', 'special_ndtri', 'special_expm1', 'special_exp2', 'special_psi', 'special_digamma', 'special_gammaln', 'special_erf', 'special_erfc', 'special_erfcx', 'special_erfinv', 'special_ndtr', 'special_i0', 'special_i0e', 'special_i1', 'special_i1e', 'special_expit', 'special_sinc', 'special_log1p', 'linalg_det', 'det', 'linalg_matrix_exp', 'linalg_eigvals', 'linalg_inv', 'linalg_svdvals', '_test_warn_in_autograd']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_1_t)(const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_1_t,Tensor,const Tensor &>(
  batch_rule_1_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['is_leaf', 'retains_grad', 'cudnn_is_acceptable', 'is_distributed', 'is_floating_point', 'is_complex', 'is_conj', '_is_zerotensor', 'is_neg', 'is_nonzero', 'is_signed', 'is_inference', 'is_coalesced']
typedef std::tuple<bool> (*batch_rule_2_t)(const Tensor &, c10::optional<int64_t>);
template <>
bool lowerToNextLayer<batch_rule_2_t,bool,const Tensor &>(
  batch_rule_2_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::get<0>(results);
}

// ['output_nr', '_version', '_debug_has_internal_overlap', 'sparse_dim', '_dimI', 'dense_dim', '_dimV', '_nnz', 'q_zero_point', 'q_per_channel_axis']
typedef std::tuple<int64_t> (*batch_rule_3_t)(const Tensor &, c10::optional<int64_t>);
template <>
int64_t lowerToNextLayer<batch_rule_3_t,int64_t,const Tensor &>(
  batch_rule_3_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::get<0>(results);
}

// ['_fw_primal', '_dim_arange', 'diagflat', '_logcumsumexp', 'logcumsumexp', 'matrix_power', 'mvlgamma', 'pixel_shuffle', 'pixel_unshuffle', 'channel_shuffle', 'native_channel_shuffle', 'round.decimals', 'squeeze.dim', 'one_hot', 'unsqueeze', 'to_sparse.sparse_dim', 'diag', 'triu', 'tril', 'glu', 'special_round', 'special_multigammaln', 'linalg_tensorinv', 'linalg_matrix_power']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_4_t)(const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_4_t,Tensor,const Tensor &, int64_t>(
  batch_rule_4_t batch_rule,
  const Tensor & self, int64_t level
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, level);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_make_dual', '_new_zeros_with_same_feature_meta', 'cumulative_trapezoid.x', 'trapezoid.x', 'trapz.x', '_weight_norm', 'mse_loss', 'l1_loss', 'multilabel_margin_loss', 'soft_margin_loss', 'glu_backward', 'linalg_cross']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_5_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_5_t,Tensor,const Tensor &, const Tensor &, int64_t>(
  batch_rule_5_t batch_rule,
  const Tensor & primal, const Tensor & tangent, int64_t level
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor primal_value;
  optional<int64_t> primal_bdim;
  std::tie(primal_value, primal_bdim) = unwrapTensorAtLevel(primal, cur_level);
  Tensor tangent_value;
  optional<int64_t> tangent_bdim;
  std::tie(tangent_value, tangent_bdim) = unwrapTensorAtLevel(tangent, cur_level);
  auto results = batch_rule(primal_value, primal_bdim, tangent_value, tangent_bdim, level);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_unpack_dual', 'cummax', 'cummin']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_6_t)(const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_6_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t>(
  batch_rule_6_t batch_rule,
  const Tensor & dual, int64_t level
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor dual_value;
  optional<int64_t> dual_bdim;
  std::tie(dual_value, dual_bdim) = unwrapTensorAtLevel(dual, cur_level);
  auto results = batch_rule(dual_value, dual_bdim, level);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_has_same_storage_numel', 'is_same_size', '_has_compatible_shallow_copy_type', 'is_set_to', 'equal']
typedef std::tuple<bool> (*batch_rule_7_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
bool lowerToNextLayer<batch_rule_7_t,bool,const Tensor &, const Tensor &>(
  batch_rule_7_t batch_rule,
  const Tensor & self, const Tensor & other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return std::get<0>(results);
}

// ['rename']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_8_t)(const Tensor &, c10::optional<int64_t>, c10::optional<DimnameList>);
template <>
Tensor lowerToNextLayer<batch_rule_8_t,Tensor,const Tensor &, c10::optional<DimnameList>>(
  batch_rule_8_t batch_rule,
  const Tensor & self, c10::optional<DimnameList> names
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['align_to', 'refine_names']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_9_t)(const Tensor &, c10::optional<int64_t>, DimnameList);
template <>
Tensor lowerToNextLayer<batch_rule_9_t,Tensor,const Tensor &, DimnameList>(
  batch_rule_9_t batch_rule,
  const Tensor & self, DimnameList names
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['align_to.ellipsis_idx']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_10_t)(const Tensor &, c10::optional<int64_t>, DimnameList, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_10_t,Tensor,const Tensor &, DimnameList, int64_t>(
  batch_rule_10_t batch_rule,
  const Tensor & self, DimnameList order, int64_t ellipsis_idx
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, order, ellipsis_idx);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['align_as', '_reshape_from_tensor', 'copysign.Tensor', 'logical_xor', 'logical_and', 'logical_or', 'bmm', 'clamp_max.Tensor', 'clamp_min.Tensor', 'complex', 'polar', '_copy_from_and_resize', 'cudnn_grid_sampler', 'div.Tensor', 'divide.Tensor', 'true_divide.Tensor', 'dot', 'vdot', 'expand_as', 'floor_divide', 'gcd', 'lcm', 'kron', 'ldexp.Tensor', 'logaddexp', 'logaddexp2', 'xlogy.Tensor', 'matmul', 'matrix_exp_backward', '_compute_linear_combination', 'mm', '_sparse_mm', '_sparse_sparse_matmul', '_sparse_mask_helper', 'mul.Tensor', 'multiply.Tensor', 'mv', '_euclidean_dist', 'reshape_as', 'prelu', 'infinitely_differentiable_gelu_backward', 'silu_backward', 'mish_backward', 'smm', 'type_as', 'view_as', '_standard_gamma_grad', 'heaviside', 'sparse_mask', 'to_dense_backward', 'hspmm', 'to_mkldnn_backward', 'fake_quantize_per_tensor_affine_cachemask_backward', 'fake_quantize_per_channel_affine_cachemask_backward', '_masked_softmax', 'bitwise_and.Tensor', '__and__.Tensor', 'bitwise_or.Tensor', '__or__.Tensor', 'bitwise_xor.Tensor', '__xor__.Tensor', '__lshift__.Tensor', 'bitwise_left_shift.Tensor', '__rshift__.Tensor', 'bitwise_right_shift.Tensor', 'ne.Tensor', 'not_equal.Tensor', 'eq.Tensor', 'ge.Tensor', 'greater_equal.Tensor', 'le.Tensor', 'less_equal.Tensor', 'gt.Tensor', 'greater.Tensor', 'lt.Tensor', 'less.Tensor', 'take', 'masked_select', 'orgqr', 'atan2', 'arctan2', 'fmod.Tensor', 'hypot', 'igamma', 'igammac', 'nextafter', 'remainder.Tensor', 'fmin', 'fmax', 'maximum', 'max.other', 'minimum', 'min.other', 'pow.Tensor_Tensor', 'float_power.Tensor_Tensor', 'hardsigmoid_backward', 'hardswish_backward', 'mkldnn_adaptive_avg_pool2d_backward', '_adaptive_avg_pool2d_backward', '_adaptive_avg_pool3d_backward', 'sigmoid_backward', 'tanh_backward', 'special_xlog1py', 'special_xlogy', 'special_zeta', 'special_gammainc', 'special_gammaincc', 'linalg_matmul', 'linalg_householder_product', 'inner', 'outer', 'ger', 'linalg_solve']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_11_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_11_t,Tensor,const Tensor &, const Tensor &>(
  batch_rule_11_t batch_rule,
  const Tensor & self, const Tensor & other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_use_cudnn_ctc_loss']
typedef std::tuple<bool> (*batch_rule_12_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t);
template <>
bool lowerToNextLayer<batch_rule_12_t,bool,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_12_t batch_rule,
  const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor log_probs_value;
  optional<int64_t> log_probs_bdim;
  std::tie(log_probs_value, log_probs_bdim) = unwrapTensorAtLevel(log_probs, cur_level);
  Tensor targets_value;
  optional<int64_t> targets_bdim;
  std::tie(targets_value, targets_bdim) = unwrapTensorAtLevel(targets, cur_level);
  auto results = batch_rule(log_probs_value, log_probs_bdim, targets_value, targets_bdim, input_lengths, target_lengths, blank);
  return std::get<0>(results);
}

// ['_cudnn_ctc_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_13_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_13_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_13_t batch_rule,
  const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor log_probs_value;
  optional<int64_t> log_probs_bdim;
  std::tie(log_probs_value, log_probs_bdim) = unwrapTensorAtLevel(log_probs, cur_level);
  Tensor targets_value;
  optional<int64_t> targets_bdim;
  std::tie(targets_value, targets_bdim) = unwrapTensorAtLevel(targets, cur_level);
  auto results = batch_rule(log_probs_value, log_probs_bdim, targets_value, targets_bdim, input_lengths, target_lengths, blank, deterministic, zero_infinity);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_fused_dropout']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_14_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<Generator>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_14_t,std::tuple<Tensor,Tensor>,const Tensor &, double, c10::optional<Generator>>(
  batch_rule_14_t batch_rule,
  const Tensor & self, double p, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, generator);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_masked_scale', 'native_dropout_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_15_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double);
template <>
Tensor lowerToNextLayer<batch_rule_15_t,Tensor,const Tensor &, const Tensor &, double>(
  batch_rule_15_t batch_rule,
  const Tensor & self, const Tensor & mask, double scale
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mask_value;
  optional<int64_t> mask_bdim;
  std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, scale);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['native_dropout']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_16_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<bool>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_16_t,std::tuple<Tensor,Tensor>,const Tensor &, double, c10::optional<bool>>(
  batch_rule_16_t batch_rule,
  const Tensor & input, double p, c10::optional<bool> train
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, p, train);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_sobol_engine_draw']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_17_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, c10::optional<ScalarType>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_17_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, const Tensor &, int64_t, int64_t, c10::optional<ScalarType>>(
  batch_rule_17_t batch_rule,
  const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor quasi_value;
  optional<int64_t> quasi_bdim;
  std::tie(quasi_value, quasi_bdim) = unwrapTensorAtLevel(quasi, cur_level);
  Tensor sobolstate_value;
  optional<int64_t> sobolstate_bdim;
  std::tie(sobolstate_value, sobolstate_bdim) = unwrapTensorAtLevel(sobolstate, cur_level);
  auto results = batch_rule(quasi_value, quasi_bdim, n, sobolstate_value, sobolstate_bdim, dimension, num_generated, dtype);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['dropout', 'feature_dropout', 'alpha_dropout', 'feature_alpha_dropout', 'matrix_rank.tol', 'linalg_pinv', 'linalg_matrix_rank']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_18_t)(const Tensor &, c10::optional<int64_t>, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_18_t,Tensor,const Tensor &, double, bool>(
  batch_rule_18_t batch_rule,
  const Tensor & input, double p, bool train
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, p, train);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['avg_pool1d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_19_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_19_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool>(
  batch_rule_19_t batch_rule,
  const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, stride, padding, ceil_mode, count_include_pad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['adaptive_avg_pool1d', 'broadcast_to', '_sparse_broadcast_to', 'count_nonzero.dim_IntList', 'permute', 'repeat', 'reshape', '_mkldnn_reshape', 'sum_to_size', 'tile', 'flip', '_unsafe_view', '_sparse_sum.dim', 'view', 'trace_backward', 'adaptive_avg_pool2d', 'mkldnn_adaptive_avg_pool2d', '_adaptive_avg_pool2d', 'adaptive_avg_pool3d', '_adaptive_avg_pool3d', 'reflection_pad1d', 'reflection_pad2d', 'reflection_pad3d', 'replication_pad1d', 'replication_pad2d', 'replication_pad3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_20_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_20_t,Tensor,const Tensor &, IntArrayRef>(
  batch_rule_20_t batch_rule,
  const Tensor & self, IntArrayRef output_size
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_21_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_21_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef>(
  batch_rule_21_t batch_rule,
  const Tensor & self, IntArrayRef output_size
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['add.Tensor', '_add_relu.Tensor', 'hardshrink_backward', 'threshold_backward', 'where.ScalarOther', 'sub.Tensor', 'subtract.Tensor', 'rsub.Tensor', 'masked_fill.Scalar', 'dist', 'lerp.Scalar', 'softshrink_backward', '_test_serialization_subcmul']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_22_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_22_t,Tensor,const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_22_t batch_rule,
  const Tensor & self, const Tensor & other, const Scalar & alpha
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_add_relu.Scalar', 'add.Scalar', 'threshold', 'where.Scalar', 'sub.Scalar', 'subtract.Scalar', 'rsub.Scalar', 'hardtanh', 'softplus']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_23_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_23_t,Tensor,const Tensor &, const Scalar &, const Scalar &>(
  batch_rule_23_t batch_rule,
  const Tensor & self, const Scalar & other, const Scalar & alpha
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['addmv', 'addr', 'baddbmm', 'sspaddmm', '_sparse_addmm', 'sparse_sampled_addmm', 'addmm', 'addbmm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_24_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_24_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &>(
  batch_rule_24_t batch_rule,
  const Tensor & self, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mat_value;
  optional<int64_t> mat_bdim;
  std::tie(mat_value, mat_bdim) = unwrapTensorAtLevel(mat, cur_level);
  Tensor vec_value;
  optional<int64_t> vec_bdim;
  std::tie(vec_value, vec_bdim) = unwrapTensorAtLevel(vec, cur_level);
  auto results = batch_rule(self_value, self_bdim, mat_value, mat_bdim, vec_value, vec_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['affine_grid_generator', 'affine_grid_generator_backward', 'expand', 'logsumexp', 'amax', 'amin', 'frobenius_norm.dim', 'nuclear_norm.dim', 'special_logsumexp']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_25_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_25_t,Tensor,const Tensor &, IntArrayRef, bool>(
  batch_rule_25_t batch_rule,
  const Tensor & theta, IntArrayRef size, bool align_corners
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor theta_value;
  optional<int64_t> theta_bdim;
  std::tie(theta_value, theta_bdim) = unwrapTensorAtLevel(theta, cur_level);
  auto results = batch_rule(theta_value, theta_bdim, size, align_corners);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['all.dim', 'any.dim', '_log_softmax', '_softmax', '_sparse_softmax', '_sparse_log_softmax', 'combinations', 'argsort', '_convert_indices_from_coo_to_csr']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_26_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_26_t,Tensor,const Tensor &, int64_t, bool>(
  batch_rule_26_t batch_rule,
  const Tensor & self, int64_t dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['all.dimname', 'any.dimname', 'argsort.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_27_t)(const Tensor &, c10::optional<int64_t>, Dimname, bool);
template <>
Tensor lowerToNextLayer<batch_rule_27_t,Tensor,const Tensor &, Dimname, bool>(
  batch_rule_27_t batch_rule,
  const Tensor & self, Dimname dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['allclose']
typedef std::tuple<bool> (*batch_rule_28_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, bool);
template <>
bool lowerToNextLayer<batch_rule_28_t,bool,const Tensor &, const Tensor &, double, double, bool>(
  batch_rule_28_t batch_rule,
  const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, rtol, atol, equal_nan);
  return std::get<0>(results);
}

// ['argmax', 'argmin', 'vander']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_29_t)(const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_29_t,Tensor,const Tensor &, c10::optional<int64_t>, bool>(
  batch_rule_29_t batch_rule,
  const Tensor & self, c10::optional<int64_t> dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['as_strided']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_30_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_30_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(
  batch_rule_30_t batch_rule,
  const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride, storage_offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['batch_norm', 'instance_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_31_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_31_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double, bool>(
  batch_rule_31_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, training, momentum, eps, cudnn_enabled);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantized_batch_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_32_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_32_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, double, double, int64_t>(
  batch_rule_32_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor var_value;
  optional<int64_t> var_bdim;
  std::tie(var_value, var_bdim) = unwrapTensorAtLevel(var, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, mean_value, mean_bdim, var_value, var_bdim, eps, output_scale, output_zero_point);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_batch_norm_impl_index']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,int64_t> (*batch_rule_33_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> lowerToNextLayer<batch_rule_33_t,std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double, bool>(
  batch_rule_33_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, training, momentum, eps, cudnn_enabled);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level), std::get<8>(results));
}

// ['_batch_norm_impl_index_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_34_t)(int64_t, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, ::std::array<bool,3>, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_34_t,std::tuple<Tensor,Tensor,Tensor>,int64_t, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, ::std::array<bool,3>, const Tensor &>(
  batch_rule_34_t batch_rule,
  int64_t impl_index, const Tensor & input, const Tensor & grad_output, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_var_transform, bool train, double eps, ::std::array<bool,3> output_mask, const Tensor & reservedSpace
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor reservedSpace_value;
  optional<int64_t> reservedSpace_bdim;
  std::tie(reservedSpace_value, reservedSpace_bdim) = unwrapTensorAtLevel(reservedSpace, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  optional<Tensor> save_mean_value;
  optional<int64_t> save_mean_bdim;
  if (save_mean) {
      std::tie(save_mean_value, save_mean_bdim) = unwrapTensorAtLevel(save_mean.value(), cur_level);
  }
  optional<Tensor> save_var_transform_value;
  optional<int64_t> save_var_transform_bdim;
  if (save_var_transform) {
      std::tie(save_var_transform_value, save_var_transform_bdim) = unwrapTensorAtLevel(save_var_transform.value(), cur_level);
  }
  auto results = batch_rule(impl_index, input_value, input_bdim, grad_output_value, grad_output_bdim, weight_value, weight_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, save_mean_value, save_mean_bdim, save_var_transform_value, save_var_transform_bdim, train, eps, output_mask, reservedSpace_value, reservedSpace_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['bernoulli', '_standard_gamma', '_sample_dirichlet', 'poisson']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_35_t)(const Tensor &, c10::optional<int64_t>, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_35_t,Tensor,const Tensor &, c10::optional<Generator>>(
  batch_rule_35_t batch_rule,
  const Tensor & self, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['bernoulli.p', 'normal.Tensor_float']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_36_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_36_t,Tensor,const Tensor &, double, c10::optional<Generator>>(
  batch_rule_36_t batch_rule,
  const Tensor & self, double p, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['bilinear']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_37_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_37_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &>(
  batch_rule_37_t batch_rule,
  const Tensor & input1, const Tensor & input2, const Tensor & weight, const c10::optional<Tensor> & bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input1_value;
  optional<int64_t> input1_bdim;
  std::tie(input1_value, input1_bdim) = unwrapTensorAtLevel(input1, cur_level);
  Tensor input2_value;
  optional<int64_t> input2_bdim;
  std::tie(input2_value, input2_bdim) = unwrapTensorAtLevel(input2, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input1_value, input1_bdim, input2_value, input2_bdim, weight_value, weight_bdim, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['binary_cross_entropy']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_38_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_38_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_38_t batch_rule,
  const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['binary_cross_entropy_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_39_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_39_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_39_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['binary_cross_entropy_with_logits']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_40_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_40_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_40_t batch_rule,
  const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & pos_weight, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> pos_weight_value;
  optional<int64_t> pos_weight_bdim;
  if (pos_weight) {
      std::tie(pos_weight_value, pos_weight_bdim) = unwrapTensorAtLevel(pos_weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, pos_weight_value, pos_weight_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['binary_cross_entropy_with_logits_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_41_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_41_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_41_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & pos_weight, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> pos_weight_value;
  optional<int64_t> pos_weight_bdim;
  if (pos_weight) {
      std::tie(pos_weight_value, pos_weight_bdim) = unwrapTensorAtLevel(pos_weight.value(), cur_level);
  }
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, pos_weight_value, pos_weight_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['bincount']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_42_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_42_t,Tensor,const Tensor &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_42_t batch_rule,
  const Tensor & self, const c10::optional<Tensor> & weights, int64_t minlength
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> weights_value;
  optional<int64_t> weights_bdim;
  if (weights) {
      std::tie(weights_value, weights_bdim) = unwrapTensorAtLevel(weights.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, weights_value, weights_bdim, minlength);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['copysign.Scalar', 'clamp_max', 'clamp_min', 'div.Scalar', 'divide.Scalar', 'true_divide.Scalar', 'floor_divide.Scalar', 'xlogy.Scalar_Other', 'mul.Scalar', 'multiply.Scalar', 'hardshrink', 'celu', 'native_norm', 'norm.Scalar', 'bitwise_and.Scalar', '__and__.Scalar', 'bitwise_or.Scalar', '__or__.Scalar', 'bitwise_xor.Scalar', '__xor__.Scalar', '__lshift__.Scalar', 'bitwise_left_shift.Tensor_Scalar', '__rshift__.Scalar', 'bitwise_right_shift.Tensor_Scalar', 'ne.Scalar', 'not_equal.Scalar', 'eq.Scalar', 'ge.Scalar', 'greater_equal.Scalar', 'le.Scalar', 'less_equal.Scalar', 'gt.Scalar', 'greater.Scalar', 'lt.Scalar', 'less.Scalar', 'fmod.Scalar', 'remainder.Scalar', 'pow.Tensor_Scalar', 'float_power.Tensor_Scalar', 'leaky_relu', 'softshrink', 'special_xlog1py.other_scalar', 'special_xlogy.other_scalar', 'special_zeta.other_scalar']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_43_t)(const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_43_t,Tensor,const Tensor &, const Scalar &>(
  batch_rule_43_t batch_rule,
  const Tensor & self, const Scalar & other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['unsafe_chunk', 'chunk', 'tensor_split.sections', 'unsafe_split.Tensor', 'split.Tensor']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_44_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_44_t,::std::vector<Tensor>,const Tensor &, int64_t, int64_t>(
  batch_rule_44_t batch_rule,
  const Tensor & self, int64_t chunks, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, chunks, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['tensor_split.indices', 'gradient.array', 'unsafe_split_with_sizes', 'split_with_sizes']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_45_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_45_t,::std::vector<Tensor>,const Tensor &, IntArrayRef, int64_t>(
  batch_rule_45_t batch_rule,
  const Tensor & self, IntArrayRef indices, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['tensor_split.tensor_indices_or_sections']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_46_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_46_t,::std::vector<Tensor>,const Tensor &, const Tensor &, int64_t>(
  batch_rule_46_t batch_rule,
  const Tensor & self, const Tensor & tensor_indices_or_sections, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor tensor_indices_or_sections_value;
  optional<int64_t> tensor_indices_or_sections_bdim;
  std::tie(tensor_indices_or_sections_value, tensor_indices_or_sections_bdim) = unwrapTensorAtLevel(tensor_indices_or_sections, cur_level);
  auto results = batch_rule(self_value, self_bdim, tensor_indices_or_sections_value, tensor_indices_or_sections_bdim, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['clamp', 'clip']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_47_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, const c10::optional<Scalar> &);
template <>
Tensor lowerToNextLayer<batch_rule_47_t,Tensor,const Tensor &, const c10::optional<Scalar> &, const c10::optional<Scalar> &>(
  batch_rule_47_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['clamp.Tensor', 'clip.Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_48_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_48_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_48_t batch_rule,
  const Tensor & self, const c10::optional<Tensor> & min, const c10::optional<Tensor> & max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> min_value;
  optional<int64_t> min_bdim;
  if (min) {
      std::tie(min_value, min_bdim) = unwrapTensorAtLevel(min.value(), cur_level);
  }
  optional<Tensor> max_value;
  optional<int64_t> max_bdim;
  if (max) {
      std::tie(max_value, max_bdim) = unwrapTensorAtLevel(max.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, min_value, min_bdim, max_value, max_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['constant_pad_nd']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_49_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_49_t,Tensor,const Tensor &, IntArrayRef, const Scalar &>(
  batch_rule_49_t batch_rule,
  const Tensor & self, IntArrayRef pad, const Scalar & value
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, pad, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['contiguous']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_50_t)(const Tensor &, c10::optional<int64_t>, MemoryFormat);
template <>
Tensor lowerToNextLayer<batch_rule_50_t,Tensor,const Tensor &, MemoryFormat>(
  batch_rule_50_t batch_rule,
  const Tensor & self, MemoryFormat memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['convolution', 'convolution_overrideable']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_51_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_51_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(
  batch_rule_51_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, dilation, transposed, output_padding, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['convolution_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_52_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_52_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, c10::optional<IntArrayRef>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, ::std::array<bool,3>>(
  batch_rule_52_t batch_rule,
  const Tensor & grad_output, const Tensor & input, const Tensor & weight, c10::optional<IntArrayRef> bias_sizes, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_value, input_bdim, weight_value, weight_bdim, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['convolution_backward_overrideable']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_53_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_53_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, ::std::array<bool,3>>(
  batch_rule_53_t batch_rule,
  const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_value, input_bdim, weight_value, weight_bdim, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_convolution']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_54_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_54_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, bool>(
  batch_rule_54_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_convolution.deprecated']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_55_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_55_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool>(
  batch_rule_55_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_convolution_mode', 'conv1d.padding', 'conv2d.padding', 'conv3d.padding']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_56_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_56_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t>(
  batch_rule_56_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, c10::string_view padding, IntArrayRef dilation, int64_t groups
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, dilation, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_convolution_double_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_57_t)(const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_57_t,std::tuple<Tensor,Tensor,Tensor>,const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, ::std::array<bool,3>>(
  batch_rule_57_t batch_rule,
  const c10::optional<Tensor> & ggI, const c10::optional<Tensor> & ggW, const c10::optional<Tensor> & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor gO_value;
  optional<int64_t> gO_bdim;
  std::tie(gO_value, gO_bdim) = unwrapTensorAtLevel(gO, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> ggI_value;
  optional<int64_t> ggI_bdim;
  if (ggI) {
      std::tie(ggI_value, ggI_bdim) = unwrapTensorAtLevel(ggI.value(), cur_level);
  }
  optional<Tensor> ggW_value;
  optional<int64_t> ggW_bdim;
  if (ggW) {
      std::tie(ggW_value, ggW_bdim) = unwrapTensorAtLevel(ggW.value(), cur_level);
  }
  optional<Tensor> ggb_value;
  optional<int64_t> ggb_bdim;
  if (ggb) {
      std::tie(ggb_value, ggb_bdim) = unwrapTensorAtLevel(ggb.value(), cur_level);
  }
  auto results = batch_rule(ggI_value, ggI_bdim, ggW_value, ggW_bdim, ggb_value, ggb_bdim, gO_value, gO_bdim, weight_value, weight_bdim, self_value, self_bdim, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['conv1d', 'conv2d', 'conv3d', 'cudnn_convolution_relu', 'mkldnn_convolution']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_58_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_58_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_58_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, dilation, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['conv_tbc', 'cummaxmin_backward', '_make_per_channel_quantized_tensor', 'mse_loss_backward', 'l1_loss_backward', 'soft_margin_loss_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_59_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_59_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t>(
  batch_rule_59_t batch_rule,
  const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor bias_value;
  optional<int64_t> bias_bdim;
  std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias, cur_level);
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim, pad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['conv_tbc_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_60_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_60_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(
  batch_rule_60_t batch_rule,
  const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor bias_value;
  optional<int64_t> bias_bdim;
  std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias, cur_level);
  auto results = batch_rule(self_value, self_bdim, input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, pad);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['conv_transpose1d', 'conv_transpose2d.input', 'conv_transpose3d.input']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_61_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_61_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(
  batch_rule_61_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, output_padding, groups, dilation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_copy_from', 'cholesky_solve', '_cholesky_solve_helper', 'linalg_pinv.rcond_tensor', 'linalg_matrix_rank.tol_tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_62_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_62_t,Tensor,const Tensor &, const Tensor &, bool>(
  batch_rule_62_t batch_rule,
  const Tensor & self, const Tensor & dst, bool non_blocking
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor dst_value;
  optional<int64_t> dst_bdim;
  std::tie(dst_value, dst_bdim) = unwrapTensorAtLevel(dst, cur_level);
  auto results = batch_rule(self_value, self_bdim, dst_value, dst_bdim, non_blocking);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cosine_embedding_loss', 'margin_ranking_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_63_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_63_t,Tensor,const Tensor &, const Tensor &, const Tensor &, double, int64_t>(
  batch_rule_63_t batch_rule,
  const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input1_value;
  optional<int64_t> input1_bdim;
  std::tie(input1_value, input1_bdim) = unwrapTensorAtLevel(input1, cur_level);
  Tensor input2_value;
  optional<int64_t> input2_bdim;
  std::tie(input2_value, input2_bdim) = unwrapTensorAtLevel(input2, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  auto results = batch_rule(input1_value, input1_bdim, input2_value, input2_bdim, target_value, target_bdim, margin, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['count_nonzero', 'repeat_interleave.Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_64_t)(const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_64_t,Tensor,const Tensor &, c10::optional<int64_t>>(
  batch_rule_64_t batch_rule,
  const Tensor & self, c10::optional<int64_t> dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cov']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_65_t)(const Tensor &, c10::optional<int64_t>, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_65_t,Tensor,const Tensor &, int64_t, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_65_t batch_rule,
  const Tensor & self, int64_t correction, const c10::optional<Tensor> & fweights, const c10::optional<Tensor> & aweights
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> fweights_value;
  optional<int64_t> fweights_bdim;
  if (fweights) {
      std::tie(fweights_value, fweights_bdim) = unwrapTensorAtLevel(fweights.value(), cur_level);
  }
  optional<Tensor> aweights_value;
  optional<int64_t> aweights_bdim;
  if (aweights) {
      std::tie(aweights_value, aweights_bdim) = unwrapTensorAtLevel(aweights.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, correction, fweights_value, fweights_bdim, aweights_value, aweights_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cudnn_affine_grid_generator', 'cudnn_affine_grid_generator_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_66_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_66_t,Tensor,const Tensor &, int64_t, int64_t, int64_t, int64_t>(
  batch_rule_66_t batch_rule,
  const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor theta_value;
  optional<int64_t> theta_bdim;
  std::tie(theta_value, theta_bdim) = unwrapTensorAtLevel(theta, cur_level);
  auto results = batch_rule(theta_value, theta_bdim, N, C, H, W);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cudnn_batch_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_67_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_67_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double>(
  batch_rule_67_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double exponential_average_factor, double epsilon
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, training, exponential_average_factor, epsilon);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level));
}

// ['cudnn_batch_norm_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_68_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_68_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, const Tensor &>(
  batch_rule_68_t batch_rule,
  const Tensor & input, const Tensor & grad_output, const Tensor & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_var, double epsilon, const Tensor & reserveSpace
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor reserveSpace_value;
  optional<int64_t> reserveSpace_bdim;
  std::tie(reserveSpace_value, reserveSpace_bdim) = unwrapTensorAtLevel(reserveSpace, cur_level);
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  optional<Tensor> save_mean_value;
  optional<int64_t> save_mean_bdim;
  if (save_mean) {
      std::tie(save_mean_value, save_mean_bdim) = unwrapTensorAtLevel(save_mean.value(), cur_level);
  }
  optional<Tensor> save_var_value;
  optional<int64_t> save_var_bdim;
  if (save_var) {
      std::tie(save_var_value, save_var_bdim) = unwrapTensorAtLevel(save_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, grad_output_value, grad_output_bdim, weight_value, weight_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, save_mean_value, save_mean_bdim, save_var_value, save_var_bdim, epsilon, reserveSpace_value, reserveSpace_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['cudnn_convolution']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_69_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_69_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool>(
  batch_rule_69_t batch_rule,
  const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cudnn_convolution_transpose']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_70_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_70_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool>(
  batch_rule_70_t batch_rule,
  const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cudnn_convolution_add_relu']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_71_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_71_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Scalar> &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_71_t batch_rule,
  const Tensor & self, const Tensor & weight, const Tensor & z, const c10::optional<Scalar> & alpha, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor z_value;
  optional<int64_t> z_bdim;
  std::tie(z_value, z_bdim) = unwrapTensorAtLevel(z, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, z_value, z_bdim, alpha, bias_value, bias_bdim, stride, padding, dilation, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cudnn_grid_sampler_backward', 'prelu_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_72_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_72_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &>(
  batch_rule_72_t batch_rule,
  const Tensor & self, const Tensor & grid, const Tensor & grad_output
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor grid_value;
  optional<int64_t> grid_bdim;
  std::tie(grid_value, grid_bdim) = unwrapTensorAtLevel(grid, cur_level);
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(self_value, self_bdim, grid_value, grid_bdim, grad_output_value, grad_output_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['cummax.dimname', 'cummin.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_73_t)(const Tensor &, c10::optional<int64_t>, Dimname);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_73_t,std::tuple<Tensor,Tensor>,const Tensor &, Dimname>(
  batch_rule_73_t batch_rule,
  const Tensor & self, Dimname dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['cumprod', 'cumsum', 'log_softmax.int', 'softmax.int', '_sparse_softmax.int', '_sparse_log_softmax.int', 'special_log_softmax', 'special_softmax']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_74_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_74_t,Tensor,const Tensor &, int64_t, c10::optional<ScalarType>>(
  batch_rule_74_t batch_rule,
  const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cumprod.dimname', 'cumsum.dimname', 'log_softmax.Dimname', 'softmax.Dimname', '_sparse_softmax.Dimname', '_sparse_log_softmax.Dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_75_t)(const Tensor &, c10::optional<int64_t>, Dimname, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_75_t,Tensor,const Tensor &, Dimname, c10::optional<ScalarType>>(
  batch_rule_75_t batch_rule,
  const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cumprod_backward', '_sparse_softmax_backward_data', '_sparse_log_softmax_backward_data']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_76_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_76_t,Tensor,const Tensor &, const Tensor &, int64_t, const Tensor &>(
  batch_rule_76_t batch_rule,
  const Tensor & grad, const Tensor & input, int64_t dim, const Tensor & output
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor output_value;
  optional<int64_t> output_bdim;
  std::tie(output_value, output_bdim) = unwrapTensorAtLevel(output, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim, dim, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cumulative_trapezoid.dx', 'trapezoid.dx']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_77_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_77_t,Tensor,const Tensor &, const Scalar &, int64_t>(
  batch_rule_77_t batch_rule,
  const Tensor & y, const Scalar & dx, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor y_value;
  optional<int64_t> y_bdim;
  std::tie(y_value, y_bdim) = unwrapTensorAtLevel(y, cur_level);
  auto results = batch_rule(y_value, y_bdim, dx, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['ctc_loss.IntList']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_78_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_78_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, int64_t, bool>(
  batch_rule_78_t batch_rule,
  const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor log_probs_value;
  optional<int64_t> log_probs_bdim;
  std::tie(log_probs_value, log_probs_bdim) = unwrapTensorAtLevel(log_probs, cur_level);
  Tensor targets_value;
  optional<int64_t> targets_bdim;
  std::tie(targets_value, targets_bdim) = unwrapTensorAtLevel(targets, cur_level);
  auto results = batch_rule(log_probs_value, log_probs_bdim, targets_value, targets_bdim, input_lengths, target_lengths, blank, reduction, zero_infinity);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['ctc_loss.Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_79_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_79_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(
  batch_rule_79_t batch_rule,
  const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor log_probs_value;
  optional<int64_t> log_probs_bdim;
  std::tie(log_probs_value, log_probs_bdim) = unwrapTensorAtLevel(log_probs, cur_level);
  Tensor targets_value;
  optional<int64_t> targets_bdim;
  std::tie(targets_value, targets_bdim) = unwrapTensorAtLevel(targets, cur_level);
  Tensor input_lengths_value;
  optional<int64_t> input_lengths_bdim;
  std::tie(input_lengths_value, input_lengths_bdim) = unwrapTensorAtLevel(input_lengths, cur_level);
  Tensor target_lengths_value;
  optional<int64_t> target_lengths_bdim;
  std::tie(target_lengths_value, target_lengths_bdim) = unwrapTensorAtLevel(target_lengths, cur_level);
  auto results = batch_rule(log_probs_value, log_probs_bdim, targets_value, targets_bdim, input_lengths_value, input_lengths_bdim, target_lengths_value, target_lengths_bdim, blank, reduction, zero_infinity);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_ctc_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_80_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_80_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool>(
  batch_rule_80_t batch_rule,
  const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor log_probs_value;
  optional<int64_t> log_probs_bdim;
  std::tie(log_probs_value, log_probs_bdim) = unwrapTensorAtLevel(log_probs, cur_level);
  Tensor targets_value;
  optional<int64_t> targets_bdim;
  std::tie(targets_value, targets_bdim) = unwrapTensorAtLevel(targets, cur_level);
  auto results = batch_rule(log_probs_value, log_probs_bdim, targets_value, targets_bdim, input_lengths, target_lengths, blank, zero_infinity);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_ctc_loss_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_81_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_81_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, int64_t, bool>(
  batch_rule_81_t batch_rule,
  const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor log_probs_value;
  optional<int64_t> log_probs_bdim;
  std::tie(log_probs_value, log_probs_bdim) = unwrapTensorAtLevel(log_probs, cur_level);
  Tensor targets_value;
  optional<int64_t> targets_bdim;
  std::tie(targets_value, targets_bdim) = unwrapTensorAtLevel(targets, cur_level);
  Tensor neg_log_likelihood_value;
  optional<int64_t> neg_log_likelihood_bdim;
  std::tie(neg_log_likelihood_value, neg_log_likelihood_bdim) = unwrapTensorAtLevel(neg_log_likelihood, cur_level);
  Tensor log_alpha_value;
  optional<int64_t> log_alpha_bdim;
  std::tie(log_alpha_value, log_alpha_bdim) = unwrapTensorAtLevel(log_alpha, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, log_probs_value, log_probs_bdim, targets_value, targets_bdim, input_lengths, target_lengths, neg_log_likelihood_value, neg_log_likelihood_bdim, log_alpha_value, log_alpha_bdim, blank, zero_infinity);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['diag_embed', 'diagonal', 'linalg_diagonal', 'narrow_copy', 'narrow', 'unfold', '_remove_batch_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_82_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_82_t,Tensor,const Tensor &, int64_t, int64_t, int64_t>(
  batch_rule_82_t batch_rule,
  const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['diagonal.Dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_83_t)(const Tensor &, c10::optional<int64_t>, Dimname, Dimname, Dimname, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_83_t,Tensor,const Tensor &, Dimname, Dimname, Dimname, int64_t>(
  batch_rule_83_t batch_rule,
  const Tensor & self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, outdim, dim1, dim2, offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['diagonal_backward', 'unfold_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_84_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_84_t,Tensor,const Tensor &, IntArrayRef, int64_t, int64_t, int64_t>(
  batch_rule_84_t batch_rule,
  const Tensor & grad_output, IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['diff']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_85_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_85_t,Tensor,const Tensor &, int64_t, int64_t, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_85_t batch_rule,
  const Tensor & self, int64_t n, int64_t dim, const c10::optional<Tensor> & prepend, const c10::optional<Tensor> & append
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> prepend_value;
  optional<int64_t> prepend_bdim;
  if (prepend) {
      std::tie(prepend_value, prepend_bdim) = unwrapTensorAtLevel(prepend.value(), cur_level);
  }
  optional<Tensor> append_value;
  optional<int64_t> append_bdim;
  if (append) {
      std::tie(append_value, append_bdim) = unwrapTensorAtLevel(append.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, n, dim, prepend_value, prepend_bdim, append_value, append_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gradient.scalarint']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_86_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, c10::optional<int64_t>, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_86_t,::std::vector<Tensor>,const Tensor &, const c10::optional<Scalar> &, c10::optional<int64_t>, int64_t>(
  batch_rule_86_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & spacing, c10::optional<int64_t> dim, int64_t edge_order
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gradient.scalararray']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_87_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, IntArrayRef, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_87_t,::std::vector<Tensor>,const Tensor &, const Scalar &, IntArrayRef, int64_t>(
  batch_rule_87_t batch_rule,
  const Tensor & self, const Scalar & spacing, IntArrayRef dim, int64_t edge_order
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gradient.scalarrayint']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_88_t)(const Tensor &, c10::optional<int64_t>, ArrayRef<Scalar>, c10::optional<int64_t>, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_88_t,::std::vector<Tensor>,const Tensor &, ArrayRef<Scalar>, c10::optional<int64_t>, int64_t>(
  batch_rule_88_t batch_rule,
  const Tensor & self, ArrayRef<Scalar> spacing, c10::optional<int64_t> dim, int64_t edge_order
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gradient.scalarrayarray']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_89_t)(const Tensor &, c10::optional<int64_t>, ArrayRef<Scalar>, IntArrayRef, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_89_t,::std::vector<Tensor>,const Tensor &, ArrayRef<Scalar>, IntArrayRef, int64_t>(
  batch_rule_89_t batch_rule,
  const Tensor & self, ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, spacing, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['div.Tensor_mode', 'divide.Tensor_mode']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_90_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_90_t,Tensor,const Tensor &, const Tensor &, c10::optional<c10::string_view>>(
  batch_rule_90_t batch_rule,
  const Tensor & self, const Tensor & other, c10::optional<c10::string_view> rounding_mode
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, rounding_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['div.Scalar_mode', 'divide.Scalar_mode']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_91_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_91_t,Tensor,const Tensor &, const Scalar &, c10::optional<c10::string_view>>(
  batch_rule_91_t batch_rule,
  const Tensor & self, const Scalar & other, c10::optional<c10::string_view> rounding_mode
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, rounding_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['embedding']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_92_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_92_t,Tensor,const Tensor &, const Tensor &, int64_t, bool, bool>(
  batch_rule_92_t batch_rule,
  const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(weight_value, weight_bdim, indices_value, indices_bdim, padding_idx, scale_grad_by_freq, sparse);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['embedding_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_93_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_93_t,Tensor,const Tensor &, const Tensor &, int64_t, int64_t, bool, bool>(
  batch_rule_93_t batch_rule,
  const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, indices_value, indices_bdim, num_weights, padding_idx, scale_grad_by_freq, sparse);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['embedding_dense_backward', 'embedding_sparse_backward', 'grid_sampler', 'grid_sampler_2d', '_grid_sampler_2d_cpu_fallback', 'grid_sampler_3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_94_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_94_t,Tensor,const Tensor &, const Tensor &, int64_t, int64_t, bool>(
  batch_rule_94_t batch_rule,
  const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, indices_value, indices_bdim, num_weights, padding_idx, scale_grad_by_freq);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_embedding_bag_forward_only', '_embedding_bag']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_95_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, int64_t);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_95_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const c10::optional<Tensor> &, bool, int64_t>(
  batch_rule_95_t batch_rule,
  const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor offsets_value;
  optional<int64_t> offsets_bdim;
  std::tie(offsets_value, offsets_bdim) = unwrapTensorAtLevel(offsets, cur_level);
  optional<Tensor> per_sample_weights_value;
  optional<int64_t> per_sample_weights_bdim;
  if (per_sample_weights) {
      std::tie(per_sample_weights_value, per_sample_weights_bdim) = unwrapTensorAtLevel(per_sample_weights.value(), cur_level);
  }
  auto results = batch_rule(weight_value, weight_bdim, indices_value, indices_bdim, offsets_value, offsets_bdim, scale_grad_by_freq, mode, sparse, per_sample_weights_value, per_sample_weights_bdim, include_last_offset, padding_idx);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level));
}

// ['_rowwise_prune']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_96_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, ScalarType);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_96_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, ScalarType>(
  batch_rule_96_t batch_rule,
  const Tensor & weight, const Tensor & mask, ScalarType compressed_indices_dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor mask_value;
  optional<int64_t> mask_bdim;
  std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(weight_value, weight_bdim, mask_value, mask_bdim, compressed_indices_dtype);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['embedding_bag']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_97_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_97_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const c10::optional<Tensor> &, bool>(
  batch_rule_97_t batch_rule,
  const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, bool include_last_offset
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor offsets_value;
  optional<int64_t> offsets_bdim;
  std::tie(offsets_value, offsets_bdim) = unwrapTensorAtLevel(offsets, cur_level);
  optional<Tensor> per_sample_weights_value;
  optional<int64_t> per_sample_weights_bdim;
  if (per_sample_weights) {
      std::tie(per_sample_weights_value, per_sample_weights_bdim) = unwrapTensorAtLevel(per_sample_weights.value(), cur_level);
  }
  auto results = batch_rule(weight_value, weight_bdim, indices_value, indices_bdim, offsets_value, offsets_bdim, scale_grad_by_freq, mode, sparse, per_sample_weights_value, per_sample_weights_bdim, include_last_offset);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level));
}

// ['embedding_bag.padding_idx']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_98_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_98_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const c10::optional<Tensor> &, bool, c10::optional<int64_t>>(
  batch_rule_98_t batch_rule,
  const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, bool include_last_offset, c10::optional<int64_t> padding_idx
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor offsets_value;
  optional<int64_t> offsets_bdim;
  std::tie(offsets_value, offsets_bdim) = unwrapTensorAtLevel(offsets, cur_level);
  optional<Tensor> per_sample_weights_value;
  optional<int64_t> per_sample_weights_bdim;
  if (per_sample_weights) {
      std::tie(per_sample_weights_value, per_sample_weights_bdim) = unwrapTensorAtLevel(per_sample_weights.value(), cur_level);
  }
  auto results = batch_rule(weight_value, weight_bdim, indices_value, indices_bdim, offsets_value, offsets_bdim, scale_grad_by_freq, mode, sparse, per_sample_weights_value, per_sample_weights_bdim, include_last_offset, padding_idx);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level));
}

// ['_embedding_bag_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_99_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_99_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const c10::optional<Tensor> &, int64_t>(
  batch_rule_99_t batch_rule,
  const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, int64_t padding_idx
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor offsets_value;
  optional<int64_t> offsets_bdim;
  std::tie(offsets_value, offsets_bdim) = unwrapTensorAtLevel(offsets, cur_level);
  Tensor offset2bag_value;
  optional<int64_t> offset2bag_bdim;
  std::tie(offset2bag_value, offset2bag_bdim) = unwrapTensorAtLevel(offset2bag, cur_level);
  Tensor bag_size_value;
  optional<int64_t> bag_size_bdim;
  std::tie(bag_size_value, bag_size_bdim) = unwrapTensorAtLevel(bag_size, cur_level);
  Tensor maximum_indices_value;
  optional<int64_t> maximum_indices_bdim;
  std::tie(maximum_indices_value, maximum_indices_bdim) = unwrapTensorAtLevel(maximum_indices, cur_level);
  optional<Tensor> per_sample_weights_value;
  optional<int64_t> per_sample_weights_bdim;
  if (per_sample_weights) {
      std::tie(per_sample_weights_value, per_sample_weights_bdim) = unwrapTensorAtLevel(per_sample_weights.value(), cur_level);
  }
  auto results = batch_rule(grad_value, grad_bdim, indices_value, indices_bdim, offsets_value, offsets_bdim, offset2bag_value, offset2bag_bdim, bag_size_value, bag_size_bdim, maximum_indices_value, maximum_indices_bdim, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights_value, per_sample_weights_bdim, padding_idx);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_embedding_bag_sparse_backward', '_embedding_bag_dense_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_100_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_100_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const c10::optional<Tensor> &, int64_t>(
  batch_rule_100_t batch_rule,
  const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor> & per_sample_weights, int64_t padding_idx
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor offsets_value;
  optional<int64_t> offsets_bdim;
  std::tie(offsets_value, offsets_bdim) = unwrapTensorAtLevel(offsets, cur_level);
  Tensor offset2bag_value;
  optional<int64_t> offset2bag_bdim;
  std::tie(offset2bag_value, offset2bag_bdim) = unwrapTensorAtLevel(offset2bag, cur_level);
  Tensor bag_size_value;
  optional<int64_t> bag_size_bdim;
  std::tie(bag_size_value, bag_size_bdim) = unwrapTensorAtLevel(bag_size, cur_level);
  optional<Tensor> per_sample_weights_value;
  optional<int64_t> per_sample_weights_bdim;
  if (per_sample_weights) {
      std::tie(per_sample_weights_value, per_sample_weights_bdim) = unwrapTensorAtLevel(per_sample_weights.value(), cur_level);
  }
  auto results = batch_rule(grad_value, grad_bdim, indices_value, indices_bdim, offsets_value, offsets_bdim, offset2bag_value, offset2bag_bdim, bag_size_value, bag_size_bdim, num_weights, scale_grad_by_freq, mode, per_sample_weights_value, per_sample_weights_bdim, padding_idx);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_embedding_bag_per_sample_weights_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_101_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_101_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(
  batch_rule_101_t batch_rule,
  const Tensor & grad, const Tensor & weight, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, int64_t mode, int64_t padding_idx
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor offsets_value;
  optional<int64_t> offsets_bdim;
  std::tie(offsets_value, offsets_bdim) = unwrapTensorAtLevel(offsets, cur_level);
  Tensor offset2bag_value;
  optional<int64_t> offset2bag_bdim;
  std::tie(offset2bag_value, offset2bag_bdim) = unwrapTensorAtLevel(offset2bag, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, weight_value, weight_bdim, indices_value, indices_bdim, offsets_value, offsets_bdim, offset2bag_value, offset2bag_bdim, mode, padding_idx);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['new_empty', 'new_zeros', 'new_ones']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_102_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_102_t,Tensor,const Tensor &, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_102_t batch_rule,
  const Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['new_empty_strided']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_103_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_103_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_103_t batch_rule,
  const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['new_full']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_104_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_104_t,Tensor,const Tensor &, IntArrayRef, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_104_t batch_rule,
  const Tensor & self, IntArrayRef size, const Scalar & fill_value, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, fill_value, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_empty_per_channel_affine_quantized']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_105_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_105_t,Tensor,IntArrayRef, const Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_105_t batch_rule,
  IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor scales_value;
  optional<int64_t> scales_bdim;
  std::tie(scales_value, scales_bdim) = unwrapTensorAtLevel(scales, cur_level);
  Tensor zero_points_value;
  optional<int64_t> zero_points_bdim;
  std::tie(zero_points_value, zero_points_bdim) = unwrapTensorAtLevel(zero_points, cur_level);
  auto results = batch_rule(size, scales_value, scales_bdim, zero_points_value, zero_points_bdim, axis, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['empty_quantized']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_106_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_106_t,Tensor,IntArrayRef, const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_106_t batch_rule,
  IntArrayRef size, const Tensor & qtensor, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor qtensor_value;
  optional<int64_t> qtensor_bdim;
  std::tie(qtensor_value, qtensor_bdim) = unwrapTensorAtLevel(qtensor, cur_level);
  auto results = batch_rule(size, qtensor_value, qtensor_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['empty_like', 'ones_like', 'rand_like', 'randn_like', 'zeros_like']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_107_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_107_t,Tensor,const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_107_t batch_rule,
  const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['flatten.using_ints', 'fbgemm_pack_quantized_matrix.KN', 'movedim.int', 'moveaxis.int', 'select.int', 'transpose.int', '_mkldnn_transpose', 'norm_except_dim', 'swapaxes', 'swapdims', '_add_batch_dim', '_test_ambiguous_defaults.a']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_108_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_108_t,Tensor,const Tensor &, int64_t, int64_t>(
  batch_rule_108_t batch_rule,
  const Tensor & self, int64_t start_dim, int64_t end_dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, start_dim, end_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['flatten.named_out_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_109_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_109_t,Tensor,const Tensor &, int64_t, int64_t, Dimname>(
  batch_rule_109_t batch_rule,
  const Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, start_dim, end_dim, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['flatten.using_names']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_110_t)(const Tensor &, c10::optional<int64_t>, Dimname, Dimname, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_110_t,Tensor,const Tensor &, Dimname, Dimname, Dimname>(
  batch_rule_110_t batch_rule,
  const Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, start_dim, end_dim, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['flatten.DimnameList']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_111_t)(const Tensor &, c10::optional<int64_t>, DimnameList, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_111_t,Tensor,const Tensor &, DimnameList, Dimname>(
  batch_rule_111_t batch_rule,
  const Tensor & self, DimnameList dims, Dimname out_dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['unflatten.int']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_112_t)(const Tensor &, c10::optional<int64_t>, int64_t, IntArrayRef, c10::optional<DimnameList>);
template <>
Tensor lowerToNextLayer<batch_rule_112_t,Tensor,const Tensor &, int64_t, IntArrayRef, c10::optional<DimnameList>>(
  batch_rule_112_t batch_rule,
  const Tensor & self, int64_t dim, IntArrayRef sizes, c10::optional<DimnameList> names
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, sizes, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['unflatten.Dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_113_t)(const Tensor &, c10::optional<int64_t>, Dimname, IntArrayRef, DimnameList);
template <>
Tensor lowerToNextLayer<batch_rule_113_t,Tensor,const Tensor &, Dimname, IntArrayRef, DimnameList>(
  batch_rule_113_t batch_rule,
  const Tensor & self, Dimname dim, IntArrayRef sizes, DimnameList names
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, sizes, names);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['full_like']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_114_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_114_t,Tensor,const Tensor &, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_114_t batch_rule,
  const Tensor & self, const Scalar & fill_value, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, fill_value, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['grid_sampler_2d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_115_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool, ::std::array<bool,2>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_115_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool, ::std::array<bool,2>>(
  batch_rule_115_t batch_rule,
  const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grid_value;
  optional<int64_t> grid_bdim;
  std::tie(grid_value, grid_bdim) = unwrapTensorAtLevel(grid, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_value, input_bdim, grid_value, grid_bdim, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_grid_sampler_2d_cpu_fallback_backward', 'grid_sampler_3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_116_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_116_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(
  batch_rule_116_t batch_rule,
  const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grid_value;
  optional<int64_t> grid_bdim;
  std::tie(grid_value, grid_bdim) = unwrapTensorAtLevel(grid, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_value, input_bdim, grid_value, grid_bdim, interpolation_mode, padding_mode, align_corners);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['hinge_embedding_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_117_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_117_t,Tensor,const Tensor &, const Tensor &, double, int64_t>(
  batch_rule_117_t batch_rule,
  const Tensor & self, const Tensor & target, double margin, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, margin, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['group_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_118_t)(const Tensor &, c10::optional<int64_t>, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_118_t,Tensor,const Tensor &, int64_t, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, bool>(
  batch_rule_118_t batch_rule,
  const Tensor & input, int64_t num_groups, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps, bool cudnn_enabled
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, num_groups, weight_value, weight_bdim, bias_value, bias_bdim, eps, cudnn_enabled);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['native_group_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_119_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t, int64_t, int64_t, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_119_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t, int64_t, int64_t, int64_t, double>(
  batch_rule_119_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, N, C, HxW, group, eps);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['native_group_norm_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_120_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t, int64_t, int64_t, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_120_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t, int64_t, int64_t, ::std::array<bool,3>>(
  batch_rule_120_t batch_rule,
  const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const c10::optional<Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_out_value;
  optional<int64_t> grad_out_bdim;
  std::tie(grad_out_value, grad_out_bdim) = unwrapTensorAtLevel(grad_out, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor rstd_value;
  optional<int64_t> rstd_bdim;
  std::tie(rstd_value, rstd_bdim) = unwrapTensorAtLevel(rstd, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(grad_out_value, grad_out_bdim, input_value, input_bdim, mean_value, mean_bdim, rstd_value, rstd_bdim, weight_value, weight_bdim, N, C, HxW, group, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_fft_r2c', '_fft_c2c']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_121_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_121_t,Tensor,const Tensor &, IntArrayRef, int64_t, bool>(
  batch_rule_121_t batch_rule,
  const Tensor & self, IntArrayRef dim, int64_t normalization, bool onesided
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, normalization, onesided);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_fft_c2r', 'select_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_122_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_122_t,Tensor,const Tensor &, IntArrayRef, int64_t, int64_t>(
  batch_rule_122_t batch_rule,
  const Tensor & self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, normalization, last_dim_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index.Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_123_t)(const Tensor &, c10::optional<int64_t>, const c10::List<c10::optional<Tensor>> &);
template <>
Tensor lowerToNextLayer<batch_rule_123_t,Tensor,const Tensor &, const c10::List<c10::optional<Tensor>> &>(
  batch_rule_123_t batch_rule,
  const Tensor & self, const c10::List<c10::optional<Tensor>> & indices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_copy', 'index_fill.int_Tensor', 'scatter.src', 'scatter_add', '_gather_sparse_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_124_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_124_t,Tensor,const Tensor &, int64_t, const Tensor &, const Tensor &>(
  batch_rule_124_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  Tensor source_value;
  optional<int64_t> source_bdim;
  std::tie(source_value, source_bdim) = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_copy.dimname', 'index_fill.Dimname_Tensor', 'scatter.dimname_src', 'scatter_add.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_125_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_125_t,Tensor,const Tensor &, Dimname, const Tensor &, const Tensor &>(
  batch_rule_125_t batch_rule,
  const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  Tensor source_value;
  optional<int64_t> source_bdim;
  std::tie(source_value, source_bdim) = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_put']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_126_t)(const Tensor &, c10::optional<int64_t>, const c10::List<c10::optional<Tensor>> &, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_126_t,Tensor,const Tensor &, const c10::List<c10::optional<Tensor>> &, const Tensor &, bool>(
  batch_rule_126_t batch_rule,
  const Tensor & self, const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices, values_value, values_bdim, accumulate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['isclose', 'pairwise_distance']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_127_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_127_t,Tensor,const Tensor &, const Tensor &, double, double, bool>(
  batch_rule_127_t batch_rule,
  const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, rtol, atol, equal_nan);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['isin.Tensor_Tensor', 'bucketize.Tensor', '_convert_indices_from_csr_to_coo']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_128_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_128_t,Tensor,const Tensor &, const Tensor &, bool, bool>(
  batch_rule_128_t batch_rule,
  const Tensor & elements, const Tensor & test_elements, bool assume_unique, bool invert
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor elements_value;
  optional<int64_t> elements_bdim;
  std::tie(elements_value, elements_bdim) = unwrapTensorAtLevel(elements, cur_level);
  Tensor test_elements_value;
  optional<int64_t> test_elements_bdim;
  std::tie(test_elements_value, test_elements_bdim) = unwrapTensorAtLevel(test_elements, cur_level);
  auto results = batch_rule(elements_value, elements_bdim, test_elements_value, test_elements_bdim, assume_unique, invert);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['isin.Tensor_Scalar']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_129_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_129_t,Tensor,const Tensor &, const Scalar &, bool, bool>(
  batch_rule_129_t batch_rule,
  const Tensor & elements, const Scalar & test_element, bool assume_unique, bool invert
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor elements_value;
  optional<int64_t> elements_bdim;
  std::tie(elements_value, elements_bdim) = unwrapTensorAtLevel(elements, cur_level);
  auto results = batch_rule(elements_value, elements_bdim, test_element, assume_unique, invert);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['isin.Scalar_Tensor', 'bucketize.Scalar']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_130_t)(const Scalar &, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_130_t,Tensor,const Scalar &, const Tensor &, bool, bool>(
  batch_rule_130_t batch_rule,
  const Scalar & element, const Tensor & test_elements, bool assume_unique, bool invert
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor test_elements_value;
  optional<int64_t> test_elements_bdim;
  std::tie(test_elements_value, test_elements_bdim) = unwrapTensorAtLevel(test_elements, cur_level);
  auto results = batch_rule(element, test_elements_value, test_elements_bdim, assume_unique, invert);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['kl_div']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_131_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_131_t,Tensor,const Tensor &, const Tensor &, int64_t, bool>(
  batch_rule_131_t batch_rule,
  const Tensor & self, const Tensor & target, int64_t reduction, bool log_target
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction, log_target);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['kl_div_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_132_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_132_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, bool>(
  batch_rule_132_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction, log_target);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['kthvalue']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_133_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_133_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, int64_t, bool>(
  batch_rule_133_t batch_rule,
  const Tensor & self, int64_t k, int64_t dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['kthvalue.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_134_t)(const Tensor &, c10::optional<int64_t>, int64_t, Dimname, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_134_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, Dimname, bool>(
  batch_rule_134_t batch_rule,
  const Tensor & self, int64_t k, Dimname dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['layer_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_135_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_135_t,Tensor,const Tensor &, IntArrayRef, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, bool>(
  batch_rule_135_t batch_rule,
  const Tensor & input, IntArrayRef normalized_shape, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps, bool cudnn_enable
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, normalized_shape, weight_value, weight_bdim, bias_value, bias_bdim, eps, cudnn_enable);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['native_layer_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_136_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_136_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, IntArrayRef, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double>(
  batch_rule_136_t batch_rule,
  const Tensor & input, IntArrayRef normalized_shape, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, normalized_shape, weight_value, weight_bdim, bias_value, bias_bdim, eps);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_native_multi_head_self_attention']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_137_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_137_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const c10::optional<Tensor> &>(
  batch_rule_137_t batch_rule,
  const Tensor & query, const Tensor & qkv_weight, const Tensor & qkv_bias, const Tensor & proj_weight, const Tensor & proj_bias, int64_t num_head, const c10::optional<Tensor> & mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor query_value;
  optional<int64_t> query_bdim;
  std::tie(query_value, query_bdim) = unwrapTensorAtLevel(query, cur_level);
  Tensor qkv_weight_value;
  optional<int64_t> qkv_weight_bdim;
  std::tie(qkv_weight_value, qkv_weight_bdim) = unwrapTensorAtLevel(qkv_weight, cur_level);
  Tensor qkv_bias_value;
  optional<int64_t> qkv_bias_bdim;
  std::tie(qkv_bias_value, qkv_bias_bdim) = unwrapTensorAtLevel(qkv_bias, cur_level);
  Tensor proj_weight_value;
  optional<int64_t> proj_weight_bdim;
  std::tie(proj_weight_value, proj_weight_bdim) = unwrapTensorAtLevel(proj_weight, cur_level);
  Tensor proj_bias_value;
  optional<int64_t> proj_bias_bdim;
  std::tie(proj_bias_value, proj_bias_bdim) = unwrapTensorAtLevel(proj_bias, cur_level);
  optional<Tensor> mask_value;
  optional<int64_t> mask_bdim;
  if (mask) {
      std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask.value(), cur_level);
  }
  auto results = batch_rule(query_value, query_bdim, qkv_weight_value, qkv_weight_bdim, qkv_bias_value, qkv_bias_bdim, proj_weight_value, proj_weight_bdim, proj_bias_value, proj_bias_bdim, num_head, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['native_layer_norm_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_138_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_138_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, IntArrayRef, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, ::std::array<bool,3>>(
  batch_rule_138_t batch_rule,
  const Tensor & grad_out, const Tensor & input, IntArrayRef normalized_shape, const Tensor & mean, const Tensor & rstd, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_out_value;
  optional<int64_t> grad_out_bdim;
  std::tie(grad_out_value, grad_out_bdim) = unwrapTensorAtLevel(grad_out, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor rstd_value;
  optional<int64_t> rstd_bdim;
  std::tie(rstd_value, rstd_bdim) = unwrapTensorAtLevel(rstd, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(grad_out_value, grad_out_bdim, input_value, input_bdim, normalized_shape, mean_value, mean_bdim, rstd_value, rstd_bdim, weight_value, weight_bdim, bias_value, bias_bdim, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['nan_to_num']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_139_t)(const Tensor &, c10::optional<int64_t>, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_139_t,Tensor,const Tensor &, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_139_t batch_rule,
  const Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, nan, posinf, neginf);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linear', 'mkldnn_linear']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_140_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_140_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &>(
  batch_rule_140_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['mkldnn_linear_backward_input']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_141_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_141_t,Tensor,IntArrayRef, const Tensor &, const Tensor &>(
  batch_rule_141_t batch_rule,
  IntArrayRef input_size, const Tensor & grad_output, const Tensor & weight
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(input_size, grad_output_value, grad_output_bdim, weight_value, weight_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['mkldnn_linear_backward_weights']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_142_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_142_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool>(
  batch_rule_142_t batch_rule,
  const Tensor & grad_output, const Tensor & input, const Tensor & weight, bool bias_defined
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_value, input_bdim, weight_value, weight_bdim, bias_defined);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['mkldnn_linear_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_143_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_143_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, ::std::array<bool,3>>(
  batch_rule_143_t batch_rule,
  const Tensor & self, const Tensor & grad_output, const Tensor & weight, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(self_value, self_bdim, grad_output_value, grad_output_bdim, weight_value, weight_bdim, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['fbgemm_linear_int8_weight_fp32_activation', 'fbgemm_linear_int8_weight']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_144_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_144_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, const Tensor &>(
  batch_rule_144_t batch_rule,
  const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, const Scalar & weight_scale, const Scalar & weight_zero_point, const Tensor & bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  Tensor packed_value;
  optional<int64_t> packed_bdim;
  std::tie(packed_value, packed_bdim) = unwrapTensorAtLevel(packed, cur_level);
  Tensor col_offsets_value;
  optional<int64_t> col_offsets_bdim;
  std::tie(col_offsets_value, col_offsets_bdim) = unwrapTensorAtLevel(col_offsets, cur_level);
  Tensor bias_value;
  optional<int64_t> bias_bdim;
  std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias, cur_level);
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, packed_value, packed_bdim, col_offsets_value, col_offsets_bdim, weight_scale, weight_zero_point, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fbgemm_linear_quantize_weight']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,double,int64_t> (*batch_rule_145_t)(const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,double,int64_t> lowerToNextLayer<batch_rule_145_t,std::tuple<Tensor,Tensor,double,int64_t>,const Tensor &>(
  batch_rule_145_t batch_rule,
  const Tensor & input
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), std::get<4>(results), std::get<5>(results));
}

// ['fbgemm_linear_fp16_weight_fp32_activation', 'fbgemm_linear_fp16_weight', 'where.self', '_s_where', '_dirichlet_grad', 'masked_fill.Tensor', 'masked_scatter', 'masked_select_backward', 'lu_solve', 'lerp.Tensor', 'log_sigmoid_backward', 'adaptive_max_pool2d_backward', 'adaptive_max_pool3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_146_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_146_t,Tensor,const Tensor &, const Tensor &, const Tensor &>(
  batch_rule_146_t batch_rule,
  const Tensor & input, const Tensor & packed_weight, const Tensor & bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor packed_weight_value;
  optional<int64_t> packed_weight_bdim;
  std::tie(packed_weight_value, packed_weight_bdim) = unwrapTensorAtLevel(packed_weight, cur_level);
  Tensor bias_value;
  optional<int64_t> bias_bdim;
  std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias, cur_level);
  auto results = batch_rule(input_value, input_bdim, packed_weight_value, packed_weight_bdim, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['xlogy.Scalar_Self', 'bitwise_left_shift.Scalar_Tensor', 'bitwise_right_shift.Scalar_Tensor', 'remainder.Scalar_Tensor', 'pow.Scalar', 'float_power.Scalar', 'special_xlog1py.self_scalar', 'special_xlogy.self_scalar', 'special_zeta.self_scalar']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_147_t)(const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_147_t,Tensor,const Scalar &, const Tensor &>(
  batch_rule_147_t batch_rule,
  const Scalar & self, const Tensor & other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_log_softmax_backward_data', '_softmax_backward_data']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_148_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_148_t,Tensor,const Tensor &, const Tensor &, int64_t, ScalarType>(
  batch_rule_148_t batch_rule,
  const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType input_dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor output_value;
  optional<int64_t> output_bdim;
  std::tie(output_value, output_bdim) = unwrapTensorAtLevel(output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim, dim, input_dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['logcumsumexp.dimname', 'squeeze.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_149_t)(const Tensor &, c10::optional<int64_t>, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_149_t,Tensor,const Tensor &, Dimname>(
  batch_rule_149_t batch_rule,
  const Tensor & self, Dimname dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['logsumexp.names']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_150_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool);
template <>
Tensor lowerToNextLayer<batch_rule_150_t,Tensor,const Tensor &, DimnameList, bool>(
  batch_rule_150_t batch_rule,
  const Tensor & self, DimnameList dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_aminmax', 'slogdet', 'frexp.Tensor', 'geqrf', 'log_sigmoid_forward', 'linalg_slogdet', 'linalg_eig']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_151_t)(const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_151_t,std::tuple<Tensor,Tensor>,const Tensor &>(
  batch_rule_151_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_aminmax.dim', 'max.dim', 'median.dim', 'nanmedian.dim', 'min.dim', 'mode', 'sort']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_152_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_152_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, bool>(
  batch_rule_152_t batch_rule,
  const Tensor & self, int64_t dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['aminmax']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_153_t)(const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_153_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::optional<int64_t>, bool>(
  batch_rule_153_t batch_rule,
  const Tensor & self, c10::optional<int64_t> dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['max.names_dim', 'median.names_dim', 'nanmedian.names_dim', 'min.names_dim', 'mode.dimname', 'sort.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_154_t)(const Tensor &, c10::optional<int64_t>, Dimname, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_154_t,std::tuple<Tensor,Tensor>,const Tensor &, Dimname, bool>(
  batch_rule_154_t batch_rule,
  const Tensor & self, Dimname dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['value_selecting_reduction_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_155_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_155_t,Tensor,const Tensor &, int64_t, const Tensor &, IntArrayRef, bool>(
  batch_rule_155_t batch_rule,
  const Tensor & grad, int64_t dim, const Tensor & indices, IntArrayRef sizes, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, dim, indices_value, indices_bdim, sizes, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['max_pool1d_with_indices', 'max_pool2d_with_indices', 'max_pool3d_with_indices']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_156_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_156_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(
  batch_rule_156_t batch_rule,
  const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, stride, padding, dilation, ceil_mode);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['max_pool1d', 'max_pool2d', 'mkldnn_max_pool2d', 'mkldnn_max_pool3d', 'quantized_max_pool1d', 'quantized_max_pool2d', 'max_pool3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_157_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_157_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(
  batch_rule_157_t batch_rule,
  const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, stride, padding, dilation, ceil_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['mkldnn_max_pool2d_backward', 'mkldnn_max_pool3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_158_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_158_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(
  batch_rule_158_t batch_rule,
  const Tensor & grad_output, const Tensor & output, const Tensor & input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor output_value;
  optional<int64_t> output_bdim;
  std::tie(output_value, output_bdim) = unwrapTensorAtLevel(output, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim, input_value, input_bdim, kernel_size, stride, padding, dilation, ceil_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['mean', 'sum', 'nansum', 'prod', 'to_dense', 'to_mkldnn']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_159_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_159_t,Tensor,const Tensor &, c10::optional<ScalarType>>(
  batch_rule_159_t batch_rule,
  const Tensor & self, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['mean.dim', 'nanmean', 'sum.dim_IntList', 'nansum.dim_IntList']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_160_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_160_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_160_t batch_rule,
  const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['mean.names_dim', 'sum.dim_DimnameList']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_161_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_161_t,Tensor,const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(
  batch_rule_161_t batch_rule,
  const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['miopen_batch_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_162_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_162_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double>(
  batch_rule_162_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double exponential_average_factor, double epsilon
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, training, exponential_average_factor, epsilon);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['miopen_batch_norm_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_163_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_163_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double>(
  batch_rule_163_t batch_rule,
  const Tensor & input, const Tensor & grad_output, const Tensor & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_var, double epsilon
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  optional<Tensor> save_mean_value;
  optional<int64_t> save_mean_bdim;
  if (save_mean) {
      std::tie(save_mean_value, save_mean_bdim) = unwrapTensorAtLevel(save_mean.value(), cur_level);
  }
  optional<Tensor> save_var_value;
  optional<int64_t> save_var_bdim;
  if (save_var) {
      std::tie(save_var_value, save_var_bdim) = unwrapTensorAtLevel(save_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, grad_output_value, grad_output_bdim, weight_value, weight_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, save_mean_value, save_mean_bdim, save_var_value, save_var_bdim, epsilon);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['miopen_convolution', 'miopen_depthwise_convolution']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_164_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_164_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_164_t batch_rule,
  const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim, padding, stride, dilation, groups, benchmark, deterministic);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['miopen_convolution_transpose']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_165_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_165_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_165_t batch_rule,
  const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['narrow.Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_166_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_166_t,Tensor,const Tensor &, int64_t, const Tensor &, int64_t>(
  batch_rule_166_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & start, int64_t length
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor start_value;
  optional<int64_t> start_bdim;
  std::tie(start_value, start_bdim) = unwrapTensorAtLevel(start, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start_value, start_bdim, length);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['native_batch_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_167_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_167_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double>(
  batch_rule_167_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, training, momentum, eps);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['batch_norm_stats']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_168_t)(const Tensor &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_168_t,std::tuple<Tensor,Tensor>,const Tensor &, double>(
  batch_rule_168_t batch_rule,
  const Tensor & input, double eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, eps);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['batch_norm_elemt']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_169_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double);
template <>
Tensor lowerToNextLayer<batch_rule_169_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, double>(
  batch_rule_169_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const Tensor & mean, const Tensor & invstd, double eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor invstd_value;
  optional<int64_t> invstd_bdim;
  std::tie(invstd_value, invstd_bdim) = unwrapTensorAtLevel(invstd, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, mean_value, mean_bdim, invstd_value, invstd_bdim, eps);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['batch_norm_gather_stats']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_170_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, double, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_170_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, double, int64_t>(
  batch_rule_170_t batch_rule,
  const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, double momentum, double eps, int64_t count
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor invstd_value;
  optional<int64_t> invstd_bdim;
  std::tie(invstd_value, invstd_bdim) = unwrapTensorAtLevel(invstd, cur_level);
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, mean_value, mean_bdim, invstd_value, invstd_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, momentum, eps, count);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['batch_norm_gather_stats_with_counts']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_171_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, double, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_171_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, double, const Tensor &>(
  batch_rule_171_t batch_rule,
  const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, double momentum, double eps, const Tensor & counts
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor invstd_value;
  optional<int64_t> invstd_bdim;
  std::tie(invstd_value, invstd_bdim) = unwrapTensorAtLevel(invstd, cur_level);
  Tensor counts_value;
  optional<int64_t> counts_bdim;
  std::tie(counts_value, counts_bdim) = unwrapTensorAtLevel(counts, cur_level);
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, mean_value, mean_bdim, invstd_value, invstd_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, momentum, eps, counts_value, counts_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['native_batch_norm_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_172_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_172_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, ::std::array<bool,3>>(
  batch_rule_172_t batch_rule,
  const Tensor & grad_out, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_out_value;
  optional<int64_t> grad_out_bdim;
  std::tie(grad_out_value, grad_out_bdim) = unwrapTensorAtLevel(grad_out, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  optional<Tensor> save_mean_value;
  optional<int64_t> save_mean_bdim;
  if (save_mean) {
      std::tie(save_mean_value, save_mean_bdim) = unwrapTensorAtLevel(save_mean.value(), cur_level);
  }
  optional<Tensor> save_invstd_value;
  optional<int64_t> save_invstd_bdim;
  if (save_invstd) {
      std::tie(save_invstd_value, save_invstd_bdim) = unwrapTensorAtLevel(save_invstd.value(), cur_level);
  }
  auto results = batch_rule(grad_out_value, grad_out_bdim, input_value, input_bdim, weight_value, weight_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, save_mean_value, save_mean_bdim, save_invstd_value, save_invstd_bdim, train, eps, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['batch_norm_backward_reduce']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_173_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_173_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, bool, bool, bool>(
  batch_rule_173_t batch_rule,
  const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & weight, bool input_g, bool weight_g, bool bias_g
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_out_value;
  optional<int64_t> grad_out_bdim;
  std::tie(grad_out_value, grad_out_bdim) = unwrapTensorAtLevel(grad_out, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor invstd_value;
  optional<int64_t> invstd_bdim;
  std::tie(invstd_value, invstd_bdim) = unwrapTensorAtLevel(invstd, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(grad_out_value, grad_out_bdim, input_value, input_bdim, mean_value, mean_bdim, invstd_value, invstd_bdim, weight_value, weight_bdim, input_g, weight_g, bias_g);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level));
}

// ['batch_norm_backward_elemt']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_174_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_174_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, const Tensor &>(
  batch_rule_174_t batch_rule,
  const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu, const Tensor & count
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_out_value;
  optional<int64_t> grad_out_bdim;
  std::tie(grad_out_value, grad_out_bdim) = unwrapTensorAtLevel(grad_out, cur_level);
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor invstd_value;
  optional<int64_t> invstd_bdim;
  std::tie(invstd_value, invstd_bdim) = unwrapTensorAtLevel(invstd, cur_level);
  Tensor mean_dy_value;
  optional<int64_t> mean_dy_bdim;
  std::tie(mean_dy_value, mean_dy_bdim) = unwrapTensorAtLevel(mean_dy, cur_level);
  Tensor mean_dy_xmu_value;
  optional<int64_t> mean_dy_xmu_bdim;
  std::tie(mean_dy_xmu_value, mean_dy_xmu_bdim) = unwrapTensorAtLevel(mean_dy_xmu, cur_level);
  Tensor count_value;
  optional<int64_t> count_bdim;
  std::tie(count_value, count_bdim) = unwrapTensorAtLevel(count, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(grad_out_value, grad_out_bdim, input_value, input_bdim, mean_value, mean_bdim, invstd_value, invstd_bdim, weight_value, weight_bdim, mean_dy_value, mean_dy_bdim, mean_dy_xmu_value, mean_dy_xmu_bdim, count_value, count_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['batch_norm_update_stats']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_175_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_175_t,std::tuple<Tensor,Tensor>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double>(
  batch_rule_175_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, double momentum
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
      std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean.value(), cur_level);
  }
  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
      std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, momentum);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_nnpack_spatial_convolution']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_176_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_176_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef>(
  batch_rule_176_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, padding, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cdist', '_cdist_forward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_177_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_177_t,Tensor,const Tensor &, const Tensor &, double, c10::optional<int64_t>>(
  batch_rule_177_t batch_rule,
  const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor x1_value;
  optional<int64_t> x1_bdim;
  std::tie(x1_value, x1_bdim) = unwrapTensorAtLevel(x1, cur_level);
  Tensor x2_value;
  optional<int64_t> x2_bdim;
  std::tie(x2_value, x2_bdim) = unwrapTensorAtLevel(x2, cur_level);
  auto results = batch_rule(x1_value, x1_bdim, x2_value, x2_bdim, p, compute_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_cdist_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_178_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_178_t,Tensor,const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(
  batch_rule_178_t batch_rule,
  const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor x1_value;
  optional<int64_t> x1_bdim;
  std::tie(x1_value, x1_bdim) = unwrapTensorAtLevel(x1, cur_level);
  Tensor x2_value;
  optional<int64_t> x2_bdim;
  std::tie(x2_value, x2_bdim) = unwrapTensorAtLevel(x2, cur_level);
  Tensor cdist_value;
  optional<int64_t> cdist_bdim;
  std::tie(cdist_value, cdist_bdim) = unwrapTensorAtLevel(cdist, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, x1_value, x1_bdim, x2_value, x2_bdim, p, cdist_value, cdist_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['pdist', '_pdist_forward', 'pinverse']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_179_t)(const Tensor &, c10::optional<int64_t>, double);
template <>
Tensor lowerToNextLayer<batch_rule_179_t,Tensor,const Tensor &, double>(
  batch_rule_179_t batch_rule,
  const Tensor & self, double p
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_pdist_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_180_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_180_t,Tensor,const Tensor &, const Tensor &, double, const Tensor &>(
  batch_rule_180_t batch_rule,
  const Tensor & grad, const Tensor & self, double p, const Tensor & pdist
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor pdist_value;
  optional<int64_t> pdist_bdim;
  std::tie(pdist_value, pdist_bdim) = unwrapTensorAtLevel(pdist, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_value, self_bdim, p, pdist_value, pdist_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cosine_similarity', 'smooth_l1_loss', 'huber_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_181_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_181_t,Tensor,const Tensor &, const Tensor &, int64_t, double>(
  batch_rule_181_t batch_rule,
  const Tensor & x1, const Tensor & x2, int64_t dim, double eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor x1_value;
  optional<int64_t> x1_bdim;
  std::tie(x1_value, x1_bdim) = unwrapTensorAtLevel(x1, cur_level);
  Tensor x2_value;
  optional<int64_t> x2_bdim;
  std::tie(x2_value, x2_bdim) = unwrapTensorAtLevel(x2, cur_level);
  auto results = batch_rule(x1_value, x1_bdim, x2_value, x2_bdim, dim, eps);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['movedim.intlist', 'moveaxis.intlist', '_reshape_alias', 'roll']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_182_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_182_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef>(
  batch_rule_182_t batch_rule,
  const Tensor & self, IntArrayRef source, IntArrayRef destination
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source, destination);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['is_pinned']
typedef std::tuple<bool> (*batch_rule_183_t)(const Tensor &, c10::optional<int64_t>, c10::optional<Device>);
template <>
bool lowerToNextLayer<batch_rule_183_t,bool,const Tensor &, c10::optional<Device>>(
  batch_rule_183_t batch_rule,
  const Tensor & self, c10::optional<Device> device
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, device);
  return std::get<0>(results);
}

// ['pin_memory', '_pin_memory']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_184_t)(const Tensor &, c10::optional<int64_t>, c10::optional<Device>);
template <>
Tensor lowerToNextLayer<batch_rule_184_t,Tensor,const Tensor &, c10::optional<Device>>(
  batch_rule_184_t batch_rule,
  const Tensor & self, c10::optional<Device> device
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, device);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['poisson_nll_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_185_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_185_t,Tensor,const Tensor &, const Tensor &, bool, bool, double, int64_t>(
  batch_rule_185_t batch_rule,
  const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  auto results = batch_rule(input_value, input_bdim, target_value, target_bdim, log_input, full, eps, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['randint_like']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_186_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_186_t,Tensor,const Tensor &, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_186_t batch_rule,
  const Tensor & self, int64_t high, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, high, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['randint_like.low_dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_187_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_187_t,Tensor,const Tensor &, int64_t, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_187_t batch_rule,
  const Tensor & self, int64_t low, int64_t high, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, low, high, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['repeat_interleave.self_Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_188_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_188_t,Tensor,const Tensor &, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>>(
  batch_rule_188_t batch_rule,
  const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor repeats_value;
  optional<int64_t> repeats_bdim;
  std::tie(repeats_value, repeats_bdim) = unwrapTensorAtLevel(repeats, cur_level);
  auto results = batch_rule(self_value, self_bdim, repeats_value, repeats_bdim, dim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['repeat_interleave.self_int']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_189_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_189_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>>(
  batch_rule_189_t batch_rule,
  const Tensor & self, int64_t repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, repeats, dim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['rrelu']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_190_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, bool, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_190_t,Tensor,const Tensor &, const Scalar &, const Scalar &, bool, c10::optional<Generator>>(
  batch_rule_190_t batch_rule,
  const Tensor & self, const Scalar & lower, const Scalar & upper, bool training, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, lower, upper, training, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gelu', 'linalg_eigvalsh', 'linalg_cond.p_str']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_191_t)(const Tensor &, c10::optional<int64_t>, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_191_t,Tensor,const Tensor &, c10::string_view>(
  batch_rule_191_t batch_rule,
  const Tensor & self, c10::string_view approximate
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, approximate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gelu_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_192_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_192_t,Tensor,const Tensor &, const Tensor &, c10::string_view>(
  batch_rule_192_t batch_rule,
  const Tensor & grad_output, const Tensor & self, c10::string_view approximate
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, approximate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['select.Dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_193_t)(const Tensor &, c10::optional<int64_t>, Dimname, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_193_t,Tensor,const Tensor &, Dimname, int64_t>(
  batch_rule_193_t batch_rule,
  const Tensor & self, Dimname dim, int64_t index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['logit', 'special_logit']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_194_t)(const Tensor &, c10::optional<int64_t>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_194_t,Tensor,const Tensor &, c10::optional<double>>(
  batch_rule_194_t batch_rule,
  const Tensor & self, c10::optional<double> eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, eps);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['size.int', 'stride.int']
typedef std::tuple<int64_t> (*batch_rule_195_t)(const Tensor &, c10::optional<int64_t>, int64_t);
template <>
int64_t lowerToNextLayer<batch_rule_195_t,int64_t,const Tensor &, int64_t>(
  batch_rule_195_t batch_rule,
  const Tensor & self, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return std::get<0>(results);
}

// ['size.Dimname', 'stride.Dimname']
typedef std::tuple<int64_t> (*batch_rule_196_t)(const Tensor &, c10::optional<int64_t>, Dimname);
template <>
int64_t lowerToNextLayer<batch_rule_196_t,int64_t,const Tensor &, Dimname>(
  batch_rule_196_t batch_rule,
  const Tensor & self, Dimname dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return std::get<0>(results);
}

// ['slice.Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_197_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_197_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t>(
  batch_rule_197_t batch_rule,
  const Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['slice_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_198_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_198_t,Tensor,const Tensor &, IntArrayRef, int64_t, int64_t, int64_t, int64_t>(
  batch_rule_198_t batch_rule,
  const Tensor & grad_output, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['slice_scatter']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_199_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_199_t,Tensor,const Tensor &, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t>(
  batch_rule_199_t batch_rule,
  const Tensor & self, const Tensor & src, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor src_value;
  optional<int64_t> src_bdim;
  std::tie(src_value, src_bdim) = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['select_scatter']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_200_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_200_t,Tensor,const Tensor &, const Tensor &, int64_t, int64_t>(
  batch_rule_200_t batch_rule,
  const Tensor & self, const Tensor & src, int64_t dim, int64_t index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor src_value;
  optional<int64_t> src_bdim;
  std::tie(src_value, src_bdim) = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['diagonal_scatter']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_201_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_201_t,Tensor,const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(
  batch_rule_201_t batch_rule,
  const Tensor & self, const Tensor & src, int64_t offset, int64_t dim1, int64_t dim2
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor src_value;
  optional<int64_t> src_bdim;
  std::tie(src_value, src_bdim) = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, src_value, src_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['hsplit.int', 'vsplit.int', 'dsplit.int', 'unbind.int']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_202_t)(const Tensor &, c10::optional<int64_t>, int64_t);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_202_t,::std::vector<Tensor>,const Tensor &, int64_t>(
  batch_rule_202_t batch_rule,
  const Tensor & self, int64_t sections
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sections);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['hsplit.array', 'vsplit.array', 'dsplit.array']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_203_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_203_t,::std::vector<Tensor>,const Tensor &, IntArrayRef>(
  batch_rule_203_t batch_rule,
  const Tensor & self, IntArrayRef indices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['stft']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_204_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, c10::optional<bool>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_204_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, bool, c10::optional<bool>, c10::optional<bool>>(
  batch_rule_204_t batch_rule,
  const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> window_value;
  optional<int64_t> window_bdim;
  if (window) {
      std::tie(window_value, window_bdim) = unwrapTensorAtLevel(window.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, n_fft, hop_length, win_length, window_value, window_bdim, normalized, onesided, return_complex);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['istft']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_205_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, bool, c10::optional<bool>, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_205_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, bool, bool, c10::optional<bool>, c10::optional<int64_t>, bool>(
  batch_rule_205_t batch_rule,
  const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<Tensor> & window, bool center, bool normalized, c10::optional<bool> onesided, c10::optional<int64_t> length, bool return_complex
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> window_value;
  optional<int64_t> window_bdim;
  if (window) {
      std::tie(window_value, window_bdim) = unwrapTensorAtLevel(window.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, n_fft, hop_length, win_length, window_value, window_bdim, center, normalized, onesided, length, return_complex);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['std.dim', 'var.dim']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_206_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_206_t,Tensor,const Tensor &, IntArrayRef, bool, bool>(
  batch_rule_206_t batch_rule,
  const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['std.correction', 'var.correction']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_207_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_207_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool>(
  batch_rule_207_t batch_rule,
  const Tensor & self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['std_mean', 'var_mean', 'eig', 'qr', 'linalg_lu_factor', 'linalg_inv_ex']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_208_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_208_t,std::tuple<Tensor,Tensor>,const Tensor &, bool>(
  batch_rule_208_t batch_rule,
  const Tensor & self, bool unbiased
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, unbiased);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['std_mean.dim', 'var_mean.dim']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_209_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_209_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef, bool, bool>(
  batch_rule_209_t batch_rule,
  const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['std_mean.correction', 'var_mean.correction']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_210_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_210_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool>(
  batch_rule_210_t batch_rule,
  const Tensor & self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['std_mean.names_dim', 'var_mean.names_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_211_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_211_t,std::tuple<Tensor,Tensor>,const Tensor &, DimnameList, bool, bool>(
  batch_rule_211_t batch_rule,
  const Tensor & self, DimnameList dim, bool unbiased, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['std_mean.correction_names', 'var_mean.correction_names']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_212_t)(const Tensor &, c10::optional<int64_t>, DimnameList, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_212_t,std::tuple<Tensor,Tensor>,const Tensor &, DimnameList, c10::optional<int64_t>, bool>(
  batch_rule_212_t batch_rule,
  const Tensor & self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['std.names_dim', 'var.names_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_213_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_213_t,Tensor,const Tensor &, DimnameList, bool, bool>(
  batch_rule_213_t batch_rule,
  const Tensor & self, DimnameList dim, bool unbiased, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, unbiased, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['std.correction_names', 'var.correction_names']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_214_t)(const Tensor &, c10::optional<int64_t>, DimnameList, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_214_t,Tensor,const Tensor &, DimnameList, c10::optional<int64_t>, bool>(
  batch_rule_214_t batch_rule,
  const Tensor & self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, correction, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['prod.dim_int']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_215_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_215_t,Tensor,const Tensor &, int64_t, bool, c10::optional<ScalarType>>(
  batch_rule_215_t batch_rule,
  const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['prod.dim_Dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_216_t)(const Tensor &, c10::optional<int64_t>, Dimname, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_216_t,Tensor,const Tensor &, Dimname, bool, c10::optional<ScalarType>>(
  batch_rule_216_t batch_rule,
  const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['tensordot']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_217_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_217_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(
  batch_rule_217_t batch_rule,
  const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, dims_self, dims_other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['transpose.Dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_218_t)(const Tensor &, c10::optional<int64_t>, Dimname, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_218_t,Tensor,const Tensor &, Dimname, Dimname>(
  batch_rule_218_t batch_rule,
  const Tensor & self, Dimname dim0, Dimname dim1
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim0, dim1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['rot90']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_219_t)(const Tensor &, c10::optional<int64_t>, int64_t, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_219_t,Tensor,const Tensor &, int64_t, IntArrayRef>(
  batch_rule_219_t batch_rule,
  const Tensor & self, int64_t k, IntArrayRef dims
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['trapz.dx', '_make_per_tensor_quantized_tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_220_t)(const Tensor &, c10::optional<int64_t>, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_220_t,Tensor,const Tensor &, double, int64_t>(
  batch_rule_220_t batch_rule,
  const Tensor & y, double dx, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor y_value;
  optional<int64_t> y_bdim;
  std::tie(y_value, y_bdim) = unwrapTensorAtLevel(y, cur_level);
  auto results = batch_rule(y_value, y_bdim, dx, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_trilinear']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_221_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_221_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_221_t batch_rule,
  const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor i1_value;
  optional<int64_t> i1_bdim;
  std::tie(i1_value, i1_bdim) = unwrapTensorAtLevel(i1, cur_level);
  Tensor i2_value;
  optional<int64_t> i2_bdim;
  std::tie(i2_value, i2_bdim) = unwrapTensorAtLevel(i2, cur_level);
  Tensor i3_value;
  optional<int64_t> i3_bdim;
  std::tie(i3_value, i3_bdim) = unwrapTensorAtLevel(i3, cur_level);
  auto results = batch_rule(i1_value, i1_bdim, i2_value, i2_bdim, i3_value, i3_bdim, expand1, expand2, expand3, sumdim, unroll_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['triplet_margin_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_222_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, double, bool, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_222_t,Tensor,const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t>(
  batch_rule_222_t batch_rule,
  const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor anchor_value;
  optional<int64_t> anchor_bdim;
  std::tie(anchor_value, anchor_bdim) = unwrapTensorAtLevel(anchor, cur_level);
  Tensor positive_value;
  optional<int64_t> positive_bdim;
  std::tie(positive_value, positive_bdim) = unwrapTensorAtLevel(positive, cur_level);
  Tensor negative_value;
  optional<int64_t> negative_bdim;
  std::tie(negative_value, negative_bdim) = unwrapTensorAtLevel(negative, cur_level);
  auto results = batch_rule(anchor_value, anchor_bdim, positive_value, positive_bdim, negative_value, negative_bdim, margin, p, eps, swap, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_unique', 'symeig', '_symeig_helper', 'linalg_cholesky_ex']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_223_t)(const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_223_t,std::tuple<Tensor,Tensor>,const Tensor &, bool, bool>(
  batch_rule_223_t batch_rule,
  const Tensor & self, bool sorted, bool return_inverse
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sorted, return_inverse);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['unique_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_224_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_224_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, int64_t, bool, bool, bool>(
  batch_rule_224_t batch_rule,
  const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, sorted, return_inverse, return_counts);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['unique_consecutive']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_225_t)(const Tensor &, c10::optional<int64_t>, bool, bool, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_225_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool, bool, c10::optional<int64_t>>(
  batch_rule_225_t batch_rule,
  const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, return_inverse, return_counts, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['unique_dim_consecutive']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_226_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_226_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, int64_t, bool, bool>(
  batch_rule_226_t batch_rule,
  const Tensor & self, int64_t dim, bool return_inverse, bool return_counts
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, return_inverse, return_counts);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_unique2']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_227_t)(const Tensor &, c10::optional<int64_t>, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_227_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool, bool, bool>(
  batch_rule_227_t batch_rule,
  const Tensor & self, bool sorted, bool return_inverse, bool return_counts
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sorted, return_inverse, return_counts);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['where.ScalarSelf']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_228_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_228_t,Tensor,const Tensor &, const Scalar &, const Tensor &>(
  batch_rule_228_t batch_rule,
  const Tensor & condition, const Scalar & self, const Tensor & other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor condition_value;
  optional<int64_t> condition_bdim;
  std::tie(condition_value, condition_bdim) = unwrapTensorAtLevel(condition, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['where', 'nonzero_numpy']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_229_t)(const Tensor &, c10::optional<int64_t>);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_229_t,::std::vector<Tensor>,const Tensor &>(
  batch_rule_229_t batch_rule,
  const Tensor & condition
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor condition_value;
  optional<int64_t> condition_bdim;
  std::tie(condition_value, condition_bdim) = unwrapTensorAtLevel(condition, cur_level);
  auto results = batch_rule(condition_value, condition_bdim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_weight_norm_cuda_interface', 'multilabel_margin_loss_forward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_230_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_230_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, int64_t>(
  batch_rule_230_t batch_rule,
  const Tensor & v, const Tensor & g, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor v_value;
  optional<int64_t> v_bdim;
  std::tie(v_value, v_bdim) = unwrapTensorAtLevel(v, cur_level);
  Tensor g_value;
  optional<int64_t> g_bdim;
  std::tie(g_value, g_bdim) = unwrapTensorAtLevel(g, cur_level);
  auto results = batch_rule(v_value, v_bdim, g_value, g_bdim, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_weight_norm_cuda_interface_backward', '_weight_norm_differentiable_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_231_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_231_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(
  batch_rule_231_t batch_rule,
  const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_w_value;
  optional<int64_t> grad_w_bdim;
  std::tie(grad_w_value, grad_w_bdim) = unwrapTensorAtLevel(grad_w, cur_level);
  Tensor saved_v_value;
  optional<int64_t> saved_v_bdim;
  std::tie(saved_v_value, saved_v_bdim) = unwrapTensorAtLevel(saved_v, cur_level);
  Tensor saved_g_value;
  optional<int64_t> saved_g_bdim;
  std::tie(saved_g_value, saved_g_bdim) = unwrapTensorAtLevel(saved_g, cur_level);
  Tensor saved_norms_value;
  optional<int64_t> saved_norms_bdim;
  std::tie(saved_norms_value, saved_norms_bdim) = unwrapTensorAtLevel(saved_norms, cur_level);
  auto results = batch_rule(grad_w_value, grad_w_bdim, saved_v_value, saved_v_bdim, saved_g_value, saved_g_bdim, saved_norms_value, saved_norms_bdim, dim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['binomial', 'normal.Tensor_Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_232_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_232_t,Tensor,const Tensor &, const Tensor &, c10::optional<Generator>>(
  batch_rule_232_t batch_rule,
  const Tensor & count, const Tensor & prob, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor count_value;
  optional<int64_t> count_bdim;
  std::tie(count_value, count_bdim) = unwrapTensorAtLevel(count, cur_level);
  Tensor prob_value;
  optional<int64_t> prob_bdim;
  std::tie(prob_value, prob_bdim) = unwrapTensorAtLevel(prob, cur_level);
  auto results = batch_rule(count_value, count_bdim, prob_value, prob_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['native_norm.ScalarOpt_dim_dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_233_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_233_t,Tensor,const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_233_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_sparse_sum.dtype', 'view.dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_234_t)(const Tensor &, c10::optional<int64_t>, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_234_t,Tensor,const Tensor &, ScalarType>(
  batch_rule_234_t batch_rule,
  const Tensor & self, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_sparse_sum.dim_dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_235_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_235_t,Tensor,const Tensor &, IntArrayRef, ScalarType>(
  batch_rule_235_t batch_rule,
  const Tensor & self, IntArrayRef dim, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_sparse_sum_backward', 'max_unpool2d', 'reflection_pad1d_backward', 'reflection_pad2d_backward', 'reflection_pad3d_backward', 'replication_pad1d_backward', 'replication_pad2d_backward', 'replication_pad3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_236_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_236_t,Tensor,const Tensor &, const Tensor &, IntArrayRef>(
  batch_rule_236_t batch_rule,
  const Tensor & grad, const Tensor & self, IntArrayRef dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['norm.ScalarOpt_dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_237_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_237_t,Tensor,const Tensor &, const c10::optional<Scalar> &, ScalarType>(
  batch_rule_237_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & p, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['norm.ScalarOpt_dim_dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_238_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, IntArrayRef, bool, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_238_t,Tensor,const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool, ScalarType>(
  batch_rule_238_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['norm.ScalarOpt_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_239_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_239_t,Tensor,const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool>(
  batch_rule_239_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['norm.names_ScalarOpt_dim_dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_240_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, DimnameList, bool, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_240_t,Tensor,const Tensor &, const c10::optional<Scalar> &, DimnameList, bool, ScalarType>(
  batch_rule_240_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & p, DimnameList dim, bool keepdim, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['norm.names_ScalarOpt_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_241_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, DimnameList, bool);
template <>
Tensor lowerToNextLayer<batch_rule_241_t,Tensor,const Tensor &, const c10::optional<Scalar> &, DimnameList, bool>(
  batch_rule_241_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & p, DimnameList dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['clone']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_242_t)(const Tensor &, c10::optional<int64_t>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_242_t,Tensor,const Tensor &, c10::optional<MemoryFormat>>(
  batch_rule_242_t batch_rule,
  const Tensor & self, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['sparse_csr_tensor.crow_col_value_size', '_sparse_csr_tensor_unsafe']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_243_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_243_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_243_t batch_rule,
  const Tensor & crow_indices, const Tensor & col_indices, const Tensor & values, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor crow_indices_value;
  optional<int64_t> crow_indices_bdim;
  std::tie(crow_indices_value, crow_indices_bdim) = unwrapTensorAtLevel(crow_indices, cur_level);
  Tensor col_indices_value;
  optional<int64_t> col_indices_bdim;
  std::tie(col_indices_value, col_indices_bdim) = unwrapTensorAtLevel(col_indices, cur_level);
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(crow_indices_value, crow_indices_bdim, col_indices_value, col_indices_bdim, values_value, values_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['sparse_csr_tensor.crow_col_value']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_244_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_244_t,Tensor,const Tensor &, const Tensor &, const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_244_t batch_rule,
  const Tensor & crow_indices, const Tensor & col_indices, const Tensor & values, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor crow_indices_value;
  optional<int64_t> crow_indices_bdim;
  std::tie(crow_indices_value, crow_indices_bdim) = unwrapTensorAtLevel(crow_indices, cur_level);
  Tensor col_indices_value;
  optional<int64_t> col_indices_bdim;
  std::tie(col_indices_value, col_indices_bdim) = unwrapTensorAtLevel(col_indices, cur_level);
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(crow_indices_value, crow_indices_bdim, col_indices_value, col_indices_bdim, values_value, values_bdim, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['sparse_coo_tensor.indices']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_245_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_245_t,Tensor,const Tensor &, const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_245_t batch_rule,
  const Tensor & indices, const Tensor & values, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(indices_value, indices_bdim, values_value, values_bdim, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['sparse_coo_tensor.indices_size', '_sparse_coo_tensor_unsafe']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_246_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_246_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_246_t batch_rule,
  const Tensor & indices, const Tensor & values, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(indices_value, indices_bdim, values_value, values_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_sparse_coo_tensor_with_dims_and_tensors']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_247_t)(int64_t, int64_t, IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_247_t,Tensor,int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_247_t batch_rule,
  int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(sparse_dim, dense_dim, size, indices_value, indices_bdim, values_value, values_bdim, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['unbind.Dimname']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_248_t)(const Tensor &, c10::optional<int64_t>, Dimname);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_248_t,::std::vector<Tensor>,const Tensor &, Dimname>(
  batch_rule_248_t batch_rule,
  const Tensor & self, Dimname dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['mkldnn_reorder_conv2d_weight', 'mkldnn_reorder_conv3d_weight']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_249_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_249_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_249_t batch_rule,
  const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding, stride, dilation, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantize_per_tensor_dynamic']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_250_t)(const Tensor &, c10::optional<int64_t>, ScalarType, bool);
template <>
Tensor lowerToNextLayer<batch_rule_250_t,Tensor,const Tensor &, ScalarType, bool>(
  batch_rule_250_t batch_rule,
  const Tensor & self, ScalarType dtype, bool reduce_range
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, reduce_range);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantize_per_tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_251_t)(const Tensor &, c10::optional<int64_t>, double, int64_t, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_251_t,Tensor,const Tensor &, double, int64_t, ScalarType>(
  batch_rule_251_t batch_rule,
  const Tensor & self, double scale, int64_t zero_point, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale, zero_point, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantize_per_tensor.tensor_qparams']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_252_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_252_t,Tensor,const Tensor &, const Tensor &, const Tensor &, ScalarType>(
  batch_rule_252_t batch_rule,
  const Tensor & self, const Tensor & scale, const Tensor & zero_point, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantize_per_channel']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_253_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_253_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, ScalarType>(
  batch_rule_253_t batch_rule,
  const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, ScalarType dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scales_value;
  optional<int64_t> scales_bdim;
  std::tie(scales_value, scales_bdim) = unwrapTensorAtLevel(scales, cur_level);
  Tensor zero_points_value;
  optional<int64_t> zero_points_bdim;
  std::tie(zero_points_value, zero_points_bdim) = unwrapTensorAtLevel(zero_points, cur_level);
  auto results = batch_rule(self_value, self_bdim, scales_value, scales_bdim, zero_points_value, zero_points_bdim, axis, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['q_scale']
typedef std::tuple<double> (*batch_rule_254_t)(const Tensor &, c10::optional<int64_t>);
template <>
double lowerToNextLayer<batch_rule_254_t,double,const Tensor &>(
  batch_rule_254_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::get<0>(results);
}

// ['qscheme']
typedef std::tuple<QScheme> (*batch_rule_255_t)(const Tensor &, c10::optional<int64_t>);
template <>
QScheme lowerToNextLayer<batch_rule_255_t,QScheme,const Tensor &>(
  batch_rule_255_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::get<0>(results);
}

// ['fake_quantize_per_tensor_affine']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_256_t)(const Tensor &, c10::optional<int64_t>, double, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_256_t,Tensor,const Tensor &, double, int64_t, int64_t, int64_t>(
  batch_rule_256_t batch_rule,
  const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale, zero_point, quant_min, quant_max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fake_quantize_per_tensor_affine.tensor_qparams']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_257_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_257_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(
  batch_rule_257_t batch_rule,
  const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, quant_min, quant_max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fake_quantize_per_tensor_affine_cachemask']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_258_t)(const Tensor &, c10::optional<int64_t>, double, int64_t, int64_t, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_258_t,std::tuple<Tensor,Tensor>,const Tensor &, double, int64_t, int64_t, int64_t>(
  batch_rule_258_t batch_rule,
  const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale, zero_point, quant_min, quant_max);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_fake_quantize_per_tensor_affine_cachemask_tensor_qparams']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_259_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_259_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(
  batch_rule_259_t batch_rule,
  const Tensor & self, const Tensor & scale, const Tensor & zero_point, const Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  Tensor fake_quant_enabled_value;
  optional<int64_t> fake_quant_enabled_bdim;
  std::tie(fake_quant_enabled_value, fake_quant_enabled_bdim) = unwrapTensorAtLevel(fake_quant_enabled, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, fake_quant_enabled_value, fake_quant_enabled_bdim, quant_min, quant_max);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_fake_quantize_learnable_per_tensor_affine']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_260_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_260_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double>(
  batch_rule_260_t batch_rule,
  const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, quant_min, quant_max, grad_factor);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_fake_quantize_learnable_per_tensor_affine_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_261_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_261_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double>(
  batch_rule_261_t batch_rule,
  const Tensor & grad, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, quant_min, quant_max, grad_factor);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['fake_quantize_per_channel_affine']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_262_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_262_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(
  batch_rule_262_t batch_rule,
  const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, axis, quant_min, quant_max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fake_quantize_per_channel_affine_cachemask']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_263_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_263_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(
  batch_rule_263_t batch_rule,
  const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, axis, quant_min, quant_max);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_fake_quantize_learnable_per_channel_affine']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_264_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_264_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, double>(
  batch_rule_264_t batch_rule,
  const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, axis, quant_min, quant_max, grad_factor);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_fake_quantize_learnable_per_channel_affine_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_265_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_265_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, double>(
  batch_rule_265_t batch_rule,
  const Tensor & grad, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor scale_value;
  optional<int64_t> scale_bdim;
  std::tie(scale_value, scale_bdim) = unwrapTensorAtLevel(scale, cur_level);
  Tensor zero_point_value;
  optional<int64_t> zero_point_bdim;
  std::tie(zero_point_value, zero_point_bdim) = unwrapTensorAtLevel(zero_point, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, axis, quant_min, quant_max, grad_factor);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_choose_qparams_per_tensor']
typedef std::tuple<double,int64_t> (*batch_rule_266_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<double,int64_t> lowerToNextLayer<batch_rule_266_t,std::tuple<double,int64_t>,const Tensor &, bool>(
  batch_rule_266_t batch_rule,
  const Tensor & self, bool reduce_range
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, reduce_range);
  return std::make_tuple(std::get<0>(results), std::get<1>(results));
}

// ['choose_qparams_optimized']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_267_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, double, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_267_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, int64_t, double, int64_t>(
  batch_rule_267_t batch_rule,
  const Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, numel, n_bins, ratio, bit_width);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_autocast_to_reduced_precision']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_268_t)(const Tensor &, c10::optional<int64_t>, bool, bool, ScalarType, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_268_t,Tensor,const Tensor &, bool, bool, ScalarType, ScalarType>(
  batch_rule_268_t batch_rule,
  const Tensor & self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_autocast_to_full_precision']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_269_t)(const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_269_t,Tensor,const Tensor &, bool, bool>(
  batch_rule_269_t batch_rule,
  const Tensor & self, bool cuda_enabled, bool cpu_enabled
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, cuda_enabled, cpu_enabled);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_to_copy']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_270_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_270_t,Tensor,const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, bool, c10::optional<MemoryFormat>>(
  batch_rule_270_t batch_rule,
  const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, non_blocking, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['to.dtype_layout']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_271_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_271_t,Tensor,const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_271_t batch_rule,
  const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['to.device']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_272_t)(const Tensor &, c10::optional<int64_t>, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_272_t,Tensor,const Tensor &, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_272_t batch_rule,
  const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, device, dtype, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['to.dtype']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_273_t)(const Tensor &, c10::optional<int64_t>, ScalarType, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_273_t,Tensor,const Tensor &, ScalarType, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_273_t batch_rule,
  const Tensor & self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['to.other']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_274_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_274_t,Tensor,const Tensor &, const Tensor &, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_274_t batch_rule,
  const Tensor & self, const Tensor & other, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['item', '_local_scalar_dense']
typedef std::tuple<Scalar> (*batch_rule_275_t)(const Tensor &, c10::optional<int64_t>);
template <>
Scalar lowerToNextLayer<batch_rule_275_t,Scalar,const Tensor &>(
  batch_rule_275_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::get<0>(results);
}

// ['result_type.Tensor']
typedef std::tuple<ScalarType> (*batch_rule_276_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
ScalarType lowerToNextLayer<batch_rule_276_t,ScalarType,const Tensor &, const Tensor &>(
  batch_rule_276_t batch_rule,
  const Tensor & tensor, const Tensor & other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor tensor_value;
  optional<int64_t> tensor_bdim;
  std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(tensor, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(tensor_value, tensor_bdim, other_value, other_bdim);
  return std::get<0>(results);
}

// ['result_type.Scalar']
typedef std::tuple<ScalarType> (*batch_rule_277_t)(const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
ScalarType lowerToNextLayer<batch_rule_277_t,ScalarType,const Tensor &, const Scalar &>(
  batch_rule_277_t batch_rule,
  const Tensor & tensor, const Scalar & other
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor tensor_value;
  optional<int64_t> tensor_bdim;
  std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(tensor, cur_level);
  auto results = batch_rule(tensor_value, tensor_bdim, other);
  return std::get<0>(results);
}

// ['result_type.Scalar_Tensor']
typedef std::tuple<ScalarType> (*batch_rule_278_t)(const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
ScalarType lowerToNextLayer<batch_rule_278_t,ScalarType,const Scalar &, const Tensor &>(
  batch_rule_278_t batch_rule,
  const Scalar & scalar, const Tensor & tensor
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor tensor_value;
  optional<int64_t> tensor_bdim;
  std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(tensor, cur_level);
  auto results = batch_rule(scalar, tensor_value, tensor_bdim);
  return std::get<0>(results);
}

// ['_thnn_fused_lstm_cell']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_279_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_279_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_279_t batch_rule,
  const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_gates_value;
  optional<int64_t> input_gates_bdim;
  std::tie(input_gates_value, input_gates_bdim) = unwrapTensorAtLevel(input_gates, cur_level);
  Tensor hidden_gates_value;
  optional<int64_t> hidden_gates_bdim;
  std::tie(hidden_gates_value, hidden_gates_bdim) = unwrapTensorAtLevel(hidden_gates, cur_level);
  Tensor cx_value;
  optional<int64_t> cx_bdim;
  std::tie(cx_value, cx_bdim) = unwrapTensorAtLevel(cx, cur_level);
  optional<Tensor> input_bias_value;
  optional<int64_t> input_bias_bdim;
  if (input_bias) {
      std::tie(input_bias_value, input_bias_bdim) = unwrapTensorAtLevel(input_bias.value(), cur_level);
  }
  optional<Tensor> hidden_bias_value;
  optional<int64_t> hidden_bias_bdim;
  if (hidden_bias) {
      std::tie(hidden_bias_value, hidden_bias_bdim) = unwrapTensorAtLevel(hidden_bias.value(), cur_level);
  }
  auto results = batch_rule(input_gates_value, input_gates_bdim, hidden_gates_value, hidden_gates_bdim, cx_value, cx_bdim, input_bias_value, input_bias_bdim, hidden_bias_value, hidden_bias_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_thnn_fused_lstm_cell_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_280_t)(const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_280_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, const Tensor &, bool>(
  batch_rule_280_t batch_rule,
  const c10::optional<Tensor> & grad_hy, const c10::optional<Tensor> & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor cx_value;
  optional<int64_t> cx_bdim;
  std::tie(cx_value, cx_bdim) = unwrapTensorAtLevel(cx, cur_level);
  Tensor cy_value;
  optional<int64_t> cy_bdim;
  std::tie(cy_value, cy_bdim) = unwrapTensorAtLevel(cy, cur_level);
  Tensor workspace_value;
  optional<int64_t> workspace_bdim;
  std::tie(workspace_value, workspace_bdim) = unwrapTensorAtLevel(workspace, cur_level);
  optional<Tensor> grad_hy_value;
  optional<int64_t> grad_hy_bdim;
  if (grad_hy) {
      std::tie(grad_hy_value, grad_hy_bdim) = unwrapTensorAtLevel(grad_hy.value(), cur_level);
  }
  optional<Tensor> grad_cy_value;
  optional<int64_t> grad_cy_bdim;
  if (grad_cy) {
      std::tie(grad_cy_value, grad_cy_bdim) = unwrapTensorAtLevel(grad_cy.value(), cur_level);
  }
  auto results = batch_rule(grad_hy_value, grad_hy_bdim, grad_cy_value, grad_cy_bdim, cx_value, cx_bdim, cy_value, cy_bdim, workspace_value, workspace_bdim, has_bias);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level), makeBatched(std::get<8>(results), std::get<9>(results), cur_level));
}

// ['_thnn_differentiable_lstm_cell_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_281_t)(const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_281_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &>(
  batch_rule_281_t batch_rule,
  const c10::optional<Tensor> & grad_hy, const c10::optional<Tensor> & grad_cy, const Tensor & input_gates, const Tensor & hidden_gates, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias, const Tensor & cx, const Tensor & cy
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_gates_value;
  optional<int64_t> input_gates_bdim;
  std::tie(input_gates_value, input_gates_bdim) = unwrapTensorAtLevel(input_gates, cur_level);
  Tensor hidden_gates_value;
  optional<int64_t> hidden_gates_bdim;
  std::tie(hidden_gates_value, hidden_gates_bdim) = unwrapTensorAtLevel(hidden_gates, cur_level);
  Tensor cx_value;
  optional<int64_t> cx_bdim;
  std::tie(cx_value, cx_bdim) = unwrapTensorAtLevel(cx, cur_level);
  Tensor cy_value;
  optional<int64_t> cy_bdim;
  std::tie(cy_value, cy_bdim) = unwrapTensorAtLevel(cy, cur_level);
  optional<Tensor> grad_hy_value;
  optional<int64_t> grad_hy_bdim;
  if (grad_hy) {
      std::tie(grad_hy_value, grad_hy_bdim) = unwrapTensorAtLevel(grad_hy.value(), cur_level);
  }
  optional<Tensor> grad_cy_value;
  optional<int64_t> grad_cy_bdim;
  if (grad_cy) {
      std::tie(grad_cy_value, grad_cy_bdim) = unwrapTensorAtLevel(grad_cy.value(), cur_level);
  }
  optional<Tensor> input_bias_value;
  optional<int64_t> input_bias_bdim;
  if (input_bias) {
      std::tie(input_bias_value, input_bias_bdim) = unwrapTensorAtLevel(input_bias.value(), cur_level);
  }
  optional<Tensor> hidden_bias_value;
  optional<int64_t> hidden_bias_bdim;
  if (hidden_bias) {
      std::tie(hidden_bias_value, hidden_bias_bdim) = unwrapTensorAtLevel(hidden_bias.value(), cur_level);
  }
  auto results = batch_rule(grad_hy_value, grad_hy_bdim, grad_cy_value, grad_cy_bdim, input_gates_value, input_gates_bdim, hidden_gates_value, hidden_gates_bdim, input_bias_value, input_bias_bdim, hidden_bias_value, hidden_bias_bdim, cx_value, cx_bdim, cy_value, cy_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level), makeBatched(std::get<8>(results), std::get<9>(results), cur_level));
}

// ['_thnn_fused_gru_cell']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_282_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_282_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_282_t batch_rule,
  const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_gates_value;
  optional<int64_t> input_gates_bdim;
  std::tie(input_gates_value, input_gates_bdim) = unwrapTensorAtLevel(input_gates, cur_level);
  Tensor hidden_gates_value;
  optional<int64_t> hidden_gates_bdim;
  std::tie(hidden_gates_value, hidden_gates_bdim) = unwrapTensorAtLevel(hidden_gates, cur_level);
  Tensor hx_value;
  optional<int64_t> hx_bdim;
  std::tie(hx_value, hx_bdim) = unwrapTensorAtLevel(hx, cur_level);
  optional<Tensor> input_bias_value;
  optional<int64_t> input_bias_bdim;
  if (input_bias) {
      std::tie(input_bias_value, input_bias_bdim) = unwrapTensorAtLevel(input_bias.value(), cur_level);
  }
  optional<Tensor> hidden_bias_value;
  optional<int64_t> hidden_bias_bdim;
  if (hidden_bias) {
      std::tie(hidden_bias_value, hidden_bias_bdim) = unwrapTensorAtLevel(hidden_bias.value(), cur_level);
  }
  auto results = batch_rule(input_gates_value, input_gates_bdim, hidden_gates_value, hidden_gates_bdim, hx_value, hx_bdim, input_bias_value, input_bias_bdim, hidden_bias_value, hidden_bias_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_thnn_fused_gru_cell_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_283_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_283_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, bool>(
  batch_rule_283_t batch_rule,
  const Tensor & grad_hy, const Tensor & workspace, bool has_bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_hy_value;
  optional<int64_t> grad_hy_bdim;
  std::tie(grad_hy_value, grad_hy_bdim) = unwrapTensorAtLevel(grad_hy, cur_level);
  Tensor workspace_value;
  optional<int64_t> workspace_bdim;
  std::tie(workspace_value, workspace_bdim) = unwrapTensorAtLevel(workspace, cur_level);
  auto results = batch_rule(grad_hy_value, grad_hy_bdim, workspace_value, workspace_bdim, has_bias);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level), makeBatched(std::get<8>(results), std::get<9>(results), cur_level));
}

// ['_thnn_differentiable_gru_cell_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_284_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_284_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_284_t batch_rule,
  const Tensor & grad_hy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_hy_value;
  optional<int64_t> grad_hy_bdim;
  std::tie(grad_hy_value, grad_hy_bdim) = unwrapTensorAtLevel(grad_hy, cur_level);
  Tensor input_gates_value;
  optional<int64_t> input_gates_bdim;
  std::tie(input_gates_value, input_gates_bdim) = unwrapTensorAtLevel(input_gates, cur_level);
  Tensor hidden_gates_value;
  optional<int64_t> hidden_gates_bdim;
  std::tie(hidden_gates_value, hidden_gates_bdim) = unwrapTensorAtLevel(hidden_gates, cur_level);
  Tensor hx_value;
  optional<int64_t> hx_bdim;
  std::tie(hx_value, hx_bdim) = unwrapTensorAtLevel(hx, cur_level);
  optional<Tensor> input_bias_value;
  optional<int64_t> input_bias_bdim;
  if (input_bias) {
      std::tie(input_bias_value, input_bias_bdim) = unwrapTensorAtLevel(input_bias.value(), cur_level);
  }
  optional<Tensor> hidden_bias_value;
  optional<int64_t> hidden_bias_bdim;
  if (hidden_bias) {
      std::tie(hidden_bias_value, hidden_bias_bdim) = unwrapTensorAtLevel(hidden_bias.value(), cur_level);
  }
  auto results = batch_rule(grad_hy_value, grad_hy_bdim, input_gates_value, input_gates_bdim, hidden_gates_value, hidden_gates_bdim, hx_value, hx_bdim, input_bias_value, input_bias_bdim, hidden_bias_value, hidden_bias_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level), makeBatched(std::get<8>(results), std::get<9>(results), cur_level));
}

// ['gru_cell', 'rnn_tanh_cell', 'rnn_relu_cell']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_285_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_285_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_285_t batch_rule,
  const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const c10::optional<Tensor> & b_ih, const c10::optional<Tensor> & b_hh
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor hx_value;
  optional<int64_t> hx_bdim;
  std::tie(hx_value, hx_bdim) = unwrapTensorAtLevel(hx, cur_level);
  Tensor w_ih_value;
  optional<int64_t> w_ih_bdim;
  std::tie(w_ih_value, w_ih_bdim) = unwrapTensorAtLevel(w_ih, cur_level);
  Tensor w_hh_value;
  optional<int64_t> w_hh_bdim;
  std::tie(w_hh_value, w_hh_bdim) = unwrapTensorAtLevel(w_hh, cur_level);
  optional<Tensor> b_ih_value;
  optional<int64_t> b_ih_bdim;
  if (b_ih) {
      std::tie(b_ih_value, b_ih_bdim) = unwrapTensorAtLevel(b_ih.value(), cur_level);
  }
  optional<Tensor> b_hh_value;
  optional<int64_t> b_hh_bdim;
  if (b_hh) {
      std::tie(b_hh_value, b_hh_bdim) = unwrapTensorAtLevel(b_hh.value(), cur_level);
  }
  auto results = batch_rule(input_value, input_bdim, hx_value, hx_bdim, w_ih_value, w_ih_bdim, w_hh_value, w_hh_bdim, b_ih_value, b_ih_bdim, b_hh_value, b_hh_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantized_gru_cell', 'quantized_rnn_relu_cell', 'quantized_rnn_tanh_cell']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_286_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_286_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, const Scalar &, const Scalar &>(
  batch_rule_286_t batch_rule,
  const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, const Scalar & scale_ih, const Scalar & scale_hh, const Scalar & zero_point_ih, const Scalar & zero_point_hh
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor hx_value;
  optional<int64_t> hx_bdim;
  std::tie(hx_value, hx_bdim) = unwrapTensorAtLevel(hx, cur_level);
  Tensor w_ih_value;
  optional<int64_t> w_ih_bdim;
  std::tie(w_ih_value, w_ih_bdim) = unwrapTensorAtLevel(w_ih, cur_level);
  Tensor w_hh_value;
  optional<int64_t> w_hh_bdim;
  std::tie(w_hh_value, w_hh_bdim) = unwrapTensorAtLevel(w_hh, cur_level);
  Tensor b_ih_value;
  optional<int64_t> b_ih_bdim;
  std::tie(b_ih_value, b_ih_bdim) = unwrapTensorAtLevel(b_ih, cur_level);
  Tensor b_hh_value;
  optional<int64_t> b_hh_bdim;
  std::tie(b_hh_value, b_hh_bdim) = unwrapTensorAtLevel(b_hh, cur_level);
  Tensor packed_ih_value;
  optional<int64_t> packed_ih_bdim;
  std::tie(packed_ih_value, packed_ih_bdim) = unwrapTensorAtLevel(packed_ih, cur_level);
  Tensor packed_hh_value;
  optional<int64_t> packed_hh_bdim;
  std::tie(packed_hh_value, packed_hh_bdim) = unwrapTensorAtLevel(packed_hh, cur_level);
  Tensor col_offsets_ih_value;
  optional<int64_t> col_offsets_ih_bdim;
  std::tie(col_offsets_ih_value, col_offsets_ih_bdim) = unwrapTensorAtLevel(col_offsets_ih, cur_level);
  Tensor col_offsets_hh_value;
  optional<int64_t> col_offsets_hh_bdim;
  std::tie(col_offsets_hh_value, col_offsets_hh_bdim) = unwrapTensorAtLevel(col_offsets_hh, cur_level);
  auto results = batch_rule(input_value, input_bdim, hx_value, hx_bdim, w_ih_value, w_ih_bdim, w_hh_value, w_hh_bdim, b_ih_value, b_ih_bdim, b_hh_value, b_hh_bdim, packed_ih_value, packed_ih_bdim, packed_hh_value, packed_hh_bdim, col_offsets_ih_value, col_offsets_ih_bdim, col_offsets_hh_value, col_offsets_hh_bdim, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_pack_padded_sequence']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_287_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_287_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, bool>(
  batch_rule_287_t batch_rule,
  const Tensor & input, const Tensor & lengths, bool batch_first
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor lengths_value;
  optional<int64_t> lengths_bdim;
  std::tie(lengths_value, lengths_bdim) = unwrapTensorAtLevel(lengths, cur_level);
  auto results = batch_rule(input_value, input_bdim, lengths_value, lengths_bdim, batch_first);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_pack_padded_sequence_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_288_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_288_t,Tensor,const Tensor &, IntArrayRef, const Tensor &, bool>(
  batch_rule_288_t batch_rule,
  const Tensor & grad, IntArrayRef input_size, const Tensor & batch_sizes, bool batch_first
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor batch_sizes_value;
  optional<int64_t> batch_sizes_bdim;
  std::tie(batch_sizes_value, batch_sizes_bdim) = unwrapTensorAtLevel(batch_sizes, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_size, batch_sizes_value, batch_sizes_bdim, batch_first);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_pad_packed_sequence']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_289_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, const Scalar &, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_289_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, bool, const Scalar &, int64_t>(
  batch_rule_289_t batch_rule,
  const Tensor & data, const Tensor & batch_sizes, bool batch_first, const Scalar & padding_value, int64_t total_length
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor data_value;
  optional<int64_t> data_bdim;
  std::tie(data_value, data_bdim) = unwrapTensorAtLevel(data, cur_level);
  Tensor batch_sizes_value;
  optional<int64_t> batch_sizes_bdim;
  std::tie(batch_sizes_value, batch_sizes_bdim) = unwrapTensorAtLevel(batch_sizes, cur_level);
  auto results = batch_rule(data_value, data_bdim, batch_sizes_value, batch_sizes_bdim, batch_first, padding_value, total_length);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['put']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_290_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_290_t,Tensor,const Tensor &, const Tensor &, const Tensor &, bool>(
  batch_rule_290_t batch_rule,
  const Tensor & self, const Tensor & index, const Tensor & source, bool accumulate
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  Tensor source_value;
  optional<int64_t> source_bdim;
  std::tie(source_value, source_bdim) = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, index_value, index_bdim, source_value, source_bdim, accumulate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_add']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_291_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_291_t,Tensor,const Tensor &, int64_t, const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_291_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, const Scalar & alpha
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  Tensor source_value;
  optional<int64_t> source_bdim;
  std::tie(source_value, source_bdim) = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_add.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_292_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_292_t,Tensor,const Tensor &, Dimname, const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_292_t batch_rule,
  const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source, const Scalar & alpha
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  Tensor source_value;
  optional<int64_t> source_bdim;
  std::tie(source_value, source_bdim) = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, source_value, source_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_fill.int_Scalar', 'scatter.value']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_293_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_293_t,Tensor,const Tensor &, int64_t, const Tensor &, const Scalar &>(
  batch_rule_293_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_fill.Dimname_Scalar', 'scatter.dimname_value']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_294_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_294_t,Tensor,const Tensor &, Dimname, const Tensor &, const Scalar &>(
  batch_rule_294_t batch_rule,
  const Tensor & self, Dimname dim, const Tensor & index, const Scalar & value
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['scatter.reduce']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_295_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_295_t,Tensor,const Tensor &, int64_t, const Tensor &, const Tensor &, c10::string_view>(
  batch_rule_295_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, c10::string_view reduce
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  Tensor src_value;
  optional<int64_t> src_bdim;
  std::tie(src_value, src_bdim) = unwrapTensorAtLevel(src, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim, reduce);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['scatter.value_reduce']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_296_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Scalar &, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_296_t,Tensor,const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view>(
  batch_rule_296_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, c10::string_view reduce
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value, reduce);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['scatter_reduce.two']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_297_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, c10::string_view, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_297_t,Tensor,const Tensor &, int64_t, const Tensor &, c10::string_view, c10::optional<int64_t>>(
  batch_rule_297_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, c10::string_view reduce, c10::optional<int64_t> output_size
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, reduce, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['diag_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_298_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_298_t,Tensor,const Tensor &, IntArrayRef, int64_t>(
  batch_rule_298_t batch_rule,
  const Tensor & grad, IntArrayRef input_sizes, int64_t diagonal
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_sizes, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cross', 'take_along_dim']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_299_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_299_t,Tensor,const Tensor &, const Tensor &, c10::optional<int64_t>>(
  batch_rule_299_t batch_rule,
  const Tensor & self, const Tensor & other, c10::optional<int64_t> dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_select']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_300_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_300_t,Tensor,const Tensor &, int64_t, const Tensor &>(
  batch_rule_300_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_select.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_301_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_301_t,Tensor,const Tensor &, Dimname, const Tensor &>(
  batch_rule_301_t batch_rule,
  const Tensor & self, Dimname dim, const Tensor & index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['index_select_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_302_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_302_t,Tensor,const Tensor &, IntArrayRef, int64_t, const Tensor &>(
  batch_rule_302_t batch_rule,
  const Tensor & grad, IntArrayRef self_sizes, int64_t dim, const Tensor & index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_sizes, dim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gather']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_303_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_303_t,Tensor,const Tensor &, int64_t, const Tensor &, bool>(
  batch_rule_303_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, sparse_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gather_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_304_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_304_t,Tensor,const Tensor &, const Tensor &, int64_t, const Tensor &, bool>(
  batch_rule_304_t batch_rule,
  const Tensor & grad, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, self_value, self_bdim, dim, index_value, index_bdim, sparse_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['gather.dimname']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_305_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_305_t,Tensor,const Tensor &, Dimname, const Tensor &, bool>(
  batch_rule_305_t batch_rule,
  const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor index_value;
  optional<int64_t> index_bdim;
  std::tie(index_value, index_bdim) = unwrapTensorAtLevel(index, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, sparse_grad);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['addcmul', 'addcdiv']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_306_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_306_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_306_t batch_rule,
  const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor tensor1_value;
  optional<int64_t> tensor1_bdim;
  std::tie(tensor1_value, tensor1_bdim) = unwrapTensorAtLevel(tensor1, cur_level);
  Tensor tensor2_value;
  optional<int64_t> tensor2_bdim;
  std::tie(tensor2_value, tensor2_bdim) = unwrapTensorAtLevel(tensor2, cur_level);
  auto results = batch_rule(self_value, self_bdim, tensor1_value, tensor1_bdim, tensor2_value, tensor2_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['cross_entropy_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_307_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_307_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t, double>(
  batch_rule_307_t batch_rule,
  const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, double label_smoothing
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, reduction, ignore_index, label_smoothing);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['lstsq', 'solve', '_solve_helper']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_308_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_308_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &>(
  batch_rule_308_t batch_rule,
  const Tensor & self, const Tensor & A
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(self_value, self_bdim, A_value, A_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['triangular_solve']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_309_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_309_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, bool, bool, bool>(
  batch_rule_309_t batch_rule,
  const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(self_value, self_bdim, A_value, A_bdim, upper, transpose, unitriangular);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['linalg_solve_triangular']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_310_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_310_t,Tensor,const Tensor &, const Tensor &, bool, bool, bool>(
  batch_rule_310_t batch_rule,
  const Tensor & self, const Tensor & B, bool upper, bool left, bool unitriangular
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor B_value;
  optional<int64_t> B_bdim;
  std::tie(B_value, B_bdim) = unwrapTensorAtLevel(B, cur_level);
  auto results = batch_rule(self_value, self_bdim, B_value, B_bdim, upper, left, unitriangular);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['svd', '_lu_with_info', 'linalg_lu_factor_ex', '_linalg_svd']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_311_t)(const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_311_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool, bool>(
  batch_rule_311_t batch_rule,
  const Tensor & self, bool some, bool compute_uv
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, some, compute_uv);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['ormqr']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_312_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_312_t,Tensor,const Tensor &, const Tensor &, const Tensor &, bool, bool>(
  batch_rule_312_t batch_rule,
  const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor input2_value;
  optional<int64_t> input2_bdim;
  std::tie(input2_value, input2_bdim) = unwrapTensorAtLevel(input2, cur_level);
  Tensor input3_value;
  optional<int64_t> input3_bdim;
  std::tie(input3_value, input3_bdim) = unwrapTensorAtLevel(input3, cur_level);
  auto results = batch_rule(self_value, self_bdim, input2_value, input2_bdim, input3_value, input3_bdim, left, transpose);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['lu_unpack']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_313_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_313_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, bool, bool>(
  batch_rule_313_t batch_rule,
  const Tensor & LU_data, const Tensor & LU_pivots, bool unpack_data, bool unpack_pivots
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor LU_data_value;
  optional<int64_t> LU_data_bdim;
  std::tie(LU_data_value, LU_data_bdim) = unwrapTensorAtLevel(LU_data, cur_level);
  Tensor LU_pivots_value;
  optional<int64_t> LU_pivots_bdim;
  std::tie(LU_pivots_value, LU_pivots_bdim) = unwrapTensorAtLevel(LU_pivots, cur_level);
  auto results = batch_rule(LU_data_value, LU_data_bdim, LU_pivots_value, LU_pivots_bdim, unpack_data, unpack_pivots);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['multinomial']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_314_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_314_t,Tensor,const Tensor &, int64_t, bool, c10::optional<Generator>>(
  batch_rule_314_t batch_rule,
  const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, num_samples, replacement, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['polygamma', 'special_polygamma']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_315_t)(int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_315_t,Tensor,int64_t, const Tensor &>(
  batch_rule_315_t batch_rule,
  int64_t n, const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(n, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['histc']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_316_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_316_t,Tensor,const Tensor &, int64_t, const Scalar &, const Scalar &>(
  batch_rule_316_t batch_rule,
  const Tensor & self, int64_t bins, const Scalar & min, const Scalar & max
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, bins, min, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['histogram.bins_tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_317_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_317_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, bool>(
  batch_rule_317_t batch_rule,
  const Tensor & self, const Tensor & bins, const c10::optional<Tensor> & weight, bool density
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor bins_value;
  optional<int64_t> bins_bdim;
  std::tie(bins_value, bins_bdim) = unwrapTensorAtLevel(bins, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins_value, bins_bdim, weight_value, weight_bdim, density);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['histogram.bin_ct']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_318_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<ArrayRef<double>>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_318_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, c10::optional<ArrayRef<double>>, const c10::optional<Tensor> &, bool>(
  batch_rule_318_t batch_rule,
  const Tensor & self, int64_t bins, c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins, range, weight_value, weight_bdim, density);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['_histogramdd_bin_edges']
typedef std::tuple<::std::vector<Tensor>,c10::optional<int64_t>> (*batch_rule_319_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ArrayRef<double>>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool);
template <>
::std::vector<Tensor> lowerToNextLayer<batch_rule_319_t,::std::vector<Tensor>,const Tensor &, IntArrayRef, c10::optional<ArrayRef<double>>, const c10::optional<Tensor> &, bool>(
  batch_rule_319_t batch_rule,
  const Tensor & self, IntArrayRef bins, c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins, range, weight_value, weight_bdim, density);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_histogramdd_from_bin_cts']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_320_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ArrayRef<double>>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_320_t,Tensor,const Tensor &, IntArrayRef, c10::optional<ArrayRef<double>>, const c10::optional<Tensor> &, bool>(
  batch_rule_320_t batch_rule,
  const Tensor & self, IntArrayRef bins, c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, bins, range, weight_value, weight_bdim, density);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantile', 'nanquantile']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_321_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, bool, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_321_t,Tensor,const Tensor &, const Tensor &, c10::optional<int64_t>, bool, c10::string_view>(
  batch_rule_321_t batch_rule,
  const Tensor & self, const Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor q_value;
  optional<int64_t> q_bdim;
  std::tie(q_value, q_bdim) = unwrapTensorAtLevel(q, cur_level);
  auto results = batch_rule(self_value, self_bdim, q_value, q_bdim, dim, keepdim, interpolation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['quantile.scalar', 'nanquantile.scalar']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_322_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<int64_t>, bool, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_322_t,Tensor,const Tensor &, double, c10::optional<int64_t>, bool, c10::string_view>(
  batch_rule_322_t batch_rule,
  const Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, q, dim, keepdim, interpolation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['sort.stable']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_323_t)(const Tensor &, c10::optional<int64_t>, c10::optional<bool>, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_323_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::optional<bool>, int64_t, bool>(
  batch_rule_323_t batch_rule,
  const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, stable, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['sort.dimname_stable']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_324_t)(const Tensor &, c10::optional<int64_t>, c10::optional<bool>, Dimname, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_324_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::optional<bool>, Dimname, bool>(
  batch_rule_324_t batch_rule,
  const Tensor & self, c10::optional<bool> stable, Dimname dim, bool descending
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, stable, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['topk']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_325_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_325_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, int64_t, bool, bool>(
  batch_rule_325_t batch_rule,
  const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, k, dim, largest, sorted);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['renorm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_326_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, int64_t, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_326_t,Tensor,const Tensor &, const Scalar &, int64_t, const Scalar &>(
  batch_rule_326_t batch_rule,
  const Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p, dim, maxnorm);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['normal.float_Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_327_t)(double, const Tensor &, c10::optional<int64_t>, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_327_t,Tensor,double, const Tensor &, c10::optional<Generator>>(
  batch_rule_327_t batch_rule,
  double mean, const Tensor & std, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor std_value;
  optional<int64_t> std_bdim;
  std::tie(std_value, std_bdim) = unwrapTensorAtLevel(std, cur_level);
  auto results = batch_rule(mean, std_value, std_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['searchsorted.Tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_328_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, c10::optional<c10::string_view>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_328_t,Tensor,const Tensor &, const Tensor &, bool, bool, c10::optional<c10::string_view>, const c10::optional<Tensor> &>(
  batch_rule_328_t batch_rule,
  const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor sorted_sequence_value;
  optional<int64_t> sorted_sequence_bdim;
  std::tie(sorted_sequence_value, sorted_sequence_bdim) = unwrapTensorAtLevel(sorted_sequence, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> sorter_value;
  optional<int64_t> sorter_bdim;
  if (sorter) {
      std::tie(sorter_value, sorter_bdim) = unwrapTensorAtLevel(sorter.value(), cur_level);
  }
  auto results = batch_rule(sorted_sequence_value, sorted_sequence_bdim, self_value, self_bdim, out_int32, right, side, sorter_value, sorter_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['searchsorted.Scalar']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_329_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, bool, bool, c10::optional<c10::string_view>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_329_t,Tensor,const Tensor &, const Scalar &, bool, bool, c10::optional<c10::string_view>, const c10::optional<Tensor> &>(
  batch_rule_329_t batch_rule,
  const Tensor & sorted_sequence, const Scalar & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor sorted_sequence_value;
  optional<int64_t> sorted_sequence_bdim;
  std::tie(sorted_sequence_value, sorted_sequence_bdim) = unwrapTensorAtLevel(sorted_sequence, cur_level);
  optional<Tensor> sorter_value;
  optional<int64_t> sorter_bdim;
  if (sorter) {
      std::tie(sorter_value, sorter_bdim) = unwrapTensorAtLevel(sorter.value(), cur_level);
  }
  auto results = batch_rule(sorted_sequence_value, sorted_sequence_bdim, self, out_int32, right, side, sorter_value, sorter_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['multi_margin_loss']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_330_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_330_t,Tensor,const Tensor &, const Tensor &, const Scalar &, const Scalar &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_330_t batch_rule,
  const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, p, margin, weight_value, weight_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['multi_margin_loss_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_331_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_331_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_331_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, p, margin, weight_value, weight_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['multilabel_margin_loss_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_332_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_332_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(
  batch_rule_332_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  Tensor is_target_value;
  optional<int64_t> is_target_bdim;
  std::tie(is_target_value, is_target_bdim) = unwrapTensorAtLevel(is_target, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction, is_target_value, is_target_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['nll_loss_nd', 'nll_loss', 'nll_loss2d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_333_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_333_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t>(
  batch_rule_333_t batch_rule,
  const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, reduction, ignore_index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['nll_loss_forward', 'nll_loss2d_forward']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_334_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_334_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t>(
  batch_rule_334_t batch_rule,
  const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, reduction, ignore_index);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['nll_loss_backward', 'nll_loss2d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_335_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_335_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t, const Tensor &>(
  batch_rule_335_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  Tensor total_weight_value;
  optional<int64_t> total_weight_bdim;
  std::tie(total_weight_value, total_weight_bdim) = unwrapTensorAtLevel(total_weight, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, weight_value, weight_bdim, reduction, ignore_index, total_weight_value, total_weight_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['smooth_l1_loss_backward', 'huber_loss_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_336_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_336_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, double>(
  batch_rule_336_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double beta
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor target_value;
  optional<int64_t> target_bdim;
  std::tie(target_value, target_bdim) = unwrapTensorAtLevel(target, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction, beta);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['elu']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_337_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_337_t,Tensor,const Tensor &, const Scalar &, const Scalar &, const Scalar &>(
  batch_rule_337_t batch_rule,
  const Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, alpha, scale, input_scale);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['elu_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_338_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Scalar &, bool, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_338_t,Tensor,const Tensor &, const Scalar &, const Scalar &, const Scalar &, bool, const Tensor &>(
  batch_rule_338_t batch_rule,
  const Tensor & grad_output, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale, bool is_result, const Tensor & self_or_result
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_or_result_value;
  optional<int64_t> self_or_result_bdim;
  std::tie(self_or_result_value, self_or_result_bdim) = unwrapTensorAtLevel(self_or_result, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, alpha, scale, input_scale, is_result, self_or_result_value, self_or_result_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['hardtanh_backward', 'softplus_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_339_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_339_t,Tensor,const Tensor &, const Tensor &, const Scalar &, const Scalar &>(
  batch_rule_339_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Scalar & min_val, const Scalar & max_val
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, min_val, max_val);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['leaky_relu_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_340_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, bool);
template <>
Tensor lowerToNextLayer<batch_rule_340_t,Tensor,const Tensor &, const Tensor &, const Scalar &, bool>(
  batch_rule_340_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Scalar & negative_slope, bool self_is_result
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, negative_slope, self_is_result);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['rrelu_with_noise']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_341_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, bool, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_341_t,Tensor,const Tensor &, const Tensor &, const Scalar &, const Scalar &, bool, c10::optional<Generator>>(
  batch_rule_341_t batch_rule,
  const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, c10::optional<Generator> generator
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor noise_value;
  optional<int64_t> noise_bdim;
  std::tie(noise_value, noise_bdim) = unwrapTensorAtLevel(noise, cur_level);
  auto results = batch_rule(self_value, self_bdim, noise_value, noise_bdim, lower, upper, training, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['rrelu_with_noise_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_342_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_342_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, bool, bool>(
  batch_rule_342_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, bool self_is_result
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor noise_value;
  optional<int64_t> noise_bdim;
  std::tie(noise_value, noise_bdim) = unwrapTensorAtLevel(noise, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, noise_value, noise_bdim, lower, upper, training, self_is_result);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['avg_pool2d', 'avg_pool3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_343_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_343_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(
  batch_rule_343_t batch_rule,
  const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['avg_pool2d_backward', 'avg_pool3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_344_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_344_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(
  batch_rule_344_t batch_rule,
  const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fractional_max_pool2d', 'fractional_max_pool3d']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_345_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_345_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(
  batch_rule_345_t batch_rule,
  const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor random_samples_value;
  optional<int64_t> random_samples_bdim;
  std::tie(random_samples_value, random_samples_bdim) = unwrapTensorAtLevel(random_samples, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, output_size, random_samples_value, random_samples_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['fractional_max_pool2d_backward', 'fractional_max_pool3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_346_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_346_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(
  batch_rule_346_t batch_rule,
  const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, kernel_size, output_size, indices_value, indices_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['max_pool2d_with_indices_backward', 'max_pool3d_with_indices_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_347_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_347_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(
  batch_rule_347_t batch_rule,
  const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, kernel_size, stride, padding, dilation, ceil_mode, indices_value, indices_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['max_unpool2d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_348_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_348_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(
  batch_rule_348_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, indices_value, indices_bdim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['max_unpool3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_349_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_349_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_349_t batch_rule,
  const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(self_value, self_bdim, indices_value, indices_bdim, output_size, stride, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['max_unpool3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_350_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_350_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_350_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, indices_value, indices_bdim, output_size, stride, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_linear1d.vec', 'upsample_bilinear2d.vec', '_upsample_bilinear2d_aa.vec', 'upsample_trilinear3d.vec', 'upsample_bicubic2d.vec', '_upsample_bicubic2d_aa.vec']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_351_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, bool, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_351_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, bool, c10::optional<ArrayRef<double>>>(
  batch_rule_351_t batch_rule,
  const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_linear1d_backward.vec', 'upsample_bilinear2d_backward.vec', '_upsample_bilinear2d_aa_backward.vec', 'upsample_trilinear3d_backward.vec', 'upsample_bicubic2d_backward.vec', '_upsample_bicubic2d_aa_backward.vec']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_352_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, IntArrayRef, bool, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_352_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, IntArrayRef, bool, c10::optional<ArrayRef<double>>>(
  batch_rule_352_t batch_rule,
  const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest1d.vec', '_upsample_nearest_exact1d.vec', 'upsample_nearest2d.vec', '_upsample_nearest_exact2d.vec', 'upsample_nearest3d.vec', '_upsample_nearest_exact3d.vec']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_353_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_353_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, c10::optional<ArrayRef<double>>>(
  batch_rule_353_t batch_rule,
  const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest1d_backward.vec', '_upsample_nearest_exact1d_backward.vec', 'upsample_nearest2d_backward.vec', '_upsample_nearest_exact2d_backward.vec', 'upsample_nearest3d_backward.vec', '_upsample_nearest_exact3d_backward.vec']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_354_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_354_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<ArrayRef<double>>>(
  batch_rule_354_t batch_rule,
  const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_linear1d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_355_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_355_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<double>>(
  batch_rule_355_t batch_rule,
  const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, align_corners, scales);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_linear1d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_356_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, bool, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_356_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(
  batch_rule_356_t batch_rule,
  const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scales);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_bilinear2d', '_upsample_bilinear2d_aa', 'upsample_bicubic2d', '_upsample_bicubic2d_aa']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_357_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_357_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(
  batch_rule_357_t batch_rule,
  const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, align_corners, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_bilinear2d_backward', '_upsample_bilinear2d_aa_backward', 'upsample_bicubic2d_backward', '_upsample_bicubic2d_aa_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_358_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_358_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(
  batch_rule_358_t batch_rule,
  const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_trilinear3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_359_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_359_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_359_t batch_rule,
  const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, align_corners, scales_d, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_trilinear3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_360_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_360_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_360_t batch_rule,
  const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest1d', '_upsample_nearest_exact1d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_361_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_361_t,Tensor,const Tensor &, IntArrayRef, c10::optional<double>>(
  batch_rule_361_t batch_rule,
  const Tensor & self, IntArrayRef output_size, c10::optional<double> scales
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, scales);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest1d_backward', '_upsample_nearest_exact1d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_362_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_362_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(
  batch_rule_362_t batch_rule,
  const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, scales);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest2d', '_upsample_nearest_exact2d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_363_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_363_t,Tensor,const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(
  batch_rule_363_t batch_rule,
  const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest2d_backward', '_upsample_nearest_exact2d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_364_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_364_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(
  batch_rule_364_t batch_rule,
  const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest3d', '_upsample_nearest_exact3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_365_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_365_t,Tensor,const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_365_t batch_rule,
  const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, scales_d, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['upsample_nearest3d_backward', '_upsample_nearest_exact3d_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_366_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_366_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_366_t batch_rule,
  const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, scales_d, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['logit_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_367_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_367_t,Tensor,const Tensor &, const Tensor &, c10::optional<double>>(
  batch_rule_367_t batch_rule,
  const Tensor & grad_output, const Tensor & self, c10::optional<double> eps
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, eps);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['slow_conv_transpose2d', 'slow_conv_transpose3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_368_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_368_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_368_t batch_rule,
  const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, kernel_size, bias_value, bias_bdim, stride, padding, output_padding, dilation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['thnn_conv2d', '_slow_conv2d_forward', 'slow_conv3d', 'slow_conv3d_forward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_369_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_369_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef>(
  batch_rule_369_t batch_rule,
  const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, kernel_size, bias_value, bias_bdim, stride, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_slow_conv2d_backward.output_mask']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_370_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, ::std::array<bool,3>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_370_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, ::std::array<bool,3>>(
  batch_rule_370_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, ::std::array<bool,3> output_mask
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, weight_value, weight_bdim, kernel_size, stride, padding, output_mask);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_conv_depthwise2d', 'conv_depthwise3d', 'slow_conv_dilated2d', 'slow_conv_dilated3d']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_371_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_371_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_371_t batch_rule,
  const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, kernel_size, bias_value, bias_bdim, stride, padding, dilation);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['col2im', 'im2col_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_372_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_372_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_372_t batch_rule,
  const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, kernel_size, dilation, padding, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['col2im_backward', 'im2col']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_373_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_373_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_373_t batch_rule,
  const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, kernel_size, dilation, padding, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fft_fft', 'fft_ifft', 'fft_rfft', 'fft_irfft', 'fft_hfft', 'fft_ihfft']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_374_t)(const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, int64_t, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_374_t,Tensor,const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<c10::string_view>>(
  batch_rule_374_t batch_rule,
  const Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, n, dim, norm);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fft_fft2', 'fft_ifft2', 'fft_rfft2', 'fft_irfft2', 'fft_hfft2', 'fft_ihfft2']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_375_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_375_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<c10::string_view>>(
  batch_rule_375_t batch_rule,
  const Tensor & self, c10::optional<IntArrayRef> s, IntArrayRef dim, c10::optional<c10::string_view> norm
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, s, dim, norm);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fft_fftn', 'fft_ifftn', 'fft_rfftn', 'fft_irfftn', 'fft_hfftn', 'fft_ihfftn']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_376_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<IntArrayRef>, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_376_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, c10::optional<IntArrayRef>, c10::optional<c10::string_view>>(
  batch_rule_376_t batch_rule,
  const Tensor & self, c10::optional<IntArrayRef> s, c10::optional<IntArrayRef> dim, c10::optional<c10::string_view> norm
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, s, dim, norm);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['fft_fftshift', 'fft_ifftshift', '_test_optional_intlist', '_test_optional_filled_intlist']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_377_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>);
template <>
Tensor lowerToNextLayer<batch_rule_377_t,Tensor,const Tensor &, c10::optional<IntArrayRef>>(
  batch_rule_377_t batch_rule,
  const Tensor & self, c10::optional<IntArrayRef> dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_det_lu_based_helper']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_378_t)(const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_378_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &>(
  batch_rule_378_t batch_rule,
  const Tensor & self
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['_det_lu_based_helper_backward_helper']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_379_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_379_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(
  batch_rule_379_t batch_rule,
  const Tensor & det_grad, const Tensor & det, const Tensor & self, const Tensor & lu, const Tensor & pivs
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor det_grad_value;
  optional<int64_t> det_grad_bdim;
  std::tie(det_grad_value, det_grad_bdim) = unwrapTensorAtLevel(det_grad, cur_level);
  Tensor det_value;
  optional<int64_t> det_bdim;
  std::tie(det_value, det_bdim) = unwrapTensorAtLevel(det, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor lu_value;
  optional<int64_t> lu_bdim;
  std::tie(lu_value, lu_bdim) = unwrapTensorAtLevel(lu, cur_level);
  Tensor pivs_value;
  optional<int64_t> pivs_bdim;
  std::tie(pivs_value, pivs_bdim) = unwrapTensorAtLevel(pivs, cur_level);
  auto results = batch_rule(det_grad_value, det_grad_bdim, det_value, det_bdim, self_value, self_bdim, lu_value, lu_bdim, pivs_value, pivs_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_lstsq']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_380_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<double>, c10::optional<c10::string_view>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_380_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, c10::optional<double>, c10::optional<c10::string_view>>(
  batch_rule_380_t batch_rule,
  const Tensor & self, const Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor b_value;
  optional<int64_t> b_bdim;
  std::tie(b_value, b_bdim) = unwrapTensorAtLevel(b, cur_level);
  auto results = batch_rule(self_value, self_bdim, b_value, b_bdim, rcond, driver);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level), makeBatched(std::get<6>(results), std::get<7>(results), cur_level));
}

// ['linalg_eigh', 'linalg_qr', '_linalg_qr_helper']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_381_t)(const Tensor &, c10::optional<int64_t>, c10::string_view);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_381_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::string_view>(
  batch_rule_381_t batch_rule,
  const Tensor & self, c10::string_view UPLO
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, UPLO);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

// ['linalg_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_382_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_382_t,Tensor,const Tensor &, const c10::optional<Scalar> &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>>(
  batch_rule_382_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, ord, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_norm.ord_str']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_383_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_383_t,Tensor,const Tensor &, c10::string_view, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>>(
  batch_rule_383_t batch_rule,
  const Tensor & self, c10::string_view ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, ord, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_vector_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_384_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_384_t,Tensor,const Tensor &, const Scalar &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>>(
  batch_rule_384_t batch_rule,
  const Tensor & self, const Scalar & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, ord, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_matrix_norm']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_385_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_385_t,Tensor,const Tensor &, const Scalar &, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_385_t batch_rule,
  const Tensor & self, const Scalar & ord, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, ord, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_matrix_norm.str_ord']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_386_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_386_t,Tensor,const Tensor &, c10::string_view, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_386_t batch_rule,
  const Tensor & self, c10::string_view ord, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, ord, dim, keepdim, dtype);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_svd']
typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_387_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_387_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool>(
  batch_rule_387_t batch_rule,
  const Tensor & A, bool full_matrices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(A_value, A_bdim, full_matrices);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

// ['linalg_cond']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_388_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &);
template <>
Tensor lowerToNextLayer<batch_rule_388_t,Tensor,const Tensor &, const c10::optional<Scalar> &>(
  batch_rule_388_t batch_rule,
  const Tensor & self, const c10::optional<Scalar> & p
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_pinv.atol_rtol_tensor', 'linalg_matrix_rank.atol_rtol_tensor']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_389_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_389_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool>(
  batch_rule_389_t batch_rule,
  const Tensor & self, const c10::optional<Tensor> & atol, const c10::optional<Tensor> & rtol, bool hermitian
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  optional<Tensor> atol_value;
  optional<int64_t> atol_bdim;
  if (atol) {
      std::tie(atol_value, atol_bdim) = unwrapTensorAtLevel(atol.value(), cur_level);
  }
  optional<Tensor> rtol_value;
  optional<int64_t> rtol_bdim;
  if (rtol) {
      std::tie(rtol_value, rtol_bdim) = unwrapTensorAtLevel(rtol.value(), cur_level);
  }
  auto results = batch_rule(self_value, self_bdim, atol_value, atol_bdim, rtol_value, rtol_bdim, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_pinv.atol_rtol_float', 'linalg_matrix_rank.atol_rtol_float']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_390_t)(const Tensor &, c10::optional<int64_t>, c10::optional<double>, c10::optional<double>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_390_t,Tensor,const Tensor &, c10::optional<double>, c10::optional<double>, bool>(
  batch_rule_390_t batch_rule,
  const Tensor & self, c10::optional<double> atol, c10::optional<double> rtol, bool hermitian
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, atol, rtol, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['linalg_tensorsolve']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_391_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>);
template <>
Tensor lowerToNextLayer<batch_rule_391_t,Tensor,const Tensor &, const Tensor &, c10::optional<IntArrayRef>>(
  batch_rule_391_t batch_rule,
  const Tensor & self, const Tensor & other, c10::optional<IntArrayRef> dims
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_test_optional_floatlist']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_392_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_392_t,Tensor,const Tensor &, c10::optional<ArrayRef<double>>>(
  batch_rule_392_t batch_rule,
  const Tensor & values, c10::optional<ArrayRef<double>> addends
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  auto results = batch_rule(values_value, values_bdim, addends);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_test_string_default']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_393_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_393_t,Tensor,const Tensor &, c10::string_view, c10::string_view>(
  batch_rule_393_t batch_rule,
  const Tensor & dummy, c10::string_view a, c10::string_view b
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor dummy_value;
  optional<int64_t> dummy_bdim;
  std::tie(dummy_value, dummy_bdim) = unwrapTensorAtLevel(dummy, cur_level);
  auto results = batch_rule(dummy_value, dummy_bdim, a, b);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_test_ambiguous_defaults.b']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_394_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_394_t,Tensor,const Tensor &, int64_t, c10::string_view>(
  batch_rule_394_t batch_rule,
  const Tensor & dummy, int64_t a, c10::string_view b
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor dummy_value;
  optional<int64_t> dummy_bdim;
  std::tie(dummy_value, dummy_bdim) = unwrapTensorAtLevel(dummy, cur_level);
  auto results = batch_rule(dummy_value, dummy_bdim, a, b);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['segment_reduce']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_395_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, bool, const c10::optional<Scalar> &);
template <>
Tensor lowerToNextLayer<batch_rule_395_t,Tensor,const Tensor &, c10::string_view, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t, bool, const c10::optional<Scalar> &>(
  batch_rule_395_t batch_rule,
  const Tensor & data, c10::string_view reduce, const c10::optional<Tensor> & lengths, const c10::optional<Tensor> & indices, int64_t axis, bool unsafe, const c10::optional<Scalar> & initial
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor data_value;
  optional<int64_t> data_bdim;
  std::tie(data_value, data_bdim) = unwrapTensorAtLevel(data, cur_level);
  optional<Tensor> lengths_value;
  optional<int64_t> lengths_bdim;
  if (lengths) {
      std::tie(lengths_value, lengths_bdim) = unwrapTensorAtLevel(lengths.value(), cur_level);
  }
  optional<Tensor> indices_value;
  optional<int64_t> indices_bdim;
  if (indices) {
      std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices.value(), cur_level);
  }
  auto results = batch_rule(data_value, data_bdim, reduce, lengths_value, lengths_bdim, indices_value, indices_bdim, axis, unsafe, initial);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

// ['_segment_reduce_backward']
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_396_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::string_view, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_396_t,Tensor,const Tensor &, const Tensor &, const Tensor &, c10::string_view, const c10::optional<Tensor> &, int64_t>(
  batch_rule_396_t batch_rule,
  const Tensor & grad, const Tensor & output, const Tensor & data, c10::string_view reduce, const c10::optional<Tensor> & lengths, int64_t axis
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor output_value;
  optional<int64_t> output_bdim;
  std::tie(output_value, output_bdim) = unwrapTensorAtLevel(output, cur_level);
  Tensor data_value;
  optional<int64_t> data_bdim;
  std::tie(data_value, data_bdim) = unwrapTensorAtLevel(data, cur_level);
  optional<Tensor> lengths_value;
  optional<int64_t> lengths_bdim;
  if (lengths) {
      std::tie(lengths_value, lengths_bdim) = unwrapTensorAtLevel(lengths.value(), cur_level);
  }
  auto results = batch_rule(grad_value, grad_bdim, output_value, output_bdim, data_value, data_bdim, reduce, lengths_value, lengths_bdim, axis);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

}}
