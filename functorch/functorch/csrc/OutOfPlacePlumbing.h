#pragma once
#include <functorch/csrc/OutOfPlacePlumbing.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>

namespace at { namespace functorch {
template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Byte_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Char_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Double_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Float_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Int_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Long_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Short_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cast_Half_generated_plumbing(const Tensor & self, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor data_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_leaf_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t output_nr_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t _version_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool retains_grad_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _fw_primal_generated_plumbing(const Tensor & self, int64_t level) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _make_dual_generated_plumbing(const Tensor & primal, const Tensor & tangent, int64_t level) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _unpack_dual_generated_plumbing(const Tensor & dual, int64_t level) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _new_zeros_with_same_feature_meta_generated_plumbing(const Tensor & self, const Tensor & other, int64_t self_num_batch_dims) {
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
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, self_num_batch_dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
bool _has_same_storage_numel_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rename_generated_plumbing(const Tensor & self, c10::optional<DimnameList> names) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor align_to_generated_plumbing(const Tensor & self, DimnameList names) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor align_to_ellipsis_idx_generated_plumbing(const Tensor & self, DimnameList order, int64_t ellipsis_idx) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor align_as_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor refine_names_generated_plumbing(const Tensor & self, DimnameList names) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool _use_cudnn_ctc_loss_generated_plumbing(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _cudnn_ctc_loss_generated_plumbing(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t _debug_has_internal_overlap_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _fused_dropout_generated_plumbing(const Tensor & self, double p, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _masked_scale_generated_plumbing(const Tensor & self, const Tensor & mask, double scale) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> native_dropout_generated_plumbing(const Tensor & input, double p, c10::optional<bool> train) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor native_dropout_backward_generated_plumbing(const Tensor & grad_output, const Tensor & mask, double scale) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor mask_value;
  optional<int64_t> mask_bdim;
  std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, mask_value, mask_bdim, scale);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _sobol_engine_draw_generated_plumbing(const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _reshape_from_tensor_generated_plumbing(const Tensor & self, const Tensor & shape) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor shape_value;
  optional<int64_t> shape_bdim;
  std::tie(shape_value, shape_bdim) = unwrapTensorAtLevel(shape, cur_level);
  auto results = batch_rule(self_value, self_bdim, shape_value, shape_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _shape_as_tensor_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor dropout_generated_plumbing(const Tensor & input, double p, bool train) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor feature_dropout_generated_plumbing(const Tensor & input, double p, bool train) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor alpha_dropout_generated_plumbing(const Tensor & input, double p, bool train) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor feature_alpha_dropout_generated_plumbing(const Tensor & input, double p, bool train) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor abs_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor absolute_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor angle_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor view_as_real_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor view_as_complex_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sgn_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor real_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor imag_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _conj_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conj_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _conj_physical_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conj_physical_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor resolve_conj_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor resolve_neg_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _neg_view_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor acos_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor arccos_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor avg_pool1d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor adaptive_avg_pool1d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> adaptive_max_pool1d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor add_Tensor_generated_plumbing(const Tensor & self, const Tensor & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _add_relu_Tensor_generated_plumbing(const Tensor & self, const Tensor & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _add_relu_Scalar_generated_plumbing(const Tensor & self, const Scalar & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor add_Scalar_generated_plumbing(const Tensor & self, const Scalar & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor addmv_generated_plumbing(const Tensor & self, const Tensor & mat, const Tensor & vec, const Scalar & beta, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor addr_generated_plumbing(const Tensor & self, const Tensor & vec1, const Tensor & vec2, const Scalar & beta, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor vec1_value;
  optional<int64_t> vec1_bdim;
  std::tie(vec1_value, vec1_bdim) = unwrapTensorAtLevel(vec1, cur_level);
  Tensor vec2_value;
  optional<int64_t> vec2_bdim;
  std::tie(vec2_value, vec2_bdim) = unwrapTensorAtLevel(vec2, cur_level);
  auto results = batch_rule(self_value, self_bdim, vec1_value, vec1_bdim, vec2_value, vec2_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor affine_grid_generator_generated_plumbing(const Tensor & theta, IntArrayRef size, bool align_corners) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor affine_grid_generator_backward_generated_plumbing(const Tensor & grad, IntArrayRef size, bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, size, align_corners);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor all_dim_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor all_dimname_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool allclose_generated_plumbing(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor any_dim_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor any_dimname_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _dim_arange_generated_plumbing(const Tensor & like, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor like_value;
  optional<int64_t> like_bdim;
  std::tie(like_value, like_bdim) = unwrapTensorAtLevel(like, cur_level);
  auto results = batch_rule(like_value, like_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor argmax_generated_plumbing(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor argmin_generated_plumbing(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor acosh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor arccosh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor asinh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor arcsinh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor atanh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor arctanh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor as_strided_generated_plumbing(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor asin_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor arcsin_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor atan_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor arctan_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor atleast_1d_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor atleast_2d_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor atleast_3d_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor baddbmm_generated_plumbing(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor batch1_value;
  optional<int64_t> batch1_bdim;
  std::tie(batch1_value, batch1_bdim) = unwrapTensorAtLevel(batch1, cur_level);
  Tensor batch2_value;
  optional<int64_t> batch2_bdim;
  std::tie(batch2_value, batch2_bdim) = unwrapTensorAtLevel(batch2, cur_level);
  auto results = batch_rule(self_value, self_bdim, batch1_value, batch1_bdim, batch2_value, batch2_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor batch_norm_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantized_batch_norm_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> _batch_norm_impl_index_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _batch_norm_impl_index_backward_generated_plumbing(int64_t impl_index, const Tensor & input, const Tensor & grad_output, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_var_transform, bool train, double eps, ::std::array<bool,3> output_mask, const Tensor & reservedSpace) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bernoulli_generated_plumbing(const Tensor & self, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bernoulli_p_generated_plumbing(const Tensor & self, double p, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bilinear_generated_plumbing(const Tensor & input1, const Tensor & input2, const Tensor & weight, const c10::optional<Tensor> & bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor binary_cross_entropy_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor binary_cross_entropy_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor binary_cross_entropy_with_logits_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & pos_weight, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor binary_cross_entropy_with_logits_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & pos_weight, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bincount_generated_plumbing(const Tensor & self, const c10::optional<Tensor> & weights, int64_t minlength) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_not_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor copysign_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor copysign_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logical_not_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logical_xor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logical_and_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logical_or_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bmm_generated_plumbing(const Tensor & self, const Tensor & mat2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mat2_value;
  optional<int64_t> mat2_bdim;
  std::tie(mat2_value, mat2_bdim) = unwrapTensorAtLevel(mat2, cur_level);
  auto results = batch_rule(self_value, self_bdim, mat2_value, mat2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor broadcast_to_generated_plumbing(const Tensor & self, IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_broadcast_to_generated_plumbing(const Tensor & self, IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ceil_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> unsafe_chunk_generated_plumbing(const Tensor & self, int64_t chunks, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> chunk_generated_plumbing(const Tensor & self, int64_t chunks, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> tensor_split_sections_generated_plumbing(const Tensor & self, int64_t sections, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sections, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> tensor_split_indices_generated_plumbing(const Tensor & self, IntArrayRef indices, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> tensor_split_tensor_indices_or_sections_generated_plumbing(const Tensor & self, const Tensor & tensor_indices_or_sections, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clamp_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clamp_Tensor_generated_plumbing(const Tensor & self, const c10::optional<Tensor> & min, const c10::optional<Tensor> & max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clamp_max_generated_plumbing(const Tensor & self, const Scalar & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clamp_max_Tensor_generated_plumbing(const Tensor & self, const Tensor & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor max_value;
  optional<int64_t> max_bdim;
  std::tie(max_value, max_bdim) = unwrapTensorAtLevel(max, cur_level);
  auto results = batch_rule(self_value, self_bdim, max_value, max_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clamp_min_generated_plumbing(const Tensor & self, const Scalar & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clamp_min_Tensor_generated_plumbing(const Tensor & self, const Tensor & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor min_value;
  optional<int64_t> min_bdim;
  std::tie(min_value, min_bdim) = unwrapTensorAtLevel(min, cur_level);
  auto results = batch_rule(self_value, self_bdim, min_value, min_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clip_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clip_Tensor_generated_plumbing(const Tensor & self, const c10::optional<Tensor> & min, const c10::optional<Tensor> & max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool cudnn_is_acceptable_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor complex_generated_plumbing(const Tensor & real, const Tensor & imag) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor real_value;
  optional<int64_t> real_bdim;
  std::tie(real_value, real_bdim) = unwrapTensorAtLevel(real, cur_level);
  Tensor imag_value;
  optional<int64_t> imag_bdim;
  std::tie(imag_value, imag_bdim) = unwrapTensorAtLevel(imag, cur_level);
  auto results = batch_rule(real_value, real_bdim, imag_value, imag_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor polar_generated_plumbing(const Tensor & abs, const Tensor & angle) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor abs_value;
  optional<int64_t> abs_bdim;
  std::tie(abs_value, abs_bdim) = unwrapTensorAtLevel(abs, cur_level);
  Tensor angle_value;
  optional<int64_t> angle_bdim;
  std::tie(angle_value, angle_bdim) = unwrapTensorAtLevel(angle, cur_level);
  auto results = batch_rule(abs_value, abs_bdim, angle_value, angle_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor constant_pad_nd_generated_plumbing(const Tensor & self, IntArrayRef pad, const Scalar & value) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor contiguous_generated_plumbing(const Tensor & self, MemoryFormat memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor convolution_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> convolution_backward_generated_plumbing(const Tensor & grad_output, const Tensor & input, const Tensor & weight, c10::optional<IntArrayRef> bias_sizes, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor convolution_overrideable_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable_generated_plumbing(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _convolution_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _convolution_deprecated_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _convolution_mode_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, c10::string_view padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward_generated_plumbing(const c10::optional<Tensor> & ggI, const c10::optional<Tensor> & ggW, const c10::optional<Tensor> & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv1d_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv2d_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv3d_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv1d_padding_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, c10::string_view padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv2d_padding_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, c10::string_view padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv3d_padding_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, c10::string_view padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv_tbc_generated_plumbing(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward_generated_plumbing(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv_transpose1d_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv_transpose2d_input_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv_transpose3d_input_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _copy_from_generated_plumbing(const Tensor & self, const Tensor & dst, bool non_blocking) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _copy_from_and_resize_generated_plumbing(const Tensor & self, const Tensor & dst) {
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
  auto results = batch_rule(self_value, self_bdim, dst_value, dst_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cos_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cosh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cosine_embedding_loss_generated_plumbing(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor count_nonzero_dim_IntList_generated_plumbing(const Tensor & self, IntArrayRef dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor count_nonzero_generated_plumbing(const Tensor & self, c10::optional<int64_t> dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cov_generated_plumbing(const Tensor & self, int64_t correction, const c10::optional<Tensor> & fweights, const c10::optional<Tensor> & aweights) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor corrcoef_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cudnn_affine_grid_generator_generated_plumbing(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cudnn_affine_grid_generator_backward_generated_plumbing(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, N, C, H, W);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor> cudnn_batch_norm_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward_generated_plumbing(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_var, double epsilon, const Tensor & reserveSpace) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cudnn_convolution_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cudnn_convolution_transpose_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cudnn_convolution_relu_generated_plumbing(const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
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
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, dilation, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cudnn_convolution_add_relu_generated_plumbing(const Tensor & self, const Tensor & weight, const Tensor & z, const c10::optional<Scalar> & alpha, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cudnn_grid_sampler_generated_plumbing(const Tensor & self, const Tensor & grid) {
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
  auto results = batch_rule(self_value, self_bdim, grid_value, grid_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward_generated_plumbing(const Tensor & self, const Tensor & grid, const Tensor & grad_output) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> cummax_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> cummax_dimname_generated_plumbing(const Tensor & self, Dimname dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> cummin_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> cummin_dimname_generated_plumbing(const Tensor & self, Dimname dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cummaxmin_backward_generated_plumbing(const Tensor & grad, const Tensor & input, const Tensor & indices, int64_t dim) {
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
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim, indices_value, indices_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cumprod_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cumprod_dimname_generated_plumbing(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cumprod_backward_generated_plumbing(const Tensor & grad, const Tensor & input, int64_t dim, const Tensor & output) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cumsum_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cumsum_dimname_generated_plumbing(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cumulative_trapezoid_x_generated_plumbing(const Tensor & y, const Tensor & x, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor y_value;
  optional<int64_t> y_bdim;
  std::tie(y_value, y_bdim) = unwrapTensorAtLevel(y, cur_level);
  Tensor x_value;
  optional<int64_t> x_bdim;
  std::tie(x_value, x_bdim) = unwrapTensorAtLevel(x, cur_level);
  auto results = batch_rule(y_value, y_bdim, x_value, x_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cumulative_trapezoid_dx_generated_plumbing(const Tensor & y, const Scalar & dx, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ctc_loss_IntList_generated_plumbing(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ctc_loss_Tensor_generated_plumbing(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _ctc_loss_generated_plumbing(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _ctc_loss_backward_generated_plumbing(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diag_embed_generated_plumbing(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diagflat_generated_plumbing(const Tensor & self, int64_t offset) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diagonal_generated_plumbing(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_diagonal_generated_plumbing(const Tensor & A, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(A_value, A_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diagonal_Dimname_generated_plumbing(const Tensor & self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diagonal_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diff_generated_plumbing(const Tensor & self, int64_t n, int64_t dim, const c10::optional<Tensor> & prepend, const c10::optional<Tensor> & append) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> gradient_scalarint_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & spacing, c10::optional<int64_t> dim, int64_t edge_order) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> gradient_scalararray_generated_plumbing(const Tensor & self, const Scalar & spacing, IntArrayRef dim, int64_t edge_order) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> gradient_array_generated_plumbing(const Tensor & self, IntArrayRef dim, int64_t edge_order) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, edge_order);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> gradient_scalarrayint_generated_plumbing(const Tensor & self, ArrayRef<Scalar> spacing, c10::optional<int64_t> dim, int64_t edge_order) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> gradient_scalarrayarray_generated_plumbing(const Tensor & self, ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor div_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor div_Tensor_mode_generated_plumbing(const Tensor & self, const Tensor & other, c10::optional<c10::string_view> rounding_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor div_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor div_Scalar_mode_generated_plumbing(const Tensor & self, const Scalar & other, c10::optional<c10::string_view> rounding_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor divide_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor divide_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor divide_Tensor_mode_generated_plumbing(const Tensor & self, const Tensor & other, c10::optional<c10::string_view> rounding_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor divide_Scalar_mode_generated_plumbing(const Tensor & self, const Scalar & other, c10::optional<c10::string_view> rounding_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor true_divide_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor true_divide_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor dot_generated_plumbing(const Tensor & self, const Tensor & tensor) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor tensor_value;
  optional<int64_t> tensor_bdim;
  std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(tensor, cur_level);
  auto results = batch_rule(self_value, self_bdim, tensor_value, tensor_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor vdot_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor embedding_generated_plumbing(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor embedding_backward_generated_plumbing(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor embedding_dense_backward_generated_plumbing(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor embedding_sparse_backward_generated_plumbing(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
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
  auto results = batch_rule(grad_value, grad_bdim, indices_value, indices_bdim, num_weights, padding_idx, scale_grad_by_freq);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag_forward_only_generated_plumbing(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _rowwise_prune_generated_plumbing(const Tensor & weight, const Tensor & mask, ScalarType compressed_indices_dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag_generated_plumbing(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, bool include_last_offset) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag_padding_idx_generated_plumbing(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, bool include_last_offset, c10::optional<int64_t> padding_idx) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag_generated_plumbing(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _embedding_bag_backward_generated_plumbing(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<Tensor> & per_sample_weights, int64_t padding_idx) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _embedding_bag_sparse_backward_generated_plumbing(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor> & per_sample_weights, int64_t padding_idx) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _embedding_bag_dense_backward_generated_plumbing(const Tensor & grad, const Tensor & indices, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor> & per_sample_weights, int64_t padding_idx) {
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
  auto results = batch_rule(grad_value, grad_bdim, indices_value, indices_bdim, offset2bag_value, offset2bag_bdim, bag_size_value, bag_size_bdim, maximum_indices_value, maximum_indices_bdim, num_weights, scale_grad_by_freq, mode, per_sample_weights_value, per_sample_weights_bdim, padding_idx);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _embedding_bag_per_sample_weights_backward_generated_plumbing(const Tensor & grad, const Tensor & weight, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, int64_t mode, int64_t padding_idx) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor new_empty_generated_plumbing(const Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor new_empty_strided_generated_plumbing(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor new_full_generated_plumbing(const Tensor & self, IntArrayRef size, const Scalar & fill_value, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor new_zeros_generated_plumbing(const Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor new_ones_generated_plumbing(const Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _empty_per_channel_affine_quantized_generated_plumbing(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor empty_quantized_generated_plumbing(IntArrayRef size, const Tensor & qtensor, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor empty_like_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor erf_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor erfc_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor exp_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor exp2_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor expm1_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor expand_generated_plumbing(const Tensor & self, IntArrayRef size, bool implicit) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, implicit);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor expand_as_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor flatten_using_ints_generated_plumbing(const Tensor & self, int64_t start_dim, int64_t end_dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor flatten_named_out_dim_generated_plumbing(const Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor flatten_using_names_generated_plumbing(const Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor flatten_DimnameList_generated_plumbing(const Tensor & self, DimnameList dims, Dimname out_dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor unflatten_int_generated_plumbing(const Tensor & self, int64_t dim, IntArrayRef sizes, c10::optional<DimnameList> names) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor unflatten_Dimname_generated_plumbing(const Tensor & self, Dimname dim, IntArrayRef sizes, DimnameList names) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor floor_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor floor_divide_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor floor_divide_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor frac_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor full_like_generated_plumbing(const Tensor & self, const Scalar & fill_value, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gcd_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor lcm_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor grid_sampler_generated_plumbing(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grid_value;
  optional<int64_t> grid_bdim;
  std::tie(grid_value, grid_bdim) = unwrapTensorAtLevel(grid, cur_level);
  auto results = batch_rule(input_value, input_bdim, grid_value, grid_bdim, interpolation_mode, padding_mode, align_corners);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor grid_sampler_2d_generated_plumbing(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grid_value;
  optional<int64_t> grid_bdim;
  std::tie(grid_value, grid_bdim) = unwrapTensorAtLevel(grid, cur_level);
  auto results = batch_rule(input_value, input_bdim, grid_value, grid_bdim, interpolation_mode, padding_mode, align_corners);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> grid_sampler_2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _grid_sampler_2d_cpu_fallback_generated_plumbing(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grid_value;
  optional<int64_t> grid_bdim;
  std::tie(grid_value, grid_bdim) = unwrapTensorAtLevel(grid, cur_level);
  auto results = batch_rule(input_value, input_bdim, grid_value, grid_bdim, interpolation_mode, padding_mode, align_corners);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _grid_sampler_2d_cpu_fallback_backward_generated_plumbing(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor grid_sampler_3d_generated_plumbing(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor grid_value;
  optional<int64_t> grid_bdim;
  std::tie(grid_value, grid_bdim) = unwrapTensorAtLevel(grid, cur_level);
  auto results = batch_rule(input_value, input_bdim, grid_value, grid_bdim, interpolation_mode, padding_mode, align_corners);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> grid_sampler_3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hinge_embedding_loss_generated_plumbing(const Tensor & self, const Tensor & target, double margin, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor group_norm_generated_plumbing(const Tensor & input, int64_t num_groups, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps, bool cudnn_enabled) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> native_group_norm_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> native_group_norm_backward_generated_plumbing(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const c10::optional<Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _fft_r2c_generated_plumbing(const Tensor & self, IntArrayRef dim, int64_t normalization, bool onesided) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _fft_c2r_generated_plumbing(const Tensor & self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _fft_c2c_generated_plumbing(const Tensor & self, IntArrayRef dim, int64_t normalization, bool forward) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, normalization, forward);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_Tensor_generated_plumbing(const Tensor & self, const c10::List<c10::optional<Tensor>> & indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_copy_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_copy_dimname_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_put_generated_plumbing(const Tensor & self, const c10::List<c10::optional<Tensor>> & indices, const Tensor & values, bool accumulate) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor instance_norm_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
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
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, use_input_stats, momentum, eps, cudnn_enabled);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor inverse_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isclose_generated_plumbing(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isin_Tensor_Tensor_generated_plumbing(const Tensor & elements, const Tensor & test_elements, bool assume_unique, bool invert) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isin_Tensor_Scalar_generated_plumbing(const Tensor & elements, const Scalar & test_element, bool assume_unique, bool invert) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isin_Scalar_Tensor_generated_plumbing(const Scalar & element, const Tensor & test_elements, bool assume_unique, bool invert) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isnan_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_distributed_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_floating_point_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_complex_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_conj_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool _is_zerotensor_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_neg_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isreal_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_nonzero_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_same_size_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_signed_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_inference_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor kl_div_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor kl_div_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor kron_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> kthvalue_generated_plumbing(const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> kthvalue_dimname_generated_plumbing(const Tensor & self, int64_t k, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor layer_norm_generated_plumbing(const Tensor & input, IntArrayRef normalized_shape, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps, bool cudnn_enable) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> native_layer_norm_generated_plumbing(const Tensor & input, IntArrayRef normalized_shape, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _native_multi_head_self_attention_generated_plumbing(const Tensor & query, const Tensor & qkv_weight, const Tensor & qkv_bias, const Tensor & proj_weight, const Tensor & proj_bias, int64_t num_head, const c10::optional<Tensor> & mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _transform_bias_rescale_qkv_generated_plumbing(const Tensor & qkv, const Tensor & qkv_bias, int64_t num_head) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor qkv_value;
  optional<int64_t> qkv_bdim;
  std::tie(qkv_value, qkv_bdim) = unwrapTensorAtLevel(qkv, cur_level);
  Tensor qkv_bias_value;
  optional<int64_t> qkv_bias_bdim;
  std::tie(qkv_bias_value, qkv_bias_bdim) = unwrapTensorAtLevel(qkv_bias, cur_level);
  auto results = batch_rule(qkv_value, qkv_bdim, qkv_bias_value, qkv_bias_bdim, num_head);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward_generated_plumbing(const Tensor & grad_out, const Tensor & input, IntArrayRef normalized_shape, const Tensor & mean, const Tensor & rstd, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nan_to_num_generated_plumbing(const Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linear_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_linear_generated_plumbing(const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias) {
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
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_linear_backward_input_generated_plumbing(IntArrayRef input_size, const Tensor & grad_output, const Tensor & weight) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> mkldnn_linear_backward_weights_generated_plumbing(const Tensor & grad_output, const Tensor & input, const Tensor & weight, bool bias_defined) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> mkldnn_linear_backward_generated_plumbing(const Tensor & self, const Tensor & grad_output, const Tensor & weight, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fbgemm_linear_int8_weight_fp32_activation_generated_plumbing(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, const Scalar & weight_scale, const Scalar & weight_zero_point, const Tensor & bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fbgemm_linear_int8_weight_generated_plumbing(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, const Scalar & weight_scale, const Scalar & weight_zero_point, const Tensor & bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,double,int64_t> fbgemm_linear_quantize_weight_generated_plumbing(const Tensor & input) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fbgemm_pack_gemm_matrix_fp16_generated_plumbing(const Tensor & input) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fbgemm_linear_fp16_weight_fp32_activation_generated_plumbing(const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fbgemm_linear_fp16_weight_generated_plumbing(const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fbgemm_pack_quantized_matrix_generated_plumbing(const Tensor & input) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fbgemm_pack_quantized_matrix_KN_generated_plumbing(const Tensor & input, int64_t K, int64_t N) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, K, N);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ldexp_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log10_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log1p_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log2_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logaddexp_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logaddexp2_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor xlogy_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor xlogy_Scalar_Self_generated_plumbing(const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor xlogy_Scalar_Other_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logdet_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log_softmax_int_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log_softmax_Dimname_generated_plumbing(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _log_softmax_generated_plumbing(const Tensor & self, int64_t dim, bool half_to_float) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, half_to_float);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _log_softmax_backward_data_generated_plumbing(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType input_dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _logcumsumexp_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logcumsumexp_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logcumsumexp_dimname_generated_plumbing(const Tensor & self, Dimname dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logsumexp_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logsumexp_names_generated_plumbing(const Tensor & self, DimnameList dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor margin_ranking_loss_generated_plumbing(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor matmul_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor matrix_rank_tol_generated_plumbing(const Tensor & self, double tol, bool symmetric) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, tol, symmetric);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor matrix_rank_generated_plumbing(const Tensor & self, bool symmetric) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, symmetric);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor matrix_power_generated_plumbing(const Tensor & self, int64_t n) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, n);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor matrix_exp_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor matrix_exp_backward_generated_plumbing(const Tensor & self, const Tensor & grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(self_value, self_bdim, grad_value, grad_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _aminmax_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _aminmax_dim_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> aminmax_generated_plumbing(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _compute_linear_combination_generated_plumbing(const Tensor & input, const Tensor & coefficients) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor coefficients_value;
  optional<int64_t> coefficients_bdim;
  std::tie(coefficients_value, coefficients_bdim) = unwrapTensorAtLevel(coefficients, cur_level);
  auto results = batch_rule(input_value, input_bdim, coefficients_value, coefficients_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> max_dim_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> max_names_dim_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor value_selecting_reduction_backward_generated_plumbing(const Tensor & grad, int64_t dim, const Tensor & indices, IntArrayRef sizes, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor amax_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> max_pool1d_with_indices_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_pool1d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_pool2d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_max_pool2d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_max_pool2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & output, const Tensor & input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_max_pool3d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_max_pool3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & output, const Tensor & input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantized_max_pool1d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantized_max_pool2d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_pool3d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mean_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mean_dim_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mean_names_dim_generated_plumbing(const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nanmean_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor median_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> median_dim_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> median_names_dim_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nanmedian_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> nanmedian_dim_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> nanmedian_names_dim_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> min_dim_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> min_names_dim_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor amin_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_convolution_generated_plumbing(const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
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
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim, padding, stride, dilation, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm_backward_generated_plumbing(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_var, double epsilon) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor miopen_convolution_generated_plumbing(const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor miopen_convolution_transpose_generated_plumbing(const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor miopen_depthwise_convolution_generated_plumbing(const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mm_generated_plumbing(const Tensor & self, const Tensor & mat2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mat2_value;
  optional<int64_t> mat2_bdim;
  std::tie(mat2_value, mat2_bdim) = unwrapTensorAtLevel(mat2, cur_level);
  auto results = batch_rule(self_value, self_bdim, mat2_value, mat2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_mm_generated_plumbing(const Tensor & sparse, const Tensor & dense) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor sparse_value;
  optional<int64_t> sparse_bdim;
  std::tie(sparse_value, sparse_bdim) = unwrapTensorAtLevel(sparse, cur_level);
  Tensor dense_value;
  optional<int64_t> dense_bdim;
  std::tie(dense_value, dense_bdim) = unwrapTensorAtLevel(dense, cur_level);
  auto results = batch_rule(sparse_value, sparse_bdim, dense_value, dense_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_sparse_matmul_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_mask_helper_generated_plumbing(const Tensor & t, const Tensor & mask_indices) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor t_value;
  optional<int64_t> t_bdim;
  std::tie(t_value, t_bdim) = unwrapTensorAtLevel(t, cur_level);
  Tensor mask_indices_value;
  optional<int64_t> mask_indices_bdim;
  std::tie(mask_indices_value, mask_indices_bdim) = unwrapTensorAtLevel(mask_indices, cur_level);
  auto results = batch_rule(t_value, t_bdim, mask_indices_value, mask_indices_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> mode_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> mode_dimname_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mul_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mul_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor multiply_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor multiply_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mv_generated_plumbing(const Tensor & self, const Tensor & vec) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor vec_value;
  optional<int64_t> vec_bdim;
  std::tie(vec_value, vec_bdim) = unwrapTensorAtLevel(vec, cur_level);
  auto results = batch_rule(self_value, self_bdim, vec_value, vec_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mvlgamma_generated_plumbing(const Tensor & self, int64_t p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor narrow_copy_generated_plumbing(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, length);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor narrow_generated_plumbing(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, length);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor narrow_Tensor_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & start, int64_t length) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> batch_norm_stats_generated_plumbing(const Tensor & input, double eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor batch_norm_elemt_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const Tensor & mean, const Tensor & invstd, double eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> batch_norm_gather_stats_generated_plumbing(const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, double momentum, double eps, int64_t count) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts_generated_plumbing(const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, double momentum, double eps, const Tensor & counts) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward_generated_plumbing(const Tensor & grad_out, const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, const c10::optional<Tensor> & save_mean, const c10::optional<Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor> batch_norm_backward_reduce_generated_plumbing(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor batch_norm_backward_elemt_generated_plumbing(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const c10::optional<Tensor> & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu, const Tensor & count) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> batch_norm_update_stats_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, double momentum) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _nnpack_spatial_convolution_generated_plumbing(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ones_like_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pairwise_distance_generated_plumbing(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) {
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
  auto results = batch_rule(x1_value, x1_bdim, x2_value, x2_bdim, p, eps, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cdist_generated_plumbing(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _euclidean_dist_generated_plumbing(const Tensor & x1, const Tensor & x2) {
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
  auto results = batch_rule(x1_value, x1_bdim, x2_value, x2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cdist_forward_generated_plumbing(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cdist_backward_generated_plumbing(const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pdist_generated_plumbing(const Tensor & self, double p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _pdist_forward_generated_plumbing(const Tensor & self, double p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _pdist_backward_generated_plumbing(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cosine_similarity_generated_plumbing(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor permute_generated_plumbing(const Tensor & self, IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor movedim_intlist_generated_plumbing(const Tensor & self, IntArrayRef source, IntArrayRef destination) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor movedim_int_generated_plumbing(const Tensor & self, int64_t source, int64_t destination) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor moveaxis_intlist_generated_plumbing(const Tensor & self, IntArrayRef source, IntArrayRef destination) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor moveaxis_int_generated_plumbing(const Tensor & self, int64_t source, int64_t destination) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor numpy_T_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor matrix_H_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mT_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mH_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor adjoint_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pixel_shuffle_generated_plumbing(const Tensor & self, int64_t upscale_factor) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upscale_factor);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pixel_unshuffle_generated_plumbing(const Tensor & self, int64_t downscale_factor) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, downscale_factor);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor channel_shuffle_generated_plumbing(const Tensor & self, int64_t groups) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor native_channel_shuffle_generated_plumbing(const Tensor & self, int64_t groups) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_pinned_generated_plumbing(const Tensor & self, c10::optional<Device> device) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pin_memory_generated_plumbing(const Tensor & self, c10::optional<Device> device) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _pin_memory_generated_plumbing(const Tensor & self, c10::optional<Device> device) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pinverse_generated_plumbing(const Tensor & self, double rcond) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, rcond);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor poisson_nll_loss_generated_plumbing(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rad2deg_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor deg2rad_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rand_like_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor randint_like_generated_plumbing(const Tensor & self, int64_t high, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor randint_like_low_dtype_generated_plumbing(const Tensor & self, int64_t low, int64_t high, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor randn_like_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ravel_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reciprocal_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor neg_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor negative_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor repeat_generated_plumbing(const Tensor & self, IntArrayRef repeats) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, repeats);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor repeat_interleave_Tensor_generated_plumbing(const Tensor & repeats, c10::optional<int64_t> output_size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor repeats_value;
  optional<int64_t> repeats_bdim;
  std::tie(repeats_value, repeats_bdim) = unwrapTensorAtLevel(repeats, cur_level);
  auto results = batch_rule(repeats_value, repeats_bdim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor repeat_interleave_self_Tensor_generated_plumbing(const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor repeat_interleave_self_int_generated_plumbing(const Tensor & self, int64_t repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reshape_generated_plumbing(const Tensor & self, IntArrayRef shape) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, shape);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _reshape_alias_generated_plumbing(const Tensor & self, IntArrayRef size, IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _mkldnn_reshape_generated_plumbing(const Tensor & self, IntArrayRef shape) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, shape);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reshape_as_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor round_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor round_decimals_generated_plumbing(const Tensor & self, int64_t decimals) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, decimals);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rrelu_generated_plumbing(const Tensor & self, const Scalar & lower, const Scalar & upper, bool training, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor relu_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor relu6_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor prelu_generated_plumbing(const Tensor & self, const Tensor & weight) {
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
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> prelu_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & weight) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, weight_value, weight_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gelu_generated_plumbing(const Tensor & self, c10::string_view approximate) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gelu_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, c10::string_view approximate) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor infinitely_differentiable_gelu_backward_generated_plumbing(const Tensor & grad, const Tensor & self) {
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
  auto results = batch_rule(grad_value, grad_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardshrink_generated_plumbing(const Tensor & self, const Scalar & lambd) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, lambd);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardshrink_backward_generated_plumbing(const Tensor & grad_out, const Tensor & self, const Scalar & lambd) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_out_value;
  optional<int64_t> grad_out_bdim;
  std::tie(grad_out_value, grad_out_bdim) = unwrapTensorAtLevel(grad_out, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_out_value, grad_out_bdim, self_value, self_bdim, lambd);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rsqrt_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor select_Dimname_generated_plumbing(const Tensor & self, Dimname dim, int64_t index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor select_int_generated_plumbing(const Tensor & self, int64_t dim, int64_t index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor select_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef input_sizes, int64_t dim, int64_t index) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor selu_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor celu_generated_plumbing(const Tensor & self, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor silu_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor silu_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mish_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mish_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sigmoid_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logit_generated_plumbing(const Tensor & self, c10::optional<double> eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sin_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sinc_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sinh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor detach_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t size_int_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t size_Dimname_generated_plumbing(const Tensor & self, Dimname dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slice_Tensor_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slice_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slice_scatter_generated_plumbing(const Tensor & self, const Tensor & src, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor select_scatter_generated_plumbing(const Tensor & self, const Tensor & src, int64_t dim, int64_t index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diagonal_scatter_generated_plumbing(const Tensor & self, const Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> slogdet_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor smm_generated_plumbing(const Tensor & self, const Tensor & mat2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mat2_value;
  optional<int64_t> mat2_bdim;
  std::tie(mat2_value, mat2_bdim) = unwrapTensorAtLevel(mat2, cur_level);
  auto results = batch_rule(self_value, self_bdim, mat2_value, mat2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor softmax_int_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor softmax_Dimname_generated_plumbing(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _softmax_generated_plumbing(const Tensor & self, int64_t dim, bool half_to_float) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, half_to_float);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _softmax_backward_data_generated_plumbing(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType input_dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> unsafe_split_Tensor_generated_plumbing(const Tensor & self, int64_t split_size, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_size, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> split_Tensor_generated_plumbing(const Tensor & self, int64_t split_size, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_size, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> unsafe_split_with_sizes_generated_plumbing(const Tensor & self, IntArrayRef split_sizes, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_sizes, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> split_with_sizes_generated_plumbing(const Tensor & self, IntArrayRef split_sizes, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, split_sizes, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> hsplit_int_generated_plumbing(const Tensor & self, int64_t sections) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> hsplit_array_generated_plumbing(const Tensor & self, IntArrayRef indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> vsplit_int_generated_plumbing(const Tensor & self, int64_t sections) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> vsplit_array_generated_plumbing(const Tensor & self, IntArrayRef indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> dsplit_int_generated_plumbing(const Tensor & self, int64_t sections) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> dsplit_array_generated_plumbing(const Tensor & self, IntArrayRef indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor squeeze_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor squeeze_dim_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor squeeze_dimname_generated_plumbing(const Tensor & self, Dimname dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sspaddmm_generated_plumbing(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mat1_value;
  optional<int64_t> mat1_bdim;
  std::tie(mat1_value, mat1_bdim) = unwrapTensorAtLevel(mat1, cur_level);
  Tensor mat2_value;
  optional<int64_t> mat2_bdim;
  std::tie(mat2_value, mat2_bdim) = unwrapTensorAtLevel(mat2, cur_level);
  auto results = batch_rule(self_value, self_bdim, mat1_value, mat1_bdim, mat2_value, mat2_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor stft_generated_plumbing(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor istft_generated_plumbing(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<Tensor> & window, bool center, bool normalized, c10::optional<bool> onesided, c10::optional<int64_t> length, bool return_complex) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t stride_int_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t stride_Dimname_generated_plumbing(const Tensor & self, Dimname dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sum_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sum_dim_IntList_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sum_dim_DimnameList_generated_plumbing(const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nansum_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nansum_dim_IntList_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sum_to_size_generated_plumbing(const Tensor & self, IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sqrt_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor square_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor std_generated_plumbing(const Tensor & self, bool unbiased) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, unbiased);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor std_dim_generated_plumbing(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor std_correction_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> std_mean_generated_plumbing(const Tensor & self, bool unbiased) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> std_mean_dim_generated_plumbing(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> std_mean_correction_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> std_mean_names_dim_generated_plumbing(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> std_mean_correction_names_generated_plumbing(const Tensor & self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor std_names_dim_generated_plumbing(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor std_correction_names_generated_plumbing(const Tensor & self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor prod_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor prod_dim_int_generated_plumbing(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor prod_dim_Dimname_generated_plumbing(const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor t_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor tan_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor tanh_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor tensordot_generated_plumbing(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor threshold_generated_plumbing(const Tensor & self, const Scalar & threshold, const Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, threshold, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor threshold_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Scalar & threshold) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, threshold);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor tile_generated_plumbing(const Tensor & self, IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor transpose_int_generated_plumbing(const Tensor & self, int64_t dim0, int64_t dim1) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor transpose_Dimname_generated_plumbing(const Tensor & self, Dimname dim0, Dimname dim1) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _mkldnn_transpose_generated_plumbing(const Tensor & self, int64_t dim0, int64_t dim1) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor one_hot_generated_plumbing(const Tensor & self, int64_t num_classes) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, num_classes);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor flip_generated_plumbing(const Tensor & self, IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fliplr_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor flipud_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor roll_generated_plumbing(const Tensor & self, IntArrayRef shifts, IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, shifts, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rot90_generated_plumbing(const Tensor & self, int64_t k, IntArrayRef dims) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor trapezoid_x_generated_plumbing(const Tensor & y, const Tensor & x, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor y_value;
  optional<int64_t> y_bdim;
  std::tie(y_value, y_bdim) = unwrapTensorAtLevel(y, cur_level);
  Tensor x_value;
  optional<int64_t> x_bdim;
  std::tie(x_value, x_bdim) = unwrapTensorAtLevel(x, cur_level);
  auto results = batch_rule(y_value, y_bdim, x_value, x_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor trapezoid_dx_generated_plumbing(const Tensor & y, const Scalar & dx, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor trapz_x_generated_plumbing(const Tensor & y, const Tensor & x, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor y_value;
  optional<int64_t> y_bdim;
  std::tie(y_value, y_bdim) = unwrapTensorAtLevel(y, cur_level);
  Tensor x_value;
  optional<int64_t> x_bdim;
  std::tie(x_value, x_bdim) = unwrapTensorAtLevel(x, cur_level);
  auto results = batch_rule(y_value, y_bdim, x_value, x_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor trapz_dx_generated_plumbing(const Tensor & y, double dx, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _trilinear_generated_plumbing(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor triplet_margin_loss_generated_plumbing(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor trunc_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fix_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor type_as_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool _has_compatible_shallow_copy_type_generated_plumbing(const Tensor & self, const Tensor & from) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor from_value;
  optional<int64_t> from_bdim;
  std::tie(from_value, from_bdim) = unwrapTensorAtLevel(from, cur_level);
  auto results = batch_rule(self_value, self_bdim, from_value, from_bdim);
  return std::get<0>(results);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _unique_generated_plumbing(const Tensor & self, bool sorted, bool return_inverse) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> unique_dim_generated_plumbing(const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> unique_consecutive_generated_plumbing(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive_generated_plumbing(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _unique2_generated_plumbing(const Tensor & self, bool sorted, bool return_inverse, bool return_counts) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _unsafe_view_generated_plumbing(const Tensor & self, IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor unsqueeze_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor vander_generated_plumbing(const Tensor & x, c10::optional<int64_t> N, bool increasing) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor x_value;
  optional<int64_t> x_bdim;
  std::tie(x_value, x_bdim) = unwrapTensorAtLevel(x, cur_level);
  auto results = batch_rule(x_value, x_bdim, N, increasing);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor var_generated_plumbing(const Tensor & self, bool unbiased) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, unbiased);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor var_dim_generated_plumbing(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor var_correction_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor var_names_dim_generated_plumbing(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor var_correction_names_generated_plumbing(const Tensor & self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> var_mean_generated_plumbing(const Tensor & self, bool unbiased) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> var_mean_dim_generated_plumbing(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> var_mean_correction_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> var_mean_names_dim_generated_plumbing(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> var_mean_correction_names_generated_plumbing(const Tensor & self, DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor view_as_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor where_self_generated_plumbing(const Tensor & condition, const Tensor & self, const Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor condition_value;
  optional<int64_t> condition_bdim;
  std::tie(condition_value, condition_bdim) = unwrapTensorAtLevel(condition, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor where_ScalarSelf_generated_plumbing(const Tensor & condition, const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor where_ScalarOther_generated_plumbing(const Tensor & condition, const Tensor & self, const Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor condition_value;
  optional<int64_t> condition_bdim;
  std::tie(condition_value, condition_bdim) = unwrapTensorAtLevel(condition, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor where_Scalar_generated_plumbing(const Tensor & condition, const Scalar & self, const Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor condition_value;
  optional<int64_t> condition_bdim;
  std::tie(condition_value, condition_bdim) = unwrapTensorAtLevel(condition, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> where_generated_plumbing(const Tensor & condition) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _s_where_generated_plumbing(const Tensor & condition, const Tensor & self, const Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor condition_value;
  optional<int64_t> condition_bdim;
  std::tie(condition_value, condition_bdim) = unwrapTensorAtLevel(condition, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(condition_value, condition_bdim, self_value, self_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor norm_except_dim_generated_plumbing(const Tensor & v, int64_t pow, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor v_value;
  optional<int64_t> v_bdim;
  std::tie(v_value, v_bdim) = unwrapTensorAtLevel(v, cur_level);
  auto results = batch_rule(v_value, v_bdim, pow, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _weight_norm_generated_plumbing(const Tensor & v, const Tensor & g, int64_t dim) {
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
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_generated_plumbing(const Tensor & v, const Tensor & g, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_backward_generated_plumbing(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _weight_norm_differentiable_backward_generated_plumbing(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor zeros_like_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _standard_gamma_grad_generated_plumbing(const Tensor & self, const Tensor & output) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor output_value;
  optional<int64_t> output_bdim;
  std::tie(output_value, output_bdim) = unwrapTensorAtLevel(output, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _standard_gamma_generated_plumbing(const Tensor & self, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _dirichlet_grad_generated_plumbing(const Tensor & x, const Tensor & alpha, const Tensor & total) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor x_value;
  optional<int64_t> x_bdim;
  std::tie(x_value, x_bdim) = unwrapTensorAtLevel(x, cur_level);
  Tensor alpha_value;
  optional<int64_t> alpha_bdim;
  std::tie(alpha_value, alpha_bdim) = unwrapTensorAtLevel(alpha, cur_level);
  Tensor total_value;
  optional<int64_t> total_bdim;
  std::tie(total_value, total_bdim) = unwrapTensorAtLevel(total, cur_level);
  auto results = batch_rule(x_value, x_bdim, alpha_value, alpha_bdim, total_value, total_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sample_dirichlet_generated_plumbing(const Tensor & self, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor poisson_generated_plumbing(const Tensor & self, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor binomial_generated_plumbing(const Tensor & count, const Tensor & prob, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor native_norm_generated_plumbing(const Tensor & self, const Scalar & p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor native_norm_ScalarOpt_dim_dtype_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_sum_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_sum_dtype_generated_plumbing(const Tensor & self, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_sum_dim_generated_plumbing(const Tensor & self, IntArrayRef dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_sum_dim_dtype_generated_plumbing(const Tensor & self, IntArrayRef dim, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_sum_backward_generated_plumbing(const Tensor & grad, const Tensor & self, IntArrayRef dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_softmax_int_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_softmax_Dimname_generated_plumbing(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_softmax_generated_plumbing(const Tensor & self, int64_t dim, bool half_to_float) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, half_to_float);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_softmax_backward_data_generated_plumbing(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
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
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim, dim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_log_softmax_int_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_log_softmax_Dimname_generated_plumbing(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_log_softmax_generated_plumbing(const Tensor & self, int64_t dim, bool half_to_float) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, half_to_float);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_log_softmax_backward_data_generated_plumbing(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
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
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim, dim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor norm_ScalarOpt_dtype_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & p, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor norm_Scalar_generated_plumbing(const Tensor & self, const Scalar & p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor norm_ScalarOpt_dim_dtype_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor norm_ScalarOpt_dim_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & p, IntArrayRef dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor norm_names_ScalarOpt_dim_dtype_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & p, DimnameList dim, bool keepdim, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor norm_names_ScalarOpt_dim_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & p, DimnameList dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> frexp_Tensor_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor frobenius_norm_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor frobenius_norm_dim_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nuclear_norm_generated_plumbing(const Tensor & self, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nuclear_norm_dim_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor clone_generated_plumbing(const Tensor & self, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor positive_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sub_Tensor_generated_plumbing(const Tensor & self, const Tensor & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sub_Scalar_generated_plumbing(const Tensor & self, const Scalar & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor subtract_Tensor_generated_plumbing(const Tensor & self, const Tensor & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor subtract_Scalar_generated_plumbing(const Tensor & self, const Scalar & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rsub_Tensor_generated_plumbing(const Tensor & self, const Tensor & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor heaviside_generated_plumbing(const Tensor & self, const Tensor & values) {
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
  auto results = batch_rule(self_value, self_bdim, values_value, values_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rsub_Scalar_generated_plumbing(const Tensor & self, const Scalar & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_addmm_generated_plumbing(const Tensor & self, const Tensor & sparse, const Tensor & dense, const Scalar & beta, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor sparse_value;
  optional<int64_t> sparse_bdim;
  std::tie(sparse_value, sparse_bdim) = unwrapTensorAtLevel(sparse, cur_level);
  Tensor dense_value;
  optional<int64_t> dense_bdim;
  std::tie(dense_value, dense_bdim) = unwrapTensorAtLevel(dense, cur_level);
  auto results = batch_rule(self_value, self_bdim, sparse_value, sparse_bdim, dense_value, dense_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sparse_sampled_addmm_generated_plumbing(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mat1_value;
  optional<int64_t> mat1_bdim;
  std::tie(mat1_value, mat1_bdim) = unwrapTensorAtLevel(mat1, cur_level);
  Tensor mat2_value;
  optional<int64_t> mat2_bdim;
  std::tie(mat2_value, mat2_bdim) = unwrapTensorAtLevel(mat2, cur_level);
  auto results = batch_rule(self_value, self_bdim, mat1_value, mat1_bdim, mat2_value, mat2_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor addmm_generated_plumbing(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mat1_value;
  optional<int64_t> mat1_bdim;
  std::tie(mat1_value, mat1_bdim) = unwrapTensorAtLevel(mat1, cur_level);
  Tensor mat2_value;
  optional<int64_t> mat2_bdim;
  std::tie(mat2_value, mat2_bdim) = unwrapTensorAtLevel(mat2, cur_level);
  auto results = batch_rule(self_value, self_bdim, mat1_value, mat1_bdim, mat2_value, mat2_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sparse_csr_tensor_crow_col_value_size_generated_plumbing(const Tensor & crow_indices, const Tensor & col_indices, const Tensor & values, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sparse_csr_tensor_crow_col_value_generated_plumbing(const Tensor & crow_indices, const Tensor & col_indices, const Tensor & values, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_csr_tensor_unsafe_generated_plumbing(const Tensor & crow_indices, const Tensor & col_indices, const Tensor & values, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sparse_coo_tensor_indices_generated_plumbing(const Tensor & indices, const Tensor & values, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sparse_coo_tensor_indices_size_generated_plumbing(const Tensor & indices, const Tensor & values, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_coo_tensor_unsafe_generated_plumbing(const Tensor & indices, const Tensor & values, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _sparse_coo_tensor_with_dims_and_tensors_generated_plumbing(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sparse_mask_generated_plumbing(const Tensor & self, const Tensor & mask) {
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
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_dense_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_dense_backward_generated_plumbing(const Tensor & grad, const Tensor & input) {
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
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t sparse_dim_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t _dimI_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t dense_dim_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t _dimV_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t _nnz_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor coalesce_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _coalesce_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_coalesced_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _indices_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _values_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor indices_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor values_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor crow_indices_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor col_indices_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hspmm_generated_plumbing(const Tensor & mat1, const Tensor & mat2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor mat1_value;
  optional<int64_t> mat1_bdim;
  std::tie(mat1_value, mat1_bdim) = unwrapTensorAtLevel(mat1, cur_level);
  Tensor mat2_value;
  optional<int64_t> mat2_bdim;
  std::tie(mat2_value, mat2_bdim) = unwrapTensorAtLevel(mat2, cur_level);
  auto results = batch_rule(mat1_value, mat1_bdim, mat2_value, mat2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> unbind_int_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> unbind_Dimname_generated_plumbing(const Tensor & self, Dimname dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_sparse_sparse_dim_generated_plumbing(const Tensor & self, int64_t sparse_dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, sparse_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_sparse_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_mkldnn_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_reorder_conv2d_weight_generated_plumbing(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_reorder_conv3d_weight_generated_plumbing(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_mkldnn_backward_generated_plumbing(const Tensor & grad, const Tensor & input) {
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
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantize_per_tensor_dynamic_generated_plumbing(const Tensor & self, ScalarType dtype, bool reduce_range) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantize_per_tensor_generated_plumbing(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantize_per_tensor_tensor_qparams_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantize_per_channel_generated_plumbing(const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor dequantize_self_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
double q_scale_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t q_zero_point_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor q_per_channel_scales_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor q_per_channel_zero_points_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
int64_t q_per_channel_axis_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor int_repr_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _make_per_tensor_quantized_tensor_generated_plumbing(const Tensor & self, double scale, int64_t zero_point) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, scale, zero_point);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _make_per_channel_quantized_tensor_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis) {
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
  auto results = batch_rule(self_value, self_bdim, scale_value, scale_bdim, zero_point_value, zero_point_bdim, axis);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
QScheme qscheme_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fake_quantize_per_tensor_affine_generated_plumbing(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fake_quantize_per_tensor_affine_tensor_qparams_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> fake_quantize_per_tensor_affine_cachemask_generated_plumbing(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, const Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fake_quantize_per_tensor_affine_cachemask_backward_generated_plumbing(const Tensor & grad, const Tensor & mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor mask_value;
  optional<int64_t> mask_bdim;
  std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _fake_quantize_learnable_per_tensor_affine_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _fake_quantize_learnable_per_tensor_affine_backward_generated_plumbing(const Tensor & grad, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fake_quantize_per_channel_affine_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> fake_quantize_per_channel_affine_cachemask_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fake_quantize_per_channel_affine_cachemask_backward_generated_plumbing(const Tensor & grad, const Tensor & mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  Tensor mask_value;
  optional<int64_t> mask_bdim;
  std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _fake_quantize_learnable_per_channel_affine_generated_plumbing(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _fake_quantize_learnable_per_channel_affine_backward_generated_plumbing(const Tensor & grad, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<double,int64_t> _choose_qparams_per_tensor_generated_plumbing(const Tensor & self, bool reduce_range) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _saturate_weight_to_fp16_generated_plumbing(const Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(weight_value, weight_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> choose_qparams_optimized_generated_plumbing(const Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _autocast_to_reduced_precision_generated_plumbing(const Tensor & self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _autocast_to_full_precision_generated_plumbing(const Tensor & self, bool cuda_enabled, bool cpu_enabled) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _to_copy_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_dtype_layout_generated_plumbing(const Tensor & self, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_device_generated_plumbing(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_dtype_generated_plumbing(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor to_other_generated_plumbing(const Tensor & self, const Tensor & other, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor combinations_generated_plumbing(const Tensor & self, int64_t r, bool with_replacement) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, r, with_replacement);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Scalar item_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
ScalarType result_type_Tensor_generated_plumbing(const Tensor & tensor, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
ScalarType result_type_Scalar_generated_plumbing(const Tensor & tensor, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
ScalarType result_type_Scalar_Tensor_generated_plumbing(const Scalar & scalar, const Tensor & tensor) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Scalar _local_scalar_dense_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell_generated_plumbing(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_lstm_cell_backward_generated_plumbing(const c10::optional<Tensor> & grad_hy, const c10::optional<Tensor> & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_lstm_cell_backward_generated_plumbing(const c10::optional<Tensor> & grad_hy, const c10::optional<Tensor> & grad_cy, const Tensor & input_gates, const Tensor & hidden_gates, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias, const Tensor & cx, const Tensor & cy) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _thnn_fused_gru_cell_generated_plumbing(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_gru_cell_backward_generated_plumbing(const Tensor & grad_hy, const Tensor & workspace, bool has_bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_gru_cell_backward_generated_plumbing(const Tensor & grad_hy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const c10::optional<Tensor> & input_bias, const c10::optional<Tensor> & hidden_bias) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gru_cell_generated_plumbing(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const c10::optional<Tensor> & b_ih, const c10::optional<Tensor> & b_hh) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rnn_tanh_cell_generated_plumbing(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const c10::optional<Tensor> & b_ih, const c10::optional<Tensor> & b_hh) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rnn_relu_cell_generated_plumbing(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const c10::optional<Tensor> & b_ih, const c10::optional<Tensor> & b_hh) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantized_gru_cell_generated_plumbing(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, const Scalar & scale_ih, const Scalar & scale_hh, const Scalar & zero_point_ih, const Scalar & zero_point_hh) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantized_rnn_relu_cell_generated_plumbing(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, const Scalar & scale_ih, const Scalar & scale_hh, const Scalar & zero_point_ih, const Scalar & zero_point_hh) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantized_rnn_tanh_cell_generated_plumbing(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, const Scalar & scale_ih, const Scalar & scale_hh, const Scalar & zero_point_ih, const Scalar & zero_point_hh) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _pack_padded_sequence_generated_plumbing(const Tensor & input, const Tensor & lengths, bool batch_first) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _pack_padded_sequence_backward_generated_plumbing(const Tensor & grad, IntArrayRef input_size, const Tensor & batch_sizes, bool batch_first) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _pad_packed_sequence_generated_plumbing(const Tensor & data, const Tensor & batch_sizes, bool batch_first, const Scalar & padding_value, int64_t total_length) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
bool is_set_to_generated_plumbing(const Tensor & self, const Tensor & tensor) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor tensor_value;
  optional<int64_t> tensor_bdim;
  std::tie(tensor_value, tensor_bdim) = unwrapTensorAtLevel(tensor, cur_level);
  auto results = batch_rule(self_value, self_bdim, tensor_value, tensor_bdim);
  return std::get<0>(results);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor masked_fill_Scalar_generated_plumbing(const Tensor & self, const Tensor & mask, const Scalar & value) {
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
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor masked_fill_Tensor_generated_plumbing(const Tensor & self, const Tensor & mask, const Tensor & value) {
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
  Tensor value_value;
  optional<int64_t> value_bdim;
  std::tie(value_value, value_bdim) = unwrapTensorAtLevel(value, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, value_value, value_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor masked_scatter_generated_plumbing(const Tensor & self, const Tensor & mask, const Tensor & source) {
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
  Tensor source_value;
  optional<int64_t> source_bdim;
  std::tie(source_value, source_bdim) = unwrapTensorAtLevel(source, cur_level);
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim, source_value, source_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _masked_softmax_generated_plumbing(const Tensor & self, const Tensor & mask) {
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
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor view_generated_plumbing(const Tensor & self, IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor view_dtype_generated_plumbing(const Tensor & self, ScalarType dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor put_generated_plumbing(const Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_add_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_add_dimname_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_fill_int_Scalar_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_fill_int_Tensor_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
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
  Tensor value_value;
  optional<int64_t> value_bdim;
  std::tie(value_value, value_bdim) = unwrapTensorAtLevel(value, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value_value, value_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_fill_Dimname_Scalar_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, const Scalar & value) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_fill_Dimname_Tensor_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {
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
  Tensor value_value;
  optional<int64_t> value_bdim;
  std::tie(value_value, value_bdim) = unwrapTensorAtLevel(value, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, value_value, value_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_src_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
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
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_value_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_reduce_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, c10::string_view reduce) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_value_reduce_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Scalar & value, c10::string_view reduce) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_dimname_src_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {
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
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_dimname_value_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, const Scalar & value) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_add_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
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
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor scatter_add_dimname_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {
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
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _scatter_reduce_two_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, c10::string_view reduce, c10::optional<int64_t> output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_and_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_and_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __and___Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __and___Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_or_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_or_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __or___Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __or___Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_xor_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_xor_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __xor___Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __xor___Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __lshift___Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __lshift___Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_left_shift_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_left_shift_Tensor_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_left_shift_Scalar_Tensor_generated_plumbing(const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __rshift___Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor __rshift___Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_right_shift_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_right_shift_Tensor_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bitwise_right_shift_Scalar_Tensor_generated_plumbing(const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor addbmm_generated_plumbing(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor batch1_value;
  optional<int64_t> batch1_bdim;
  std::tie(batch1_value, batch1_bdim) = unwrapTensorAtLevel(batch1, cur_level);
  Tensor batch2_value;
  optional<int64_t> batch2_bdim;
  std::tie(batch2_value, batch2_bdim) = unwrapTensorAtLevel(batch2, cur_level);
  auto results = batch_rule(self_value, self_bdim, batch1_value, batch1_bdim, batch2_value, batch2_bdim, beta, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diag_generated_plumbing(const Tensor & self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor diag_backward_generated_plumbing(const Tensor & grad, IntArrayRef input_sizes, int64_t diagonal) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cross_generated_plumbing(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor triu_generated_plumbing(const Tensor & self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor tril_generated_plumbing(const Tensor & self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor trace_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor trace_backward_generated_plumbing(const Tensor & grad, IntArrayRef sizes) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, sizes);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ne_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ne_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor not_equal_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor not_equal_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor eq_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor eq_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ge_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ge_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor greater_equal_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor greater_equal_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor le_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor le_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor less_equal_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor less_equal_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gt_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gt_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor greater_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor greater_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor lt_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor lt_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor less_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor less_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor take_generated_plumbing(const Tensor & self, const Tensor & index) {
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
  auto results = batch_rule(self_value, self_bdim, index_value, index_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor take_along_dim_generated_plumbing(const Tensor & self, const Tensor & indices, c10::optional<int64_t> dim) {
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
  auto results = batch_rule(self_value, self_bdim, indices_value, indices_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_select_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_select_dimname_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor index_select_backward_generated_plumbing(const Tensor & grad, IntArrayRef self_sizes, int64_t dim, const Tensor & index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor masked_select_generated_plumbing(const Tensor & self, const Tensor & mask) {
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
  auto results = batch_rule(self_value, self_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor masked_select_backward_generated_plumbing(const Tensor & grad, const Tensor & input, const Tensor & mask) {
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
  Tensor mask_value;
  optional<int64_t> mask_bdim;
  std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_value, input_bdim, mask_value, mask_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nonzero_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> nonzero_numpy_generated_plumbing(const Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor argwhere_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gather_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gather_backward_generated_plumbing(const Tensor & grad, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor gather_dimname_generated_plumbing(const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _gather_sparse_backward_generated_plumbing(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad) {
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
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, grad_value, grad_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor addcmul_generated_plumbing(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor addcdiv_generated_plumbing(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cross_entropy_loss_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, double label_smoothing) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> lstsq_generated_plumbing(const Tensor & self, const Tensor & A) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> triangular_solve_generated_plumbing(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_solve_triangular_generated_plumbing(const Tensor & self, const Tensor & B, bool upper, bool left, bool unitriangular) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> symeig_generated_plumbing(const Tensor & self, bool eigenvectors, bool upper) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, eigenvectors, upper);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _symeig_helper_generated_plumbing(const Tensor & self, bool eigenvectors, bool upper) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, eigenvectors, upper);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> eig_generated_plumbing(const Tensor & self, bool eigenvectors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, eigenvectors);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> svd_generated_plumbing(const Tensor & self, bool some, bool compute_uv) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor swapaxes_generated_plumbing(const Tensor & self, int64_t axis0, int64_t axis1) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, axis0, axis1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor swapdims_generated_plumbing(const Tensor & self, int64_t dim0, int64_t dim1) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cholesky_generated_plumbing(const Tensor & self, bool upper) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upper);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cholesky_solve_generated_plumbing(const Tensor & self, const Tensor & input2, bool upper) {
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
  auto results = batch_rule(self_value, self_bdim, input2_value, input2_bdim, upper);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _cholesky_solve_helper_generated_plumbing(const Tensor & self, const Tensor & A, bool upper) {
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
  auto results = batch_rule(self_value, self_bdim, A_value, A_bdim, upper);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> solve_generated_plumbing(const Tensor & self, const Tensor & A) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _solve_helper_generated_plumbing(const Tensor & self, const Tensor & A) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor cholesky_inverse_generated_plumbing(const Tensor & self, bool upper) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upper);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> qr_generated_plumbing(const Tensor & self, bool some) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, some);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> geqrf_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor orgqr_generated_plumbing(const Tensor & self, const Tensor & input2) {
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
  auto results = batch_rule(self_value, self_bdim, input2_value, input2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ormqr_generated_plumbing(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _lu_with_info_generated_plumbing(const Tensor & self, bool pivot, bool check_errors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, pivot, check_errors);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor lu_solve_generated_plumbing(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor LU_data_value;
  optional<int64_t> LU_data_bdim;
  std::tie(LU_data_value, LU_data_bdim) = unwrapTensorAtLevel(LU_data, cur_level);
  Tensor LU_pivots_value;
  optional<int64_t> LU_pivots_bdim;
  std::tie(LU_pivots_value, LU_pivots_bdim) = unwrapTensorAtLevel(LU_pivots, cur_level);
  auto results = batch_rule(self_value, self_bdim, LU_data_value, LU_data_bdim, LU_pivots_value, LU_pivots_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> lu_unpack_generated_plumbing(const Tensor & LU_data, const Tensor & LU_pivots, bool unpack_data, bool unpack_pivots) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor multinomial_generated_plumbing(const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor lgamma_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor digamma_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor polygamma_generated_plumbing(int64_t n, const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor erfinv_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor i0_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sign_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor signbit_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor dist_generated_plumbing(const Tensor & self, const Tensor & other, const Scalar & p) {
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
  auto results = batch_rule(self_value, self_bdim, other_value, other_bdim, p);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor atan2_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor arctan2_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor lerp_Scalar_generated_plumbing(const Tensor & self, const Tensor & end, const Scalar & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor end_value;
  optional<int64_t> end_bdim;
  std::tie(end_value, end_bdim) = unwrapTensorAtLevel(end, cur_level);
  auto results = batch_rule(self_value, self_bdim, end_value, end_bdim, weight);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor lerp_Tensor_generated_plumbing(const Tensor & self, const Tensor & end, const Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor end_value;
  optional<int64_t> end_bdim;
  std::tie(end_value, end_bdim) = unwrapTensorAtLevel(end, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  auto results = batch_rule(self_value, self_bdim, end_value, end_bdim, weight_value, weight_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor histc_generated_plumbing(const Tensor & self, int64_t bins, const Scalar & min, const Scalar & max) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> histogram_bins_tensor_generated_plumbing(const Tensor & self, const Tensor & bins, const c10::optional<Tensor> & weight, bool density) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> histogram_bin_ct_generated_plumbing(const Tensor & self, int64_t bins, c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<Tensor> _histogramdd_bin_edges_generated_plumbing(const Tensor & self, IntArrayRef bins, c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _histogramdd_from_bin_cts_generated_plumbing(const Tensor & self, IntArrayRef bins, c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fmod_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fmod_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hypot_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor igamma_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor igammac_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nextafter_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor remainder_Scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor remainder_Tensor_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor remainder_Scalar_Tensor_generated_plumbing(const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor min_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fmin_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fmax_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor maximum_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_other_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor minimum_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor min_other_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantile_generated_plumbing(const Tensor & self, const Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor quantile_scalar_generated_plumbing(const Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nanquantile_generated_plumbing(const Tensor & self, const Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nanquantile_scalar_generated_plumbing(const Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> sort_generated_plumbing(const Tensor & self, int64_t dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> sort_stable_generated_plumbing(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> sort_dimname_generated_plumbing(const Tensor & self, Dimname dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> sort_dimname_stable_generated_plumbing(const Tensor & self, c10::optional<bool> stable, Dimname dim, bool descending) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor msort_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor argsort_generated_plumbing(const Tensor & self, int64_t dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor argsort_dimname_generated_plumbing(const Tensor & self, Dimname dim, bool descending) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, descending);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> topk_generated_plumbing(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor all_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor any_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor renorm_generated_plumbing(const Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor unfold_generated_plumbing(const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dimension, size, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor unfold_backward_generated_plumbing(const Tensor & grad_in, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_in_value;
  optional<int64_t> grad_in_bdim;
  std::tie(grad_in_value, grad_in_bdim) = unwrapTensorAtLevel(grad_in, cur_level);
  auto results = batch_rule(grad_in_value, grad_in_bdim, input_sizes, dim, size, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
bool equal_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pow_Tensor_Tensor_generated_plumbing(const Tensor & self, const Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor exponent_value;
  optional<int64_t> exponent_bdim;
  std::tie(exponent_value, exponent_bdim) = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pow_Scalar_generated_plumbing(const Scalar & self, const Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor exponent_value;
  optional<int64_t> exponent_bdim;
  std::tie(exponent_value, exponent_bdim) = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor pow_Tensor_Scalar_generated_plumbing(const Tensor & self, const Scalar & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor float_power_Tensor_Tensor_generated_plumbing(const Tensor & self, const Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor exponent_value;
  optional<int64_t> exponent_bdim;
  std::tie(exponent_value, exponent_bdim) = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor float_power_Scalar_generated_plumbing(const Scalar & self, const Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor exponent_value;
  optional<int64_t> exponent_bdim;
  std::tie(exponent_value, exponent_bdim) = unwrapTensorAtLevel(exponent, cur_level);
  auto results = batch_rule(self, exponent_value, exponent_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor float_power_Tensor_Scalar_generated_plumbing(const Tensor & self, const Scalar & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor normal_Tensor_float_generated_plumbing(const Tensor & mean, double std, c10::optional<Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  auto results = batch_rule(mean_value, mean_bdim, std, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor normal_float_Tensor_generated_plumbing(double mean, const Tensor & std, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor normal_Tensor_Tensor_generated_plumbing(const Tensor & mean, const Tensor & std, c10::optional<Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor mean_value;
  optional<int64_t> mean_bdim;
  std::tie(mean_value, mean_bdim) = unwrapTensorAtLevel(mean, cur_level);
  Tensor std_value;
  optional<int64_t> std_bdim;
  std::tie(std_value, std_bdim) = unwrapTensorAtLevel(std, cur_level);
  auto results = batch_rule(mean_value, mean_bdim, std_value, std_bdim, generator);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor alias_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bucketize_Tensor_generated_plumbing(const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor boundaries_value;
  optional<int64_t> boundaries_bdim;
  std::tie(boundaries_value, boundaries_bdim) = unwrapTensorAtLevel(boundaries, cur_level);
  auto results = batch_rule(self_value, self_bdim, boundaries_value, boundaries_bdim, out_int32, right);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor bucketize_Scalar_generated_plumbing(const Scalar & self, const Tensor & boundaries, bool out_int32, bool right) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor boundaries_value;
  optional<int64_t> boundaries_bdim;
  std::tie(boundaries_value, boundaries_bdim) = unwrapTensorAtLevel(boundaries, cur_level);
  auto results = batch_rule(self, boundaries_value, boundaries_bdim, out_int32, right);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor searchsorted_Tensor_generated_plumbing(const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _torch_cuda_cu_linker_symbol_op_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor searchsorted_Scalar_generated_plumbing(const Tensor & sorted_sequence, const Scalar & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _convert_indices_from_coo_to_csr_generated_plumbing(const Tensor & self, int64_t size, bool out_int32) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, out_int32);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _convert_indices_from_csr_to_coo_generated_plumbing(const Tensor & crow_indices, const Tensor & col_indices, bool out_int32, bool transpose) {
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
  auto results = batch_rule(crow_indices_value, crow_indices_bdim, col_indices_value, col_indices_bdim, out_int32, transpose);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mse_loss_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mse_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor l1_loss_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor l1_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor multi_margin_loss_generated_plumbing(const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor multi_margin_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar & p, const Scalar & margin, const c10::optional<Tensor> & weight, int64_t reduction) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor multilabel_margin_loss_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> multilabel_margin_loss_forward_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor multilabel_margin_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nll_loss_nd_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nll_loss_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> nll_loss_forward_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nll_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nll_loss2d_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> nll_loss2d_forward_generated_plumbing(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor nll_loss2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor smooth_l1_loss_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction, double beta) {
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction, beta);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor smooth_l1_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double beta) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor huber_loss_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction, double delta) {
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction, delta);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor huber_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction, delta);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor soft_margin_loss_generated_plumbing(const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor soft_margin_loss_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor elu_generated_plumbing(const Tensor & self, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor elu_backward_generated_plumbing(const Tensor & grad_output, const Scalar & alpha, const Scalar & scale, const Scalar & input_scale, bool is_result, const Tensor & self_or_result) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor glu_generated_plumbing(const Tensor & self, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor glu_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, int64_t dim) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardsigmoid_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardsigmoid_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardtanh_generated_plumbing(const Tensor & self, const Scalar & min_val, const Scalar & max_val) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min_val, max_val);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardtanh_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Scalar & min_val, const Scalar & max_val) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardswish_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor hardswish_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor leaky_relu_generated_plumbing(const Tensor & self, const Scalar & negative_slope) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, negative_slope);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor leaky_relu_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Scalar & negative_slope, bool self_is_result) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log_sigmoid_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> log_sigmoid_forward_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor log_sigmoid_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
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
  Tensor buffer_value;
  optional<int64_t> buffer_bdim;
  std::tie(buffer_value, buffer_bdim) = unwrapTensorAtLevel(buffer, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, buffer_value, buffer_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rrelu_with_noise_generated_plumbing(const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, c10::optional<Generator> generator) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor rrelu_with_noise_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & noise, const Scalar & lower, const Scalar & upper, bool training, bool self_is_result) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor softplus_generated_plumbing(const Tensor & self, const Scalar & beta, const Scalar & threshold) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, beta, threshold);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor softplus_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Scalar & beta, const Scalar & threshold) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, beta, threshold);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor softshrink_generated_plumbing(const Tensor & self, const Scalar & lambd) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, lambd);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor softshrink_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Scalar & lambd) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, lambd);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor adaptive_avg_pool2d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_adaptive_avg_pool2d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor mkldnn_adaptive_avg_pool2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _adaptive_avg_pool2d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _adaptive_avg_pool2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor adaptive_avg_pool3d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _adaptive_avg_pool3d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _adaptive_avg_pool3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> adaptive_max_pool2d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor adaptive_max_pool2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, indices_value, indices_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> adaptive_max_pool3d_generated_plumbing(const Tensor & self, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor adaptive_max_pool3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, indices_value, indices_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor avg_pool2d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor avg_pool2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor avg_pool3d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor avg_pool3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> fractional_max_pool2d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fractional_max_pool2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> fractional_max_pool3d_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fractional_max_pool3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> max_pool2d_with_indices_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_pool2d_with_indices_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> max_pool3d_with_indices_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_pool3d_with_indices_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_unpool2d_generated_plumbing(const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
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
  auto results = batch_rule(self_value, self_bdim, indices_value, indices_bdim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_unpool2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_unpool3d_generated_plumbing(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor max_unpool3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reflection_pad1d_generated_plumbing(const Tensor & self, IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reflection_pad1d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reflection_pad2d_generated_plumbing(const Tensor & self, IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reflection_pad2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reflection_pad3d_generated_plumbing(const Tensor & self, IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor reflection_pad3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor replication_pad1d_generated_plumbing(const Tensor & self, IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor replication_pad1d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor replication_pad2d_generated_plumbing(const Tensor & self, IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor replication_pad2d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor replication_pad3d_generated_plumbing(const Tensor & self, IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor replication_pad3d_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_linear1d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_linear1d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bilinear2d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bilinear2d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bilinear2d_aa_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bilinear2d_aa_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_trilinear3d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_trilinear3d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bicubic2d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bicubic2d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bicubic2d_aa_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bicubic2d_aa_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, bool align_corners, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest1d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact1d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest1d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact1d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest2d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact2d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest2d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact2d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest3d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact3d_vec_generated_plumbing(const Tensor & input, c10::optional<IntArrayRef> output_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest3d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact3d_backward_vec_generated_plumbing(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_linear1d_generated_plumbing(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_linear1d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bilinear2d_generated_plumbing(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bilinear2d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bilinear2d_aa_generated_plumbing(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bilinear2d_aa_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bicubic2d_generated_plumbing(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_bicubic2d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bicubic2d_aa_generated_plumbing(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_bicubic2d_aa_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_trilinear3d_generated_plumbing(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_trilinear3d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest1d_generated_plumbing(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact1d_generated_plumbing(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest1d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact1d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest2d_generated_plumbing(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact2d_generated_plumbing(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest2d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact2d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest3d_generated_plumbing(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact3d_generated_plumbing(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor upsample_nearest3d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _upsample_nearest_exact3d_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor sigmoid_backward_generated_plumbing(const Tensor & grad_output, const Tensor & output) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor logit_backward_generated_plumbing(const Tensor & grad_output, const Tensor & self, c10::optional<double> eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor tanh_backward_generated_plumbing(const Tensor & grad_output, const Tensor & output) {
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slow_conv_transpose2d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slow_conv_transpose3d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor thnn_conv2d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _slow_conv2d_forward_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _slow_conv2d_backward_output_mask_generated_plumbing(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, ::std::array<bool,3> output_mask) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _conv_depthwise2d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor conv_depthwise3d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slow_conv3d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slow_conv3d_forward_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slow_conv_dilated2d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor slow_conv_dilated3d_generated_plumbing(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor col2im_generated_plumbing(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor col2im_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor im2col_generated_plumbing(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, dilation, padding, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor im2col_backward_generated_plumbing(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_size, kernel_size, dilation, padding, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isfinite_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isinf_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isposinf_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor isneginf_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _add_batch_dim_generated_plumbing(const Tensor & self, int64_t batch_dim, int64_t level) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, batch_dim, level);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _remove_batch_dim_generated_plumbing(const Tensor & self, int64_t level, int64_t batch_size, int64_t out_dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, level, batch_size, out_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_entr_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_ndtri_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_expm1_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_exp2_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_psi_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_digamma_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_gammaln_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_erf_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_erfc_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_erfcx_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_erfinv_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_ndtr_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_xlog1py_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_xlog1py_self_scalar_generated_plumbing(const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_xlog1py_other_scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_xlogy_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_xlogy_self_scalar_generated_plumbing(const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_xlogy_other_scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_zeta_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_zeta_self_scalar_generated_plumbing(const Scalar & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_zeta_other_scalar_generated_plumbing(const Tensor & self, const Scalar & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_i0_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_i0e_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_i1_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_i1e_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_logit_generated_plumbing(const Tensor & self, c10::optional<double> eps) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_polygamma_generated_plumbing(int64_t n, const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_logsumexp_generated_plumbing(const Tensor & self, IntArrayRef dim, bool keepdim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_expit_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_sinc_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_round_generated_plumbing(const Tensor & self, int64_t decimals) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, decimals);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_log1p_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_log_softmax_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_gammainc_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_gammaincc_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_multigammaln_generated_plumbing(const Tensor & self, int64_t p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor special_softmax_generated_plumbing(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_fft_generated_plumbing(const Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_ifft_generated_plumbing(const Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_rfft_generated_plumbing(const Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_irfft_generated_plumbing(const Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_hfft_generated_plumbing(const Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_ihfft_generated_plumbing(const Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_fft2_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, IntArrayRef dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_ifft2_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, IntArrayRef dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_rfft2_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, IntArrayRef dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_irfft2_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, IntArrayRef dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_hfft2_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, IntArrayRef dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_ihfft2_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, IntArrayRef dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_fftn_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, c10::optional<IntArrayRef> dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_ifftn_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, c10::optional<IntArrayRef> dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_rfftn_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, c10::optional<IntArrayRef> dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_irfftn_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, c10::optional<IntArrayRef> dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_hfftn_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, c10::optional<IntArrayRef> dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_ihfftn_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> s, c10::optional<IntArrayRef> dim, c10::optional<c10::string_view> norm) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_fftshift_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor fft_ifftshift_generated_plumbing(const Tensor & self, c10::optional<IntArrayRef> dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> linalg_cholesky_ex_generated_plumbing(const Tensor & self, bool upper, bool check_errors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upper, check_errors);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_cholesky_generated_plumbing(const Tensor & self, bool upper) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upper);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_cross_generated_plumbing(const Tensor & self, const Tensor & other, int64_t dim) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> linalg_lu_factor_generated_plumbing(const Tensor & A, bool pivot) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(A_value, A_bdim, pivot);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> linalg_lu_factor_ex_generated_plumbing(const Tensor & A, bool pivot, bool check_errors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(A_value, A_bdim, pivot, check_errors);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_det_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor det_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _det_lu_based_helper_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _det_lu_based_helper_backward_helper_generated_plumbing(const Tensor & det_grad, const Tensor & det, const Tensor & self, const Tensor & lu, const Tensor & pivs) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor,Tensor> linalg_lstsq_generated_plumbing(const Tensor & self, const Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matmul_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_exp_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> linalg_slogdet_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> linalg_eig_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_eigvals_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> linalg_eigh_generated_plumbing(const Tensor & self, c10::string_view UPLO) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_eigvalsh_generated_plumbing(const Tensor & self, c10::string_view UPLO) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, UPLO);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_householder_product_generated_plumbing(const Tensor & input, const Tensor & tau) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor tau_value;
  optional<int64_t> tau_bdim;
  std::tie(tau_value, tau_bdim) = unwrapTensorAtLevel(tau, cur_level);
  auto results = batch_rule(input_value, input_bdim, tau_value, tau_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> linalg_inv_ex_generated_plumbing(const Tensor & self, bool check_errors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, check_errors);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_inv_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor inner_generated_plumbing(const Tensor & self, const Tensor & other) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor outer_generated_plumbing(const Tensor & self, const Tensor & vec2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor vec2_value;
  optional<int64_t> vec2_bdim;
  std::tie(vec2_value, vec2_bdim) = unwrapTensorAtLevel(vec2, cur_level);
  auto results = batch_rule(self_value, self_bdim, vec2_value, vec2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor ger_generated_plumbing(const Tensor & self, const Tensor & vec2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor vec2_value;
  optional<int64_t> vec2_bdim;
  std::tie(vec2_value, vec2_bdim) = unwrapTensorAtLevel(vec2, cur_level);
  auto results = batch_rule(self_value, self_bdim, vec2_value, vec2_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_norm_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_norm_ord_str_generated_plumbing(const Tensor & self, c10::string_view ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_vector_norm_generated_plumbing(const Tensor & self, const Scalar & ord, c10::optional<IntArrayRef> dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_norm_generated_plumbing(const Tensor & self, const Scalar & ord, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_norm_str_ord_generated_plumbing(const Tensor & self, c10::string_view ord, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> _linalg_svd_generated_plumbing(const Tensor & A, bool full_matrices, bool compute_uv) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(A_value, A_bdim, full_matrices, compute_uv);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor,Tensor> linalg_svd_generated_plumbing(const Tensor & A, bool full_matrices) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_svdvals_generated_plumbing(const Tensor & A) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor A_value;
  optional<int64_t> A_bdim;
  std::tie(A_value, A_bdim) = unwrapTensorAtLevel(A, cur_level);
  auto results = batch_rule(A_value, A_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_cond_generated_plumbing(const Tensor & self, const c10::optional<Scalar> & p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_cond_p_str_generated_plumbing(const Tensor & self, c10::string_view p) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_pinv_atol_rtol_tensor_generated_plumbing(const Tensor & self, const c10::optional<Tensor> & atol, const c10::optional<Tensor> & rtol, bool hermitian) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_pinv_atol_rtol_float_generated_plumbing(const Tensor & self, c10::optional<double> atol, c10::optional<double> rtol, bool hermitian) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_pinv_generated_plumbing(const Tensor & self, double rcond, bool hermitian) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, rcond, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_pinv_rcond_tensor_generated_plumbing(const Tensor & self, const Tensor & rcond, bool hermitian) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor rcond_value;
  optional<int64_t> rcond_bdim;
  std::tie(rcond_value, rcond_bdim) = unwrapTensorAtLevel(rcond, cur_level);
  auto results = batch_rule(self_value, self_bdim, rcond_value, rcond_bdim, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_solve_generated_plumbing(const Tensor & input, const Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  auto results = batch_rule(input_value, input_bdim, other_value, other_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_tensorinv_generated_plumbing(const Tensor & self, int64_t ind) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, ind);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_tensorsolve_generated_plumbing(const Tensor & self, const Tensor & other, c10::optional<IntArrayRef> dims) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> linalg_qr_generated_plumbing(const Tensor & self, c10::string_view mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, mode);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
std::tuple<Tensor,Tensor> _linalg_qr_helper_generated_plumbing(const Tensor & self, c10::string_view mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, mode);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_power_generated_plumbing(const Tensor & self, int64_t n) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, n);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_rank_atol_rtol_tensor_generated_plumbing(const Tensor & input, const c10::optional<Tensor> & atol, const c10::optional<Tensor> & rtol, bool hermitian) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
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
  auto results = batch_rule(input_value, input_bdim, atol_value, atol_bdim, rtol_value, rtol_bdim, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_rank_atol_rtol_float_generated_plumbing(const Tensor & self, c10::optional<double> atol, c10::optional<double> rtol, bool hermitian) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_rank_generated_plumbing(const Tensor & self, double tol, bool hermitian) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, tol, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor linalg_matrix_rank_tol_tensor_generated_plumbing(const Tensor & input, const Tensor & tol, bool hermitian) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  Tensor tol_value;
  optional<int64_t> tol_bdim;
  std::tie(tol_value, tol_bdim) = unwrapTensorAtLevel(tol, cur_level);
  auto results = batch_rule(input_value, input_bdim, tol_value, tol_bdim, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_serialization_subcmul_generated_plumbing(const Tensor & self, const Tensor & other, const Scalar & alpha) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_optional_intlist_generated_plumbing(const Tensor & values, c10::optional<IntArrayRef> addends) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_optional_filled_intlist_generated_plumbing(const Tensor & values, c10::optional<IntArrayRef> addends) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_optional_floatlist_generated_plumbing(const Tensor & values, c10::optional<ArrayRef<double>> addends) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_string_default_generated_plumbing(const Tensor & dummy, c10::string_view a, c10::string_view b) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_ambiguous_defaults_a_generated_plumbing(const Tensor & dummy, int64_t a, int64_t b) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_ambiguous_defaults_b_generated_plumbing(const Tensor & dummy, int64_t a, c10::string_view b) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _test_warn_in_autograd_generated_plumbing(const Tensor & self) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor segment_reduce_generated_plumbing(const Tensor & data, c10::string_view reduce, const c10::optional<Tensor> & lengths, const c10::optional<Tensor> & indices, int64_t axis, bool unsafe, const c10::optional<Scalar> & initial) {
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

template <typename batch_rule_t, batch_rule_t batch_rule>
Tensor _segment_reduce_backward_generated_plumbing(const Tensor & grad, const Tensor & output, const Tensor & data, c10::string_view reduce, const c10::optional<Tensor> & lengths, int64_t axis) {
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
