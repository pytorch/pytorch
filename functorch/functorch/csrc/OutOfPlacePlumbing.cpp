// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/OutOfPlacePlumbing.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>

namespace at { namespace functorch {
typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_0_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_0_t,Tensor,const Tensor &, bool>(
  batch_rule_0_t batch_rule,
  const Tensor & self, bool upper
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_1_t)(const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_1_t,Tensor,const Tensor &>(
  batch_rule_1_t batch_rule,
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
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_4_t)(const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_4_t,Tensor,const Tensor &, int64_t>(
  batch_rule_4_t batch_rule,
  const Tensor & self, int64_t n
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_5_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_5_t,Tensor,const Tensor &, const Tensor &, int64_t>(
  batch_rule_5_t batch_rule,
  const Tensor & grad_output, const Tensor & self, int64_t dim
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_6_t)(const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_6_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t>(
  batch_rule_6_t batch_rule,
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
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_7_t)(const Tensor &, c10::optional<int64_t>, c10::optional<DimnameList>);
template <>
Tensor lowerToNextLayer<batch_rule_7_t,Tensor,const Tensor &, c10::optional<DimnameList>>(
  batch_rule_7_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_8_t)(const Tensor &, c10::optional<int64_t>, DimnameList);
template <>
Tensor lowerToNextLayer<batch_rule_8_t,Tensor,const Tensor &, DimnameList>(
  batch_rule_8_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_9_t)(const Tensor &, c10::optional<int64_t>, DimnameList, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_9_t,Tensor,const Tensor &, DimnameList, int64_t>(
  batch_rule_9_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_10_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_10_t,Tensor,const Tensor &, const Tensor &>(
  batch_rule_10_t batch_rule,
  const Tensor & input, const Tensor & other
) {
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

typedef std::tuple<bool> (*batch_rule_11_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t);
template <>
bool lowerToNextLayer<batch_rule_11_t,bool,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_11_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_12_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_12_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_12_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_13_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<Generator>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_13_t,std::tuple<Tensor,Tensor>,const Tensor &, double, c10::optional<Generator>>(
  batch_rule_13_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_14_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double);
template <>
Tensor lowerToNextLayer<batch_rule_14_t,Tensor,const Tensor &, const Tensor &, double>(
  batch_rule_14_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_15_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, c10::optional<ScalarType>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_15_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, const Tensor &, int64_t, int64_t, c10::optional<ScalarType>>(
  batch_rule_15_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_16_t)(const Tensor &, c10::optional<int64_t>, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_16_t,Tensor,const Tensor &, double, bool>(
  batch_rule_16_t batch_rule,
  const Tensor & self, double rcond, bool hermitian
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_17_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_17_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool>(
  batch_rule_17_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_18_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_18_t,Tensor,const Tensor &, IntArrayRef>(
  batch_rule_18_t batch_rule,
  const Tensor & self, IntArrayRef padding
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_19_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_19_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef>(
  batch_rule_19_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_20_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_20_t,Tensor,const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_20_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_21_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_21_t,Tensor,const Tensor &, const Scalar &, const Scalar &>(
  batch_rule_21_t batch_rule,
  const Tensor & self, const Scalar & beta, const Scalar & threshold
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_22_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_22_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &>(
  batch_rule_22_t batch_rule,
  const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_23_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_23_t,Tensor,const Tensor &, IntArrayRef, bool>(
  batch_rule_23_t batch_rule,
  const Tensor & self, IntArrayRef dim, bool keepdim
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_24_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_24_t,Tensor,const Tensor &, int64_t, bool>(
  batch_rule_24_t batch_rule,
  const Tensor & self, int64_t dim, bool descending
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_25_t)(const Tensor &, c10::optional<int64_t>, Dimname, bool);
template <>
Tensor lowerToNextLayer<batch_rule_25_t,Tensor,const Tensor &, Dimname, bool>(
  batch_rule_25_t batch_rule,
  const Tensor & self, Dimname dim, bool descending
) {
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

typedef std::tuple<bool> (*batch_rule_26_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, bool);
template <>
bool lowerToNextLayer<batch_rule_26_t,bool,const Tensor &, const Tensor &, double, double, bool>(
  batch_rule_26_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_27_t)(const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_27_t,Tensor,const Tensor &, c10::optional<int64_t>, bool>(
  batch_rule_27_t batch_rule,
  const Tensor & x, c10::optional<int64_t> N, bool increasing
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_28_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_28_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(
  batch_rule_28_t batch_rule,
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

typedef std::tuple<const Tensor &,c10::optional<int64_t>> (*batch_rule_29_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<int64_t>);
template <>
const Tensor & lowerToNextLayer<batch_rule_29_t,const Tensor &,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(
  batch_rule_29_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_30_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_30_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double, bool>(
  batch_rule_30_t batch_rule,
  const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled
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
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, running_mean_value, running_mean_bdim, running_var_value, running_var_bdim, use_input_stats, momentum, eps, cudnn_enabled);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_31_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_31_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, double, double, int64_t>(
  batch_rule_31_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,int64_t> (*batch_rule_32_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> lowerToNextLayer<batch_rule_32_t,std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double, bool>(
  batch_rule_32_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_33_t)(const Tensor &, c10::optional<int64_t>, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_33_t,Tensor,const Tensor &, c10::optional<Generator>>(
  batch_rule_33_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_34_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_34_t,Tensor,const Tensor &, double, c10::optional<Generator>>(
  batch_rule_34_t batch_rule,
  const Tensor & mean, double std, c10::optional<Generator> generator
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_35_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_35_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &>(
  batch_rule_35_t batch_rule,
  const Tensor & grad, const Tensor & output, const Tensor & data, const c10::optional<Tensor> & lengths
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
  auto results = batch_rule(grad_value, grad_bdim, output_value, output_bdim, data_value, data_bdim, lengths_value, lengths_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_36_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_36_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_36_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_37_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_37_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_37_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_38_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_38_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_38_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_39_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_39_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_39_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_40_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_40_t,Tensor,const Tensor &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_40_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_41_t)(const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_41_t,Tensor,const Tensor &, const Scalar &>(
  batch_rule_41_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_42_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_42_t,Tensor,const Tensor &, const Tensor &, bool>(
  batch_rule_42_t batch_rule,
  const Tensor & input, const Tensor & tol, bool hermitian
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_43_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, const c10::optional<Scalar> &);
template <>
Tensor lowerToNextLayer<batch_rule_43_t,Tensor,const Tensor &, const c10::optional<Scalar> &, const c10::optional<Scalar> &>(
  batch_rule_43_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_44_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_44_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_44_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_45_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_45_t,Tensor,const Tensor &, IntArrayRef, const Scalar &>(
  batch_rule_45_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_46_t)(const Tensor &, c10::optional<int64_t>, MemoryFormat);
template <>
Tensor lowerToNextLayer<batch_rule_46_t,Tensor,const Tensor &, MemoryFormat>(
  batch_rule_46_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_47_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_47_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(
  batch_rule_47_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_48_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_48_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, bool>(
  batch_rule_48_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_49_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_49_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool>(
  batch_rule_49_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_50_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_50_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t>(
  batch_rule_50_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_51_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_51_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef>(
  batch_rule_51_t batch_rule,
  const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding
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
  auto results = batch_rule(input_value, input_bdim, weight_value, weight_bdim, bias_value, bias_bdim, stride, padding, dilation, transposed, output_padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_52_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_52_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_52_t batch_rule,
  const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
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
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim, padding, stride, dilation, groups);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_53_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_53_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t>(
  batch_rule_53_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_54_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_54_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(
  batch_rule_54_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_55_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_55_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(
  batch_rule_55_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_56_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_56_t,Tensor,const Tensor &, const Tensor &, const Tensor &, double, int64_t>(
  batch_rule_56_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_57_t)(const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_57_t,Tensor,const Tensor &, c10::optional<int64_t>>(
  batch_rule_57_t batch_rule,
  const Tensor & repeats, c10::optional<int64_t> output_size
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_58_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_58_t,Tensor,const Tensor &, int64_t, int64_t, int64_t, int64_t>(
  batch_rule_58_t batch_rule,
  const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_59_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_59_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double>(
  batch_rule_59_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_60_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_60_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, const Tensor &>(
  batch_rule_60_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_61_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_61_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_61_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_62_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_62_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_62_t batch_rule,
  const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, weight_value, weight_bdim, padding, stride, dilation, groups, benchmark, deterministic);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_63_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_63_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool>(
  batch_rule_63_t batch_rule,
  const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, weight_value, weight_bdim, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_64_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_64_t,Tensor,IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool>(
  batch_rule_64_t batch_rule,
  IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32
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
  auto results = batch_rule(weight_size, grad_output_value, grad_output_bdim, self_value, self_bdim, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_65_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_65_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_65_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_66_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_66_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_66_t batch_rule,
  const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic
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
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_67_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_67_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool>(
  batch_rule_67_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_68_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_68_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Scalar> &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_68_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_69_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_69_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &>(
  batch_rule_69_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & weight
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, weight_value, weight_bdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_70_t)(const Tensor &, c10::optional<int64_t>, Dimname);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_70_t,std::tuple<Tensor,Tensor>,const Tensor &, Dimname>(
  batch_rule_70_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_71_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_71_t,Tensor,const Tensor &, int64_t, c10::optional<ScalarType>>(
  batch_rule_71_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_72_t)(const Tensor &, c10::optional<int64_t>, Dimname, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_72_t,Tensor,const Tensor &, Dimname, c10::optional<ScalarType>>(
  batch_rule_72_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_73_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_73_t,Tensor,const Tensor &, const Tensor &, int64_t, const Tensor &>(
  batch_rule_73_t batch_rule,
  const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self
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
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_value, output_bdim, dim, self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_74_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_74_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, int64_t, bool>(
  batch_rule_74_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_75_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_75_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(
  batch_rule_75_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_76_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_76_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool>(
  batch_rule_76_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_77_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_77_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, int64_t, bool>(
  batch_rule_77_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_78_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_78_t,Tensor,const Tensor &, int64_t, int64_t, int64_t>(
  batch_rule_78_t batch_rule,
  const Tensor & self, int64_t level, int64_t batch_size, int64_t out_dim
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_79_t)(const Tensor &, c10::optional<int64_t>, Dimname, Dimname, Dimname, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_79_t,Tensor,const Tensor &, Dimname, Dimname, Dimname, int64_t>(
  batch_rule_79_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_80_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_80_t,Tensor,const Tensor &, IntArrayRef, int64_t, int64_t, int64_t>(
  batch_rule_80_t batch_rule,
  const Tensor & grad_in, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_81_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_81_t,Tensor,const Tensor &, int64_t, int64_t, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_81_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_82_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_82_t,Tensor,const Tensor &, const Tensor &, c10::optional<c10::string_view>>(
  batch_rule_82_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_83_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_83_t,Tensor,const Tensor &, const Scalar &, c10::optional<c10::string_view>>(
  batch_rule_83_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_84_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_84_t,Tensor,const Tensor &, const Tensor &, int64_t, bool, bool>(
  batch_rule_84_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_85_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_85_t,Tensor,const Tensor &, const Tensor &, int64_t, int64_t, bool, bool>(
  batch_rule_85_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_86_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_86_t,Tensor,const Tensor &, const Tensor &, int64_t, int64_t, bool>(
  batch_rule_86_t batch_rule,
  const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_87_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, int64_t);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_87_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const c10::optional<Tensor> &, bool, int64_t>(
  batch_rule_87_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_88_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, ScalarType);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_88_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, ScalarType>(
  batch_rule_88_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_89_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_89_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const c10::optional<Tensor> &, bool>(
  batch_rule_89_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_90_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_90_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const c10::optional<Tensor> &, bool, c10::optional<int64_t>>(
  batch_rule_90_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_91_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool, int64_t, bool, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_91_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const c10::optional<Tensor> &, int64_t>(
  batch_rule_91_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_92_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_92_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const c10::optional<Tensor> &, int64_t>(
  batch_rule_92_t batch_rule,
  const Tensor & grad, const Tensor & indices, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor> & per_sample_weights, int64_t padding_idx
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_93_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_93_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(
  batch_rule_93_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_94_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_94_t,Tensor,const Tensor &, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_94_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_95_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_95_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_95_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_96_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_96_t,Tensor,const Tensor &, IntArrayRef, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_96_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_97_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_97_t,Tensor,IntArrayRef, const Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_97_t batch_rule,
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

typedef std::tuple<const Tensor &,c10::optional<int64_t>> (*batch_rule_98_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<MemoryFormat>);
template <>
const Tensor & lowerToNextLayer<batch_rule_98_t,const Tensor &,const Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(
  batch_rule_98_t batch_rule,
  const Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_99_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_99_t,Tensor,IntArrayRef, const Tensor &>(
  batch_rule_99_t batch_rule,
  IntArrayRef size, const Tensor & qtensor
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor qtensor_value;
  optional<int64_t> qtensor_bdim;
  std::tie(qtensor_value, qtensor_bdim) = unwrapTensorAtLevel(qtensor, cur_level);
  auto results = batch_rule(size, qtensor_value, qtensor_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_100_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_100_t,Tensor,const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_100_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_101_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_101_t,Tensor,const Tensor &, int64_t, int64_t>(
  batch_rule_101_t batch_rule,
  const Tensor & dummy, int64_t a, int64_t b
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_102_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_102_t,Tensor,const Tensor &, int64_t, int64_t, Dimname>(
  batch_rule_102_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_103_t)(const Tensor &, c10::optional<int64_t>, Dimname, Dimname, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_103_t,Tensor,const Tensor &, Dimname, Dimname, Dimname>(
  batch_rule_103_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_104_t)(const Tensor &, c10::optional<int64_t>, DimnameList, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_104_t,Tensor,const Tensor &, DimnameList, Dimname>(
  batch_rule_104_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_105_t)(const Tensor &, c10::optional<int64_t>, int64_t, IntArrayRef, c10::optional<DimnameList>);
template <>
Tensor lowerToNextLayer<batch_rule_105_t,Tensor,const Tensor &, int64_t, IntArrayRef, c10::optional<DimnameList>>(
  batch_rule_105_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_106_t)(const Tensor &, c10::optional<int64_t>, Dimname, IntArrayRef, DimnameList);
template <>
Tensor lowerToNextLayer<batch_rule_106_t,Tensor,const Tensor &, Dimname, IntArrayRef, DimnameList>(
  batch_rule_106_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_107_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_107_t,Tensor,const Tensor &, const Scalar &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_107_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_108_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_108_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(
  batch_rule_108_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_109_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_109_t,Tensor,const Tensor &, const Tensor &, double, int64_t>(
  batch_rule_109_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_110_t)(const Tensor &, c10::optional<int64_t>, int64_t, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_110_t,Tensor,const Tensor &, int64_t, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, bool>(
  batch_rule_110_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_111_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t, int64_t, int64_t, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_111_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t, int64_t, int64_t, int64_t, double>(
  batch_rule_111_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_112_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_112_t,Tensor,const Tensor &, IntArrayRef, int64_t, bool>(
  batch_rule_112_t batch_rule,
  const Tensor & self, IntArrayRef dim, int64_t normalization, bool forward
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_113_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_113_t,Tensor,const Tensor &, IntArrayRef, int64_t, int64_t>(
  batch_rule_113_t batch_rule,
  const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t index
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_sizes, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_114_t)(const Tensor &, c10::optional<int64_t>, const c10::List<c10::optional<Tensor>> &);
template <>
Tensor lowerToNextLayer<batch_rule_114_t,Tensor,const Tensor &, const c10::List<c10::optional<Tensor>> &>(
  batch_rule_114_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_115_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_115_t,Tensor,const Tensor &, int64_t, const Tensor &, const Tensor &>(
  batch_rule_115_t batch_rule,
  const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad
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
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, grad_value, grad_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_116_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_116_t,Tensor,const Tensor &, Dimname, const Tensor &, const Tensor &>(
  batch_rule_116_t batch_rule,
  const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src
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
  auto results = batch_rule(self_value, self_bdim, dim, index_value, index_bdim, src_value, src_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_117_t)(const Tensor &, c10::optional<int64_t>, const c10::List<c10::optional<Tensor>> &, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_117_t,Tensor,const Tensor &, const c10::List<c10::optional<Tensor>> &, const Tensor &, bool>(
  batch_rule_117_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_118_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_118_t,Tensor,const Tensor &, const Tensor &, double, double, bool>(
  batch_rule_118_t batch_rule,
  const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim
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
  auto results = batch_rule(x1_value, x1_bdim, x2_value, x2_bdim, p, eps, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<bool> (*batch_rule_119_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
bool lowerToNextLayer<batch_rule_119_t,bool,const Tensor &, const Tensor &>(
  batch_rule_119_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_120_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_120_t,Tensor,const Tensor &, const Tensor &, int64_t, bool>(
  batch_rule_120_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_121_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_121_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, bool>(
  batch_rule_121_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_122_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_122_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, int64_t, bool>(
  batch_rule_122_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_123_t)(const Tensor &, c10::optional<int64_t>, int64_t, Dimname, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_123_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, Dimname, bool>(
  batch_rule_123_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_124_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, bool);
template <>
Tensor lowerToNextLayer<batch_rule_124_t,Tensor,const Tensor &, IntArrayRef, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, bool>(
  batch_rule_124_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_125_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_125_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, IntArrayRef, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double>(
  batch_rule_125_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_126_t)(const Tensor &, c10::optional<int64_t>, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_126_t,Tensor,const Tensor &, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_126_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_127_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_127_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &>(
  batch_rule_127_t batch_rule,
  const Tensor & self, const Tensor & weight, const c10::optional<Tensor> & bias
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
  auto results = batch_rule(self_value, self_bdim, weight_value, weight_bdim, bias_value, bias_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_128_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_128_t,Tensor,IntArrayRef, const Tensor &, const Tensor &>(
  batch_rule_128_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_129_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_129_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, bool>(
  batch_rule_129_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_130_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_130_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, const Tensor &>(
  batch_rule_130_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,double,int64_t> (*batch_rule_131_t)(const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,double,int64_t> lowerToNextLayer<batch_rule_131_t,std::tuple<Tensor,Tensor,double,int64_t>,const Tensor &>(
  batch_rule_131_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_132_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_132_t,Tensor,const Tensor &, const Tensor &, const Tensor &>(
  batch_rule_132_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & indices
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, indices_value, indices_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_133_t)(const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_133_t,Tensor,const Scalar &, const Tensor &>(
  batch_rule_133_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_134_t)(const Tensor &, c10::optional<int64_t>, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_134_t,Tensor,const Tensor &, Dimname>(
  batch_rule_134_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_135_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool);
template <>
Tensor lowerToNextLayer<batch_rule_135_t,Tensor,const Tensor &, DimnameList, bool>(
  batch_rule_135_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_136_t)(const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_136_t,std::tuple<Tensor,Tensor>,const Tensor &>(
  batch_rule_136_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_137_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_137_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, bool>(
  batch_rule_137_t batch_rule,
  const Tensor & self, int64_t dim, bool descending
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_138_t)(const Tensor &, c10::optional<int64_t>, Dimname, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_138_t,std::tuple<Tensor,Tensor>,const Tensor &, Dimname, bool>(
  batch_rule_138_t batch_rule,
  const Tensor & self, Dimname dim, bool descending
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_139_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_139_t,Tensor,const Tensor &, int64_t, const Tensor &, IntArrayRef, bool>(
  batch_rule_139_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_140_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_140_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(
  batch_rule_140_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_141_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_141_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(
  batch_rule_141_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_142_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_142_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(
  batch_rule_142_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_143_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_143_t,Tensor,const Tensor &, c10::optional<ScalarType>>(
  batch_rule_143_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_144_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_144_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_144_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_145_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_145_t,Tensor,const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(
  batch_rule_145_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_146_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool);
template <>
Tensor lowerToNextLayer<batch_rule_146_t,Tensor,IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool>(
  batch_rule_146_t batch_rule,
  IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined
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
  auto results = batch_rule(self_size, grad_output_value, grad_output_bdim, weight_value, weight_bdim, padding, stride, dilation, groups, bias_defined);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_147_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_147_t,std::tuple<Tensor,Tensor>,IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool>(
  batch_rule_147_t batch_rule,
  IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined
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
  auto results = batch_rule(weight_size, grad_output_value, grad_output_bdim, self_value, self_bdim, padding, stride, dilation, groups, bias_defined);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_148_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_148_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double>(
  batch_rule_148_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_149_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_149_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double>(
  batch_rule_149_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_150_t)(IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_150_t,Tensor,IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(
  batch_rule_150_t batch_rule,
  IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic
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
  auto results = batch_rule(weight_size, grad_output_value, grad_output_bdim, self_value, self_bdim, padding, stride, dilation, groups, benchmark, deterministic);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_151_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_151_t,Tensor,const Tensor &, int64_t, const Tensor &, int64_t>(
  batch_rule_151_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_152_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, double, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_152_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double>(
  batch_rule_152_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_153_t)(const Tensor &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_153_t,std::tuple<Tensor,Tensor>,const Tensor &, double>(
  batch_rule_153_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_154_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double);
template <>
Tensor lowerToNextLayer<batch_rule_154_t,Tensor,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, double>(
  batch_rule_154_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_155_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, double, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_155_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, double, int64_t>(
  batch_rule_155_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_156_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double, double, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_156_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double, double, const Tensor &>(
  batch_rule_156_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_157_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_157_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, bool, bool, bool>(
  batch_rule_157_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_158_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_158_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, const Tensor &>(
  batch_rule_158_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_159_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, double);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_159_t,std::tuple<Tensor,Tensor>,const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, double>(
  batch_rule_159_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_160_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_160_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef>(
  batch_rule_160_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_161_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_161_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(
  batch_rule_161_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_162_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_162_t,Tensor,const Tensor &, IntArrayRef, const Tensor &, IntArrayRef>(
  batch_rule_162_t batch_rule,
  const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding
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
  auto results = batch_rule(input_value, input_bdim, weightsize, grad_output_value, grad_output_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_163_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_163_t,Tensor,const Tensor &, const Tensor &, double, c10::optional<int64_t>>(
  batch_rule_163_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_164_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_164_t,Tensor,const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(
  batch_rule_164_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_165_t)(const Tensor &, c10::optional<int64_t>, double);
template <>
Tensor lowerToNextLayer<batch_rule_165_t,Tensor,const Tensor &, double>(
  batch_rule_165_t batch_rule,
  const Tensor & self, double rcond
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_166_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_166_t,Tensor,const Tensor &, const Tensor &, double, const Tensor &>(
  batch_rule_166_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_167_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_167_t,Tensor,const Tensor &, const Tensor &, int64_t, double>(
  batch_rule_167_t batch_rule,
  const Tensor & self, const Tensor & target, int64_t reduction, double delta
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction, delta);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_168_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_168_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef>(
  batch_rule_168_t batch_rule,
  const Tensor & self, IntArrayRef shifts, IntArrayRef dims
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_169_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_169_t,Tensor,const Tensor &, const Tensor &, bool, bool, double, int64_t>(
  batch_rule_169_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_170_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_170_t,Tensor,const Tensor &, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_170_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_171_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_171_t,Tensor,const Tensor &, int64_t, int64_t, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, c10::optional<MemoryFormat>>(
  batch_rule_171_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_172_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_172_t,Tensor,const Tensor &, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>>(
  batch_rule_172_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_173_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_173_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>>(
  batch_rule_173_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_174_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, bool, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_174_t,Tensor,const Tensor &, const Scalar &, const Scalar &, bool, c10::optional<Generator>>(
  batch_rule_174_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_175_t)(const Tensor &, c10::optional<int64_t>, Dimname, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_175_t,Tensor,const Tensor &, Dimname, int64_t>(
  batch_rule_175_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_176_t)(const Tensor &, c10::optional<int64_t>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_176_t,Tensor,const Tensor &, c10::optional<double>>(
  batch_rule_176_t batch_rule,
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

typedef std::tuple<int64_t> (*batch_rule_177_t)(const Tensor &, c10::optional<int64_t>, int64_t);
template <>
int64_t lowerToNextLayer<batch_rule_177_t,int64_t,const Tensor &, int64_t>(
  batch_rule_177_t batch_rule,
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

typedef std::tuple<int64_t> (*batch_rule_178_t)(const Tensor &, c10::optional<int64_t>, Dimname);
template <>
int64_t lowerToNextLayer<batch_rule_178_t,int64_t,const Tensor &, Dimname>(
  batch_rule_178_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_179_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_179_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, int64_t>(
  batch_rule_179_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_180_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_180_t,Tensor,const Tensor &, IntArrayRef, int64_t, int64_t, int64_t, int64_t>(
  batch_rule_180_t batch_rule,
  const Tensor & grad, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor grad_value;
  optional<int64_t> grad_bdim;
  std::tie(grad_value, grad_bdim) = unwrapTensorAtLevel(grad, cur_level);
  auto results = batch_rule(grad_value, grad_bdim, input_sizes, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_181_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, c10::optional<bool>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_181_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, bool, c10::optional<bool>, c10::optional<bool>>(
  batch_rule_181_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_182_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, bool, bool, c10::optional<bool>, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_182_t,Tensor,const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const c10::optional<Tensor> &, bool, bool, c10::optional<bool>, c10::optional<int64_t>, bool>(
  batch_rule_182_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_183_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_183_t,Tensor,const Tensor &, IntArrayRef, bool, bool>(
  batch_rule_183_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_184_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_184_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool>(
  batch_rule_184_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_185_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_185_t,std::tuple<Tensor,Tensor>,const Tensor &, bool>(
  batch_rule_185_t batch_rule,
  const Tensor & self, bool check_errors
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_186_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_186_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef, bool, bool>(
  batch_rule_186_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_187_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_187_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::optional<IntArrayRef>, c10::optional<int64_t>, bool>(
  batch_rule_187_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_188_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_188_t,std::tuple<Tensor,Tensor>,const Tensor &, DimnameList, bool, bool>(
  batch_rule_188_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_189_t)(const Tensor &, c10::optional<int64_t>, DimnameList, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_189_t,std::tuple<Tensor,Tensor>,const Tensor &, DimnameList, c10::optional<int64_t>, bool>(
  batch_rule_189_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_190_t)(const Tensor &, c10::optional<int64_t>, DimnameList, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_190_t,Tensor,const Tensor &, DimnameList, bool, bool>(
  batch_rule_190_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_191_t)(const Tensor &, c10::optional<int64_t>, DimnameList, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_191_t,Tensor,const Tensor &, DimnameList, c10::optional<int64_t>, bool>(
  batch_rule_191_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_192_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_192_t,Tensor,const Tensor &, int64_t, bool, c10::optional<ScalarType>>(
  batch_rule_192_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_193_t)(const Tensor &, c10::optional<int64_t>, Dimname, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_193_t,Tensor,const Tensor &, Dimname, bool, c10::optional<ScalarType>>(
  batch_rule_193_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_194_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_194_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(
  batch_rule_194_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_195_t)(const Tensor &, c10::optional<int64_t>, Dimname, Dimname);
template <>
Tensor lowerToNextLayer<batch_rule_195_t,Tensor,const Tensor &, Dimname, Dimname>(
  batch_rule_195_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_196_t)(const Tensor &, c10::optional<int64_t>, int64_t, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_196_t,Tensor,const Tensor &, int64_t, IntArrayRef>(
  batch_rule_196_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_197_t)(const Tensor &, c10::optional<int64_t>, double, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_197_t,Tensor,const Tensor &, double, int64_t>(
  batch_rule_197_t batch_rule,
  const Tensor & self, double scale, int64_t zero_point
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_198_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_198_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_198_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_199_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, double, double, double, bool, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_199_t,Tensor,const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t>(
  batch_rule_199_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_200_t)(const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_200_t,std::tuple<Tensor,Tensor>,const Tensor &, bool, bool>(
  batch_rule_200_t batch_rule,
  const Tensor & self, bool eigenvectors, bool upper
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_201_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_201_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, int64_t, bool, bool, bool>(
  batch_rule_201_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_202_t)(const Tensor &, c10::optional<int64_t>, bool, bool, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_202_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool, bool, c10::optional<int64_t>>(
  batch_rule_202_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_203_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_203_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, int64_t, bool, bool>(
  batch_rule_203_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_204_t)(const Tensor &, c10::optional<int64_t>, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_204_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool, bool, bool>(
  batch_rule_204_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_205_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_205_t,Tensor,const Tensor &, const Scalar &, const Tensor &>(
  batch_rule_205_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_206_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_206_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, int64_t>(
  batch_rule_206_t batch_rule,
  const Tensor & self, const Tensor & target, int64_t reduction
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
  auto results = batch_rule(self_value, self_bdim, target_value, target_bdim, reduction);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_207_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_207_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(
  batch_rule_207_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_208_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_208_t,Tensor,const Tensor &, const Tensor &, c10::optional<Generator>>(
  batch_rule_208_t batch_rule,
  const Tensor & mean, const Tensor & std, c10::optional<Generator> generator
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_209_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_209_t,Tensor,const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_209_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_210_t)(const Tensor &, c10::optional<int64_t>, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_210_t,Tensor,const Tensor &, ScalarType>(
  batch_rule_210_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_211_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_211_t,Tensor,const Tensor &, IntArrayRef, ScalarType>(
  batch_rule_211_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_212_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_212_t,Tensor,const Tensor &, const Tensor &, IntArrayRef>(
  batch_rule_212_t batch_rule,
  const Tensor & grad_output, const Tensor & self, IntArrayRef padding
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_213_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_213_t,Tensor,const Tensor &, const c10::optional<Scalar> &, ScalarType>(
  batch_rule_213_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_214_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, IntArrayRef, bool, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_214_t,Tensor,const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool, ScalarType>(
  batch_rule_214_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_215_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, IntArrayRef, bool);
template <>
Tensor lowerToNextLayer<batch_rule_215_t,Tensor,const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool>(
  batch_rule_215_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_216_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, DimnameList, bool, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_216_t,Tensor,const Tensor &, const c10::optional<Scalar> &, DimnameList, bool, ScalarType>(
  batch_rule_216_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_217_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, DimnameList, bool);
template <>
Tensor lowerToNextLayer<batch_rule_217_t,Tensor,const Tensor &, const c10::optional<Scalar> &, DimnameList, bool>(
  batch_rule_217_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_218_t)(const Tensor &, c10::optional<int64_t>, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_218_t,Tensor,const Tensor &, c10::optional<MemoryFormat>>(
  batch_rule_218_t batch_rule,
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

typedef std::tuple<const Tensor &,c10::optional<int64_t>> (*batch_rule_219_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<MemoryFormat>);
template <>
const Tensor & lowerToNextLayer<batch_rule_219_t,const Tensor &,const Tensor &, const Tensor &, c10::optional<MemoryFormat>>(
  batch_rule_219_t batch_rule,
  const Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor the_template_value;
  optional<int64_t> the_template_bdim;
  std::tie(the_template_value, the_template_bdim) = unwrapTensorAtLevel(the_template, cur_level);
  auto results = batch_rule(self_value, self_bdim, the_template_value, the_template_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<const Tensor &,c10::optional<int64_t>> (*batch_rule_220_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
const Tensor & lowerToNextLayer<batch_rule_220_t,const Tensor &,const Tensor &, const Tensor &>(
  batch_rule_220_t batch_rule,
  const Tensor & self, const Tensor & the_template
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor the_template_value;
  optional<int64_t> the_template_bdim;
  std::tie(the_template_value, the_template_bdim) = unwrapTensorAtLevel(the_template, cur_level);
  auto results = batch_rule(self_value, self_bdim, the_template_value, the_template_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_221_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_221_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_221_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_222_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_222_t,Tensor,const Tensor &, const Tensor &, const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_222_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_223_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_223_t,Tensor,const Tensor &, const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_223_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_224_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_224_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_224_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_225_t)(int64_t, int64_t, IntArrayRef, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>);
template <>
Tensor lowerToNextLayer<batch_rule_225_t,Tensor,int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>>(
  batch_rule_225_t batch_rule,
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

typedef std::tuple<const Tensor &,c10::optional<int64_t>> (*batch_rule_226_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, int64_t);
template <>
const Tensor & lowerToNextLayer<batch_rule_226_t,const Tensor &,const Tensor &, IntArrayRef, int64_t, int64_t>(
  batch_rule_226_t batch_rule,
  const Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, sparse_dim, dense_dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_227_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_227_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
  batch_rule_227_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_228_t)(const Tensor &, c10::optional<int64_t>, double, int64_t, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_228_t,Tensor,const Tensor &, double, int64_t, ScalarType>(
  batch_rule_228_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_229_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, ScalarType);
template <>
Tensor lowerToNextLayer<batch_rule_229_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, ScalarType>(
  batch_rule_229_t batch_rule,
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

typedef std::tuple<double> (*batch_rule_230_t)(const Tensor &, c10::optional<int64_t>);
template <>
double lowerToNextLayer<batch_rule_230_t,double,const Tensor &>(
  batch_rule_230_t batch_rule,
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

typedef std::tuple<QScheme> (*batch_rule_231_t)(const Tensor &, c10::optional<int64_t>);
template <>
QScheme lowerToNextLayer<batch_rule_231_t,QScheme,const Tensor &>(
  batch_rule_231_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_232_t)(const Tensor &, c10::optional<int64_t>, double, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_232_t,Tensor,const Tensor &, double, int64_t, int64_t, int64_t>(
  batch_rule_232_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_233_t)(const Tensor &, c10::optional<int64_t>, double, int64_t, int64_t, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_233_t,std::tuple<Tensor,Tensor>,const Tensor &, double, int64_t, int64_t, int64_t>(
  batch_rule_233_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_234_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_234_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double>(
  batch_rule_234_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_235_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_235_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double>(
  batch_rule_235_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_236_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_236_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(
  batch_rule_236_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_237_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_237_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(
  batch_rule_237_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_238_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_238_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, double>(
  batch_rule_238_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_239_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, int64_t, int64_t, double);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_239_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, double>(
  batch_rule_239_t batch_rule,
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

typedef std::tuple<double,int64_t> (*batch_rule_240_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<double,int64_t> lowerToNextLayer<batch_rule_240_t,std::tuple<double,int64_t>,const Tensor &, bool>(
  batch_rule_240_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_241_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, double, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_241_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, int64_t, double, int64_t>(
  batch_rule_241_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_242_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_242_t,Tensor,const Tensor &, c10::optional<ScalarType>, c10::optional<Layout>, c10::optional<Device>, c10::optional<bool>, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_242_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_243_t)(const Tensor &, c10::optional<int64_t>, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_243_t,Tensor,const Tensor &, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_243_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_244_t)(const Tensor &, c10::optional<int64_t>, ScalarType, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_244_t,Tensor,const Tensor &, ScalarType, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_244_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_245_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, c10::optional<MemoryFormat>);
template <>
Tensor lowerToNextLayer<batch_rule_245_t,Tensor,const Tensor &, const Tensor &, bool, bool, c10::optional<MemoryFormat>>(
  batch_rule_245_t batch_rule,
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

typedef std::tuple<Scalar> (*batch_rule_246_t)(const Tensor &, c10::optional<int64_t>);
template <>
Scalar lowerToNextLayer<batch_rule_246_t,Scalar,const Tensor &>(
  batch_rule_246_t batch_rule,
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

typedef std::tuple<ScalarType> (*batch_rule_247_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
ScalarType lowerToNextLayer<batch_rule_247_t,ScalarType,const Tensor &, const Tensor &>(
  batch_rule_247_t batch_rule,
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

typedef std::tuple<ScalarType> (*batch_rule_248_t)(const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
ScalarType lowerToNextLayer<batch_rule_248_t,ScalarType,const Tensor &, const Scalar &>(
  batch_rule_248_t batch_rule,
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

typedef std::tuple<ScalarType> (*batch_rule_249_t)(const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
ScalarType lowerToNextLayer<batch_rule_249_t,ScalarType,const Scalar &, const Tensor &>(
  batch_rule_249_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_250_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_250_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_250_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_251_t)(const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_251_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, const Tensor &, bool>(
  batch_rule_251_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_252_t)(const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_252_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const Tensor &, const Tensor &>(
  batch_rule_252_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_253_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_253_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_253_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_254_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_254_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, bool>(
  batch_rule_254_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_255_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_255_t,std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_255_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_256_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_256_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &>(
  batch_rule_256_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_257_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_257_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, const Scalar &, const Scalar &>(
  batch_rule_257_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_258_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_258_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, bool>(
  batch_rule_258_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_259_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_259_t,Tensor,const Tensor &, IntArrayRef, const Tensor &, bool>(
  batch_rule_259_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_260_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, const Scalar &, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_260_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, bool, const Scalar &, int64_t>(
  batch_rule_260_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_261_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_261_t,Tensor,const Tensor &, const Tensor &, const Tensor &, bool>(
  batch_rule_261_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_262_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_262_t,Tensor,const Tensor &, int64_t, const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_262_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_263_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_263_t,Tensor,const Tensor &, Dimname, const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_263_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_264_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_264_t,Tensor,const Tensor &, int64_t, const Tensor &, const Scalar &>(
  batch_rule_264_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_265_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_265_t,Tensor,const Tensor &, Dimname, const Tensor &, const Scalar &>(
  batch_rule_265_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_266_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_266_t,Tensor,const Tensor &, int64_t, const Tensor &, const Tensor &, c10::string_view>(
  batch_rule_266_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_267_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, const Scalar &, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_267_t,Tensor,const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view>(
  batch_rule_267_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_268_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_268_t,Tensor,const Tensor &, IntArrayRef, int64_t>(
  batch_rule_268_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_269_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_269_t,Tensor,const Tensor &, const Tensor &, c10::optional<int64_t>>(
  batch_rule_269_t batch_rule,
  const Tensor & self, const Tensor & indices, c10::optional<int64_t> dim
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
  auto results = batch_rule(self_value, self_bdim, indices_value, indices_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_270_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_270_t,Tensor,const Tensor &, int64_t, const Tensor &>(
  batch_rule_270_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_271_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_271_t,Tensor,const Tensor &, Dimname, const Tensor &>(
  batch_rule_271_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_272_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_272_t,Tensor,const Tensor &, IntArrayRef, int64_t, const Tensor &>(
  batch_rule_272_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_273_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_273_t,Tensor,const Tensor &, int64_t, const Tensor &, bool>(
  batch_rule_273_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_274_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_274_t,Tensor,const Tensor &, const Tensor &, int64_t, const Tensor &, bool>(
  batch_rule_274_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_275_t)(const Tensor &, c10::optional<int64_t>, Dimname, const Tensor &, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_275_t,Tensor,const Tensor &, Dimname, const Tensor &, bool>(
  batch_rule_275_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_276_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_276_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &>(
  batch_rule_276_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_277_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_277_t,Tensor,const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t>(
  batch_rule_277_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_278_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_278_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &>(
  batch_rule_278_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_279_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_279_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, bool, bool, bool>(
  batch_rule_279_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_280_t)(const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_280_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool, bool>(
  batch_rule_280_t batch_rule,
  const Tensor & self, bool pivot, bool check_errors
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_281_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_281_t,Tensor,const Tensor &, const Tensor &, const Tensor &, bool, bool>(
  batch_rule_281_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_282_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_282_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, bool, bool>(
  batch_rule_282_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_283_t)(const Tensor &, c10::optional<int64_t>, int64_t, bool, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_283_t,Tensor,const Tensor &, int64_t, bool, c10::optional<Generator>>(
  batch_rule_283_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_284_t)(int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_284_t,Tensor,int64_t, const Tensor &>(
  batch_rule_284_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_285_t)(const Tensor &, c10::optional<int64_t>, int64_t, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_285_t,Tensor,const Tensor &, int64_t, const Scalar &, const Scalar &>(
  batch_rule_285_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_286_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_286_t,Tensor,const Tensor &, double, c10::optional<int64_t>, bool>(
  batch_rule_286_t batch_rule,
  const Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, q, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_287_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_287_t,Tensor,const Tensor &, const Tensor &, c10::optional<int64_t>, bool>(
  batch_rule_287_t batch_rule,
  const Tensor & self, const Tensor & q, c10::optional<int64_t> dim, bool keepdim
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
  auto results = batch_rule(self_value, self_bdim, q_value, q_bdim, dim, keepdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_288_t)(const Tensor &, c10::optional<int64_t>, double, c10::optional<int64_t>, bool, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_288_t,Tensor,const Tensor &, double, c10::optional<int64_t>, bool, c10::string_view>(
  batch_rule_288_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_289_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, bool, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_289_t,Tensor,const Tensor &, const Tensor &, c10::optional<int64_t>, bool, c10::string_view>(
  batch_rule_289_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_290_t)(const Tensor &, c10::optional<int64_t>, c10::optional<bool>, int64_t, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_290_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::optional<bool>, int64_t, bool>(
  batch_rule_290_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_291_t)(const Tensor &, c10::optional<int64_t>, c10::optional<bool>, Dimname, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_291_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::optional<bool>, Dimname, bool>(
  batch_rule_291_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_292_t)(const Tensor &, c10::optional<int64_t>, int64_t, int64_t, bool, bool);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_292_t,std::tuple<Tensor,Tensor>,const Tensor &, int64_t, int64_t, bool, bool>(
  batch_rule_292_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_293_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, int64_t, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_293_t,Tensor,const Tensor &, const Scalar &, int64_t, const Scalar &>(
  batch_rule_293_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_294_t)(double, const Tensor &, c10::optional<int64_t>, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_294_t,Tensor,double, const Tensor &, c10::optional<Generator>>(
  batch_rule_294_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_295_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_295_t,Tensor,const Tensor &, const Tensor &, bool, bool>(
  batch_rule_295_t batch_rule,
  const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right
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
  auto results = batch_rule(sorted_sequence_value, sorted_sequence_bdim, self_value, self_bdim, out_int32, right);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_296_t)(const Scalar &, const Tensor &, c10::optional<int64_t>, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_296_t,Tensor,const Scalar &, const Tensor &, bool, bool>(
  batch_rule_296_t batch_rule,
  const Scalar & self, const Tensor & boundaries, bool out_int32, bool right
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_297_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_297_t,Tensor,const Tensor &, const Scalar &, bool, bool>(
  batch_rule_297_t batch_rule,
  const Tensor & sorted_sequence, const Scalar & self, bool out_int32, bool right
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor sorted_sequence_value;
  optional<int64_t> sorted_sequence_bdim;
  std::tie(sorted_sequence_value, sorted_sequence_bdim) = unwrapTensorAtLevel(sorted_sequence, cur_level);
  auto results = batch_rule(sorted_sequence_value, sorted_sequence_bdim, self, out_int32, right);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_298_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_298_t,Tensor,const Tensor &, const Tensor &, const Scalar &, const Scalar &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_298_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_299_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t);
template <>
Tensor lowerToNextLayer<batch_rule_299_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, const c10::optional<Tensor> &, int64_t>(
  batch_rule_299_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_300_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_300_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(
  batch_rule_300_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_301_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_301_t,std::tuple<Tensor,Tensor>,const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t>(
  batch_rule_301_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_302_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, int64_t, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_302_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor> &, int64_t, int64_t, const Tensor &>(
  batch_rule_302_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_303_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, int64_t, double);
template <>
Tensor lowerToNextLayer<batch_rule_303_t,Tensor,const Tensor &, const Tensor &, const Tensor &, int64_t, double>(
  batch_rule_303_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta
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
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, target_value, target_bdim, reduction, delta);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_304_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_304_t,Tensor,const Tensor &, const Scalar &, const Scalar &, const Scalar &>(
  batch_rule_304_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_305_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Scalar &, bool, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_305_t,Tensor,const Tensor &, const Scalar &, const Scalar &, const Scalar &, bool, const Tensor &>(
  batch_rule_305_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_306_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &);
template <>
Tensor lowerToNextLayer<batch_rule_306_t,Tensor,const Tensor &, const Tensor &, const Scalar &, const Scalar &>(
  batch_rule_306_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_307_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, bool);
template <>
Tensor lowerToNextLayer<batch_rule_307_t,Tensor,const Tensor &, const Tensor &, const Scalar &, bool>(
  batch_rule_307_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_308_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, bool, c10::optional<Generator>);
template <>
Tensor lowerToNextLayer<batch_rule_308_t,Tensor,const Tensor &, const Tensor &, const Scalar &, const Scalar &, bool, c10::optional<Generator>>(
  batch_rule_308_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_309_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, bool, bool);
template <>
Tensor lowerToNextLayer<batch_rule_309_t,Tensor,const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &, bool, bool>(
  batch_rule_309_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_310_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Scalar &, const Scalar &, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_310_t,Tensor,const Tensor &, const Tensor &, const Scalar &, const Scalar &, const Tensor &>(
  batch_rule_310_t batch_rule,
  const Tensor & grad_output, const Tensor & self, const Scalar & beta, const Scalar & threshold, const Tensor & output
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
  Tensor output_value;
  optional<int64_t> output_bdim;
  std::tie(output_value, output_bdim) = unwrapTensorAtLevel(output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, self_value, self_bdim, beta, threshold, output_value, output_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_311_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_311_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(
  batch_rule_311_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_312_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_312_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(
  batch_rule_312_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_313_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, const Tensor &, c10::optional<int64_t>);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_313_t,std::tuple<Tensor,Tensor>,const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(
  batch_rule_313_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_314_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_314_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(
  batch_rule_314_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_315_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &, c10::optional<int64_t>);
template <>
Tensor lowerToNextLayer<batch_rule_315_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(
  batch_rule_315_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_316_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_316_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_316_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_317_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_317_t,Tensor,const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_317_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_318_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, bool, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_318_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, bool, c10::optional<ArrayRef<double>>>(
  batch_rule_318_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_319_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, IntArrayRef, bool, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_319_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, IntArrayRef, bool, c10::optional<ArrayRef<double>>>(
  batch_rule_319_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_320_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_320_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, c10::optional<ArrayRef<double>>>(
  batch_rule_320_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_321_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_321_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<ArrayRef<double>>>(
  batch_rule_321_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_322_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_322_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<double>>(
  batch_rule_322_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_323_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, bool, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_323_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(
  batch_rule_323_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_324_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_324_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(
  batch_rule_324_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_325_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_325_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(
  batch_rule_325_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_326_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_326_t,Tensor,const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_326_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_327_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_327_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_327_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_328_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_328_t,Tensor,const Tensor &, IntArrayRef, c10::optional<double>>(
  batch_rule_328_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_329_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_329_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(
  batch_rule_329_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_330_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_330_t,Tensor,const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(
  batch_rule_330_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_331_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_331_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(
  batch_rule_331_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_332_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_332_t,Tensor,const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_332_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_333_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_333_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(
  batch_rule_333_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_334_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<double>);
template <>
Tensor lowerToNextLayer<batch_rule_334_t,Tensor,const Tensor &, const Tensor &, c10::optional<double>>(
  batch_rule_334_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_335_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_335_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_335_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_336_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_336_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef>(
  batch_rule_336_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_337_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_337_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, IntArrayRef, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef>(
  batch_rule_337_t batch_rule,
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
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_338_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, IntArrayRef, const c10::optional<Tensor> &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_338_t,Tensor,const Tensor &, const Tensor &, IntArrayRef, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_338_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_339_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_339_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_339_t batch_rule,
  const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_340_t)(const Tensor &, c10::optional<int64_t>, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template <>
Tensor lowerToNextLayer<batch_rule_340_t,Tensor,const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(
  batch_rule_340_t batch_rule,
  const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_341_t)(const Tensor &, c10::optional<int64_t>, c10::optional<int64_t>, int64_t, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_341_t,Tensor,const Tensor &, c10::optional<int64_t>, int64_t, c10::optional<c10::string_view>>(
  batch_rule_341_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_342_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_342_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, IntArrayRef, c10::optional<c10::string_view>>(
  batch_rule_342_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_343_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>, c10::optional<IntArrayRef>, c10::optional<c10::string_view>);
template <>
Tensor lowerToNextLayer<batch_rule_343_t,Tensor,const Tensor &, c10::optional<IntArrayRef>, c10::optional<IntArrayRef>, c10::optional<c10::string_view>>(
  batch_rule_343_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_344_t)(const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>);
template <>
Tensor lowerToNextLayer<batch_rule_344_t,Tensor,const Tensor &, c10::optional<IntArrayRef>>(
  batch_rule_344_t batch_rule,
  const Tensor & values, c10::optional<IntArrayRef> addends
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_345_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<double>, c10::optional<c10::string_view>);
template <>
std::tuple<Tensor,Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_345_t,std::tuple<Tensor,Tensor,Tensor,Tensor>,const Tensor &, const Tensor &, c10::optional<double>, c10::optional<c10::string_view>>(
  batch_rule_345_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_346_t)(const Tensor &, c10::optional<int64_t>, c10::string_view);
template <>
std::tuple<Tensor,Tensor> lowerToNextLayer<batch_rule_346_t,std::tuple<Tensor,Tensor>,const Tensor &, c10::string_view>(
  batch_rule_346_t batch_rule,
  const Tensor & self, c10::string_view mode
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_347_t)(const Tensor &, c10::optional<int64_t>, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_347_t,Tensor,const Tensor &, c10::string_view>(
  batch_rule_347_t batch_rule,
  const Tensor & self, c10::string_view p
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_348_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_348_t,Tensor,const Tensor &, const c10::optional<Scalar> &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>>(
  batch_rule_348_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_349_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_349_t,Tensor,const Tensor &, c10::string_view, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>>(
  batch_rule_349_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_350_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_350_t,Tensor,const Tensor &, const Scalar &, c10::optional<IntArrayRef>, bool, c10::optional<ScalarType>>(
  batch_rule_350_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_351_t)(const Tensor &, c10::optional<int64_t>, const Scalar &, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_351_t,Tensor,const Tensor &, const Scalar &, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_351_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_352_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, IntArrayRef, bool, c10::optional<ScalarType>);
template <>
Tensor lowerToNextLayer<batch_rule_352_t,Tensor,const Tensor &, c10::string_view, IntArrayRef, bool, c10::optional<ScalarType>>(
  batch_rule_352_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>,Tensor,c10::optional<int64_t>> (*batch_rule_353_t)(const Tensor &, c10::optional<int64_t>, bool);
template <>
std::tuple<Tensor,Tensor,Tensor> lowerToNextLayer<batch_rule_353_t,std::tuple<Tensor,Tensor,Tensor>,const Tensor &, bool>(
  batch_rule_353_t batch_rule,
  const Tensor & self, bool full_matrices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, full_matrices);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level), makeBatched(std::get<4>(results), std::get<5>(results), cur_level));
}

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_354_t)(const Tensor &, c10::optional<int64_t>, const c10::optional<Scalar> &);
template <>
Tensor lowerToNextLayer<batch_rule_354_t,Tensor,const Tensor &, const c10::optional<Scalar> &>(
  batch_rule_354_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_355_t)(const Tensor &, c10::optional<int64_t>, const Tensor &, c10::optional<int64_t>, c10::optional<IntArrayRef>);
template <>
Tensor lowerToNextLayer<batch_rule_355_t,Tensor,const Tensor &, const Tensor &, c10::optional<IntArrayRef>>(
  batch_rule_355_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_356_t)(const Tensor &, c10::optional<int64_t>, c10::optional<double>, bool);
template <>
Tensor lowerToNextLayer<batch_rule_356_t,Tensor,const Tensor &, c10::optional<double>, bool>(
  batch_rule_356_t batch_rule,
  const Tensor & self, c10::optional<double> tol, bool hermitian
) {
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_357_t)(const Tensor &, c10::optional<int64_t>, c10::optional<ArrayRef<double>>);
template <>
Tensor lowerToNextLayer<batch_rule_357_t,Tensor,const Tensor &, c10::optional<ArrayRef<double>>>(
  batch_rule_357_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_358_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_358_t,Tensor,const Tensor &, c10::string_view, c10::string_view>(
  batch_rule_358_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_359_t)(const Tensor &, c10::optional<int64_t>, int64_t, c10::string_view);
template <>
Tensor lowerToNextLayer<batch_rule_359_t,Tensor,const Tensor &, int64_t, c10::string_view>(
  batch_rule_359_t batch_rule,
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

typedef std::tuple<Tensor,c10::optional<int64_t>> (*batch_rule_360_t)(const Tensor &, c10::optional<int64_t>, c10::string_view, const c10::optional<Tensor> &, c10::optional<int64_t>, const c10::optional<Tensor> &, c10::optional<int64_t>, int64_t, bool, const c10::optional<Scalar> &);
template <>
Tensor lowerToNextLayer<batch_rule_360_t,Tensor,const Tensor &, c10::string_view, const c10::optional<Tensor> &, const c10::optional<Tensor> &, int64_t, bool, const c10::optional<Scalar> &>(
  batch_rule_360_t batch_rule,
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

}}
