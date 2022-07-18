
#pragma once
#include <ATen/Operators.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>

namespace at { namespace functorch {

template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _new_zeros_with_same_feature_meta_generated_plumbing(const at::Tensor & self, const at::Tensor & other, int64_t self_num_batch_dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::_new_zeros_with_same_feature_meta::call(self, other, self_num_batch_dims);
  }
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
at::Tensor abs_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::abs::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor angle_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::angle::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_as_real_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_as_real::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor view_as_complex_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view_as_complex::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sgn_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sgn::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor real_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::real::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor imag_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::imag::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _conj_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_conj::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor acos_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::acos::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor add_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::add_Tensor::call(self, other, alpha);
  }
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
at::Tensor add_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::add_Scalar::call(self, other, alpha);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & add__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::add__Scalar::call(self, other, alpha);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor acosh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::acosh::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor asinh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::asinh::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atanh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atanh::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor asin_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::asin::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atan_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::atan::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_not_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_not::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_not_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logical_not::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_xor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_xor::call(self, other);
  }
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
at::Tensor & logical_xor__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_xor_::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_and_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_and::call(self, other);
  }
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
at::Tensor & logical_and__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_and_::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logical_or_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_or::call(self, other);
  }
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
at::Tensor & logical_or__generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logical_or_::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bmm_generated_plumbing(const at::Tensor & self, const at::Tensor & mat2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mat2, cur_level)) {
    return at::_ops::bmm::call(self, mat2);
  }
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
at::Tensor ceil_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ceil::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::vector<at::Tensor> chunk_generated_plumbing(const at::Tensor & self, int64_t chunks, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::chunk::call(self, chunks, dim);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, chunks, dim);
  return makeBatchedVector(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_generated_plumbing(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp::call(self, min, max);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_max_generated_plumbing(const at::Tensor & self, const at::Scalar & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp_max::call(self, max);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, max);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_max_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clamp_max_Tensor::call(self, max);
  }
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
at::Tensor & clamp_max__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & max) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(max, cur_level)) {
    return at::_ops::clamp_max__Tensor::call(self, max);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor max_value;
  optional<int64_t> max_bdim;
  std::tie(max_value, max_bdim) = unwrapTensorAtLevel(max, cur_level);
  batch_rule(self_value, self_bdim, max_value, max_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_min_generated_plumbing(const at::Tensor & self, const at::Scalar & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clamp_min::call(self, min);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clamp_min_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level)) {
    return at::_ops::clamp_min_Tensor::call(self, min);
  }
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
at::Tensor & clamp_min__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & min) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(min, cur_level)) {
    return at::_ops::clamp_min__Tensor::call(self, min);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor min_value;
  optional<int64_t> min_bdim;
  std::tie(min_value, min_bdim) = unwrapTensorAtLevel(min, cur_level);
  batch_rule(self_value, self_bdim, min_value, min_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor polar_generated_plumbing(const at::Tensor & abs, const at::Tensor & angle) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(abs, cur_level) && !isBatchedAtLevel(angle, cur_level)) {
    return at::_ops::polar::call(abs, angle);
  }
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
at::Tensor constant_pad_nd_generated_plumbing(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::constant_pad_nd::call(self, pad, value);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, pad, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor contiguous_generated_plumbing(const at::Tensor & self, at::MemoryFormat memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::contiguous::call(self, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor convolution_generated_plumbing(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(bias, cur_level)) {
    return at::_ops::convolution::call(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  }
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
at::Tensor & copy__generated_plumbing(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::copy_::call(self, src, non_blocking);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor src_value;
  optional<int64_t> src_bdim;
  std::tie(src_value, src_bdim) = unwrapTensorAtLevel(src, cur_level);
  batch_rule(self_value, self_bdim, src_value, src_bdim, non_blocking);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cos_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cos::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cosh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cosh::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm_generated_plumbing(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(bias, cur_level) && !isBatchedAtLevel(running_mean, cur_level) && !isBatchedAtLevel(running_var, cur_level)) {
    return at::_ops::cudnn_batch_norm::call(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  }
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
at::Tensor cudnn_grid_sampler_generated_plumbing(const at::Tensor & self, const at::Tensor & grid) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(grid, cur_level)) {
    return at::_ops::cudnn_grid_sampler::call(self, grid);
  }
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
::std::tuple<at::Tensor,at::Tensor> cudnn_grid_sampler_backward_generated_plumbing(const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(grid, cur_level) && !isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::cudnn_grid_sampler_backward::call(self, grid, grad_output);
  }
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
at::Tensor diag_embed_generated_plumbing(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diag_embed::call(self, offset, dim1, dim2);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_generated_plumbing(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diagonal::call(self, offset, dim1, dim2);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_backward_generated_plumbing(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::diagonal_backward::call(grad_output, input_sizes, offset, dim1, dim2);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, offset, dim1, dim2);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor div_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::div_Tensor::call(self, other);
  }
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
at::Tensor & div__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::div__Tensor::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor div_Tensor_mode_generated_plumbing(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::div_Tensor_mode::call(self, other, rounding_mode);
  }
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
at::Tensor div_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::div_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & div__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::div__Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor div_Scalar_mode_generated_plumbing(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::div_Scalar_mode::call(self, other, rounding_mode);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, rounding_mode);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor dot_generated_plumbing(const at::Tensor & self, const at::Tensor & tensor) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(tensor, cur_level)) {
    return at::_ops::dot::call(self, tensor);
  }
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
at::Tensor embedding_generated_plumbing(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::embedding::call(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  }
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
at::Tensor embedding_dense_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::_ops::embedding_dense_backward::call(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
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
at::Tensor new_empty_generated_plumbing(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_empty::call(self, size, dtype, layout, device, pin_memory);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_full_generated_plumbing(const at::Tensor & self, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_full::call(self, size, fill_value, dtype, layout, device, pin_memory);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, fill_value, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_zeros_generated_plumbing(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_zeros::call(self, size, dtype, layout, device, pin_memory);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor new_ones_generated_plumbing(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::new_ones::call(self, size, dtype, layout, device, pin_memory);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, dtype, layout, device, pin_memory);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor empty_like_generated_plumbing(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::empty_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor erf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::erf::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor erfc_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::erfc::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor exp_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exp::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor exp2_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::exp2::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor expm1_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::expm1::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor expand_generated_plumbing(const at::Tensor & self, at::IntArrayRef size, bool implicit) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::expand::call(self, size, implicit);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, implicit);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor floor_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::floor::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor floor_divide_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::floor_divide::call(self, other);
  }
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
at::Tensor floor_divide_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::floor_divide_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor frac_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::frac::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor full_like_generated_plumbing(const at::Tensor & self, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::full_like::call(self, fill_value, dtype, layout, device, pin_memory, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, fill_value, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gcd_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::gcd::call(self, other);
  }
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
at::Tensor lcm_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::lcm::call(self, other);
  }
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
at::Tensor grid_sampler_2d_generated_plumbing(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(grid, cur_level)) {
    return at::_ops::grid_sampler_2d::call(input, grid, interpolation_mode, padding_mode, align_corners);
  }
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
::std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(grid, cur_level)) {
    return at::_ops::grid_sampler_2d_backward::call(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
  }
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
at::Tensor grid_sampler_3d_generated_plumbing(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(grid, cur_level)) {
    return at::_ops::grid_sampler_3d::call(input, grid, interpolation_mode, padding_mode, align_corners);
  }
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
::std::tuple<at::Tensor,at::Tensor> grid_sampler_3d_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(grid, cur_level)) {
    return at::_ops::grid_sampler_3d_backward::call(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
  }
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
at::Tensor inverse_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::inverse::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isnan_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isnan::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isreal_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isreal::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_generated_plumbing(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(bias, cur_level)) {
    return at::_ops::native_layer_norm::call(input, normalized_shape, weight, bias, eps);
  }
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
at::Tensor nan_to_num_generated_plumbing(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::nan_to_num::call(self, nan, posinf, neginf);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, nan, posinf, neginf);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log10_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log10::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log1p_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log1p::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor log2_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log2::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logaddexp_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logaddexp::call(self, other);
  }
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
at::Tensor logaddexp2_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::logaddexp2::call(self, other);
  }
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
at::Tensor xlogy_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::xlogy_Tensor::call(self, other);
  }
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
at::Tensor xlogy_Scalar_Other_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::xlogy_Scalar_Other::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _log_softmax_backward_data_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::_log_softmax_backward_data::call(grad_output, output, dim, input_dtype);
  }
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
at::Tensor matrix_exp_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::matrix_exp::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> aminmax_generated_plumbing(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::aminmax::call(self, dim, keepdim);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, keepdim);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm_generated_plumbing(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(bias, cur_level) && !isBatchedAtLevel(running_mean, cur_level) && !isBatchedAtLevel(running_var, cur_level)) {
    return at::_ops::miopen_batch_norm::call(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  }
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
at::Tensor mm_generated_plumbing(const at::Tensor & self, const at::Tensor & mat2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mat2, cur_level)) {
    return at::_ops::mm::call(self, mat2);
  }
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
at::Tensor mul_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::mul_Tensor::call(self, other);
  }
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
at::Tensor & mul__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::mul__Tensor::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mul_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mul_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & mul__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mul__Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor mv_generated_plumbing(const at::Tensor & self, const at::Tensor & vec) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(vec, cur_level)) {
    return at::_ops::mv::call(self, vec);
  }
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
at::Tensor mvlgamma_generated_plumbing(const at::Tensor & self, int64_t p) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mvlgamma::call(self, p);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, p);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_generated_plumbing(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(weight, cur_level) && !isBatchedAtLevel(bias, cur_level) && !isBatchedAtLevel(running_mean, cur_level) && !isBatchedAtLevel(running_var, cur_level)) {
    return at::_ops::native_batch_norm::call(input, weight, bias, running_mean, running_var, training, momentum, eps);
  }
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
at::Tensor ones_like_generated_plumbing(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ones_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _cdist_forward_generated_plumbing(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(x1, cur_level) && !isBatchedAtLevel(x2, cur_level)) {
    return at::_ops::_cdist_forward::call(x1, x2, p, compute_mode);
  }
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
at::Tensor _cdist_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(x1, cur_level) && !isBatchedAtLevel(x2, cur_level) && !isBatchedAtLevel(cdist, cur_level)) {
    return at::_ops::_cdist_backward::call(grad, x1, x2, p, cdist);
  }
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
at::Tensor permute_generated_plumbing(const at::Tensor & self, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::permute::call(self, dims);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor movedim_intlist_generated_plumbing(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::movedim_intlist::call(self, source, destination);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, source, destination);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pixel_shuffle_generated_plumbing(const at::Tensor & self, int64_t upscale_factor) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::pixel_shuffle::call(self, upscale_factor);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upscale_factor);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pixel_unshuffle_generated_plumbing(const at::Tensor & self, int64_t downscale_factor) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::pixel_unshuffle::call(self, downscale_factor);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, downscale_factor);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pinverse_generated_plumbing(const at::Tensor & self, double rcond) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::pinverse::call(self, rcond);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, rcond);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rad2deg_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rad2deg::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor deg2rad_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::deg2rad::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rand_like_generated_plumbing(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rand_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor randn_like_generated_plumbing(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::randn_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor reciprocal_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::reciprocal::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor neg_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::neg::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor repeat_generated_plumbing(const at::Tensor & self, at::IntArrayRef repeats) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::repeat::call(self, repeats);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, repeats);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _reshape_alias_generated_plumbing(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_reshape_alias::call(self, size, stride);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor round_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::round::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor round_decimals_generated_plumbing(const at::Tensor & self, int64_t decimals) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::round_decimals::call(self, decimals);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, decimals);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor relu_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::relu::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor relu6_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::relu6::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor prelu_generated_plumbing(const at::Tensor & self, const at::Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::prelu::call(self, weight);
  }
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
::std::tuple<at::Tensor,at::Tensor> prelu_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(weight, cur_level)) {
    return at::_ops::prelu_backward::call(grad_output, self, weight);
  }
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
at::Tensor gelu_generated_plumbing(const at::Tensor & self, c10::string_view approximate) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gelu::call(self, approximate);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, approximate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gelu_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gelu_backward::call(grad_output, self, approximate);
  }
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
at::Tensor hardshrink_generated_plumbing(const at::Tensor & self, const at::Scalar & lambd) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardshrink::call(self, lambd);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, lambd);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hardshrink_backward_generated_plumbing(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_out, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardshrink_backward::call(grad_out, self, lambd);
  }
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
at::Tensor rsqrt_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rsqrt::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor select_int_generated_plumbing(const at::Tensor & self, int64_t dim, int64_t index) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::select_int::call(self, dim, index);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor select_backward_generated_plumbing(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t index) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::select_backward::call(grad_output, input_sizes, dim, index);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, dim, index);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor selu_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::selu::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor celu_generated_plumbing(const at::Tensor & self, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::celu::call(self, alpha);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor silu_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::silu::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor silu_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::silu_backward::call(grad_output, self);
  }
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
at::Tensor mish_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::mish::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sigmoid_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sigmoid::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logit_generated_plumbing(const at::Tensor & self, c10::optional<double> eps) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logit::call(self, eps);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, eps);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sin_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sin::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sinc_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sinc::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sinh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sinh::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor detach_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::detach::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor slice_Tensor_generated_plumbing(const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::slice_Tensor::call(self, dim, start, end, step);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor slice_backward_generated_plumbing(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::slice_backward::call(grad_output, input_sizes, dim, start, end, step);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_sizes, dim, start, end, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diagonal_scatter_generated_plumbing(const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::diagonal_scatter::call(self, src, offset, dim1, dim2);
  }
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
at::Tensor _softmax_backward_data_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::_softmax_backward_data::call(grad_output, output, dim, input_dtype);
  }
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
at::Tensor squeeze_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor squeeze_dim_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::squeeze_dim::call(self, dim);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sqrt_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sqrt::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tan_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tan::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tanh_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tanh::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor threshold_generated_plumbing(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::threshold::call(self, threshold, value);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, threshold, value);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor threshold_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::threshold_backward::call(grad_output, self, threshold);
  }
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
at::Tensor transpose_int_generated_plumbing(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::transpose_int::call(self, dim0, dim1);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim0, dim1);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor flip_generated_plumbing(const at::Tensor & self, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::flip::call(self, dims);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor roll_generated_plumbing(const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::roll::call(self, shifts, dims);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, shifts, dims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor trunc_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::trunc::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _unsafe_view_generated_plumbing(const at::Tensor & self, at::IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_unsafe_view::call(self, size);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor unsqueeze_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unsqueeze::call(self, dim);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor where_self_generated_plumbing(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(condition, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::where_self::call(condition, self, other);
  }
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
at::Tensor zeros_like_generated_plumbing(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::zeros_like::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor clone_generated_plumbing(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::clone::call(self, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor positive_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::positive::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sub_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::sub_Tensor::call(self, other, alpha);
  }
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
at::Tensor & sub__Tensor_generated_plumbing(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::sub__Tensor::call(self, other, alpha);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor other_value;
  optional<int64_t> other_bdim;
  std::tie(other_value, other_bdim) = unwrapTensorAtLevel(other, cur_level);
  batch_rule(self_value, self_bdim, other_value, other_bdim, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sub_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sub_Scalar::call(self, other, alpha);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor & sub__Scalar_generated_plumbing(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sub__Scalar::call(self, other, alpha);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  batch_rule(self_value, self_bdim, other, alpha);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rsub_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::rsub_Tensor::call(self, other, alpha);
  }
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
at::Tensor heaviside_generated_plumbing(const at::Tensor & self, const at::Tensor & values) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_ops::heaviside::call(self, values);
  }
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
at::Tensor rsub_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::rsub_Scalar::call(self, other, alpha);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other, alpha);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _to_copy_generated_plumbing(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_to_copy::call(self, dtype, layout, device, pin_memory, non_blocking, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, non_blocking, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_dtype_layout_generated_plumbing(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_dtype_layout::call(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_device_generated_plumbing(const at::Tensor & self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_device::call(self, device, dtype, non_blocking, copy, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, device, dtype, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_dtype_generated_plumbing(const at::Tensor & self, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::to_dtype::call(self, dtype, non_blocking, copy, memory_format);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dtype, non_blocking, copy, memory_format);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor to_other_generated_plumbing(const at::Tensor & self, const at::Tensor & other, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::to_other::call(self, other, non_blocking, copy, memory_format);
  }
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
at::Tensor & masked_fill__Scalar_generated_plumbing(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_fill__Scalar::call(self, mask, value);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor mask_value;
  optional<int64_t> mask_bdim;
  std::tie(mask_value, mask_bdim) = unwrapTensorAtLevel(mask, cur_level);
  batch_rule(self_value, self_bdim, mask_value, mask_bdim, value);
  return self;
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor masked_fill_Scalar_generated_plumbing(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_fill_Scalar::call(self, mask, value);
  }
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
at::Tensor view_generated_plumbing(const at::Tensor & self, at::IntArrayRef size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::view::call(self, size);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor index_add_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(source, cur_level)) {
    return at::_ops::index_add::call(self, dim, index, source, alpha);
  }
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
at::Tensor scatter_src_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_src::call(self, dim, index, src);
  }
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
at::Tensor scatter_value_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::scatter_value::call(self, dim, index, value);
  }
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
at::Tensor scatter_reduce_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_reduce::call(self, dim, index, src, reduce);
  }
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
at::Tensor scatter_value_reduce_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::scatter_value_reduce::call(self, dim, index, value, reduce);
  }
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
at::Tensor scatter_add_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level) && !isBatchedAtLevel(src, cur_level)) {
    return at::_ops::scatter_add::call(self, dim, index, src);
  }
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
at::Tensor bitwise_and_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_and_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor bitwise_and_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_and_Tensor::call(self, other);
  }
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
at::Tensor bitwise_or_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_or_Tensor::call(self, other);
  }
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
at::Tensor bitwise_xor_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_xor_Tensor::call(self, other);
  }
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
at::Tensor __lshift___Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__lshift___Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __lshift___Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__lshift___Tensor::call(self, other);
  }
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
at::Tensor bitwise_left_shift_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_left_shift_Tensor::call(self, other);
  }
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
at::Tensor bitwise_left_shift_Tensor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_left_shift_Tensor_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __rshift___Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::__rshift___Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor __rshift___Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::__rshift___Tensor::call(self, other);
  }
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
at::Tensor bitwise_right_shift_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::bitwise_right_shift_Tensor::call(self, other);
  }
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
at::Tensor bitwise_right_shift_Tensor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::bitwise_right_shift_Tensor_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor diag_generated_plumbing(const at::Tensor & self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::diag::call(self, diagonal);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cross_generated_plumbing(const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::cross::call(self, other, dim);
  }
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
at::Tensor triu_generated_plumbing(const at::Tensor & self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::triu::call(self, diagonal);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor tril_generated_plumbing(const at::Tensor & self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::tril::call(self, diagonal);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, diagonal);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ne_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ne_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ne_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ne_Tensor::call(self, other);
  }
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
at::Tensor eq_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::eq_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor eq_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::eq_Tensor::call(self, other);
  }
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
at::Tensor ge_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::ge_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor ge_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::ge_Tensor::call(self, other);
  }
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
at::Tensor le_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::le_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor le_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::le_Tensor::call(self, other);
  }
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
at::Tensor gt_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::gt_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor gt_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::gt_Tensor::call(self, other);
  }
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
at::Tensor lt_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::lt_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lt_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::lt_Tensor::call(self, other);
  }
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
at::Tensor masked_select_generated_plumbing(const at::Tensor & self, const at::Tensor & mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_select::call(self, mask);
  }
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
at::Tensor masked_select_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & mask) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(mask, cur_level)) {
    return at::_ops::masked_select_backward::call(grad, input, mask);
  }
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
at::Tensor gather_generated_plumbing(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::gather::call(self, dim, index, sparse_grad);
  }
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
at::Tensor gather_backward_generated_plumbing(const at::Tensor & grad, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(index, cur_level)) {
    return at::_ops::gather_backward::call(grad, self, dim, index, sparse_grad);
  }
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
void _linalg_check_errors_generated_plumbing(const at::Tensor & info, c10::string_view api_name, bool is_matrix) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(info, cur_level)) {
    return at::_ops::_linalg_check_errors::call(info, api_name, is_matrix);
  }
  Tensor info_value;
  optional<int64_t> info_bdim;
  std::tie(info_value, info_bdim) = unwrapTensorAtLevel(info, cur_level);
  batch_rule(info_value, info_bdim, api_name, is_matrix);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cholesky_generated_plumbing(const at::Tensor & self, bool upper) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cholesky::call(self, upper);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upper);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor cholesky_inverse_generated_plumbing(const at::Tensor & self, bool upper) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::cholesky_inverse::call(self, upper);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, upper);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor lgamma_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::lgamma::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor digamma_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::digamma::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor erfinv_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::erfinv::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor i0_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::i0::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sign_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::sign::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor signbit_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::signbit::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor atan2_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::atan2::call(self, other);
  }
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
at::Tensor fmod_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::fmod_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor fmod_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::fmod_Tensor::call(self, other);
  }
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
at::Tensor hypot_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::hypot::call(self, other);
  }
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
at::Tensor igamma_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::igamma::call(self, other);
  }
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
at::Tensor igammac_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::igammac::call(self, other);
  }
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
at::Tensor nextafter_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::nextafter::call(self, other);
  }
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
at::Tensor remainder_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::remainder_Scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor remainder_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::remainder_Tensor::call(self, other);
  }
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
at::Tensor fmin_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::fmin::call(self, other);
  }
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
at::Tensor fmax_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::fmax::call(self, other);
  }
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
at::Tensor maximum_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::maximum::call(self, other);
  }
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
at::Tensor minimum_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::minimum::call(self, other);
  }
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
at::Tensor unfold_generated_plumbing(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::unfold::call(self, dimension, size, step);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dimension, size, step);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor pow_Tensor_Tensor_generated_plumbing(const at::Tensor & self, const at::Tensor & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(exponent, cur_level)) {
    return at::_ops::pow_Tensor_Tensor::call(self, exponent);
  }
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
at::Tensor pow_Tensor_Scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & exponent) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::pow_Tensor_Scalar::call(self, exponent);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, exponent);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor alias_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::alias::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor searchsorted_Tensor_generated_plumbing(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(sorted_sequence, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(sorter, cur_level)) {
    return at::_ops::searchsorted_Tensor::call(sorted_sequence, self, out_int32, right, side, sorter);
  }
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
at::Tensor mse_loss_generated_plumbing(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(target, cur_level)) {
    return at::_ops::mse_loss::call(self, target, reduction);
  }
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
at::Tensor mse_loss_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(target, cur_level)) {
    return at::_ops::mse_loss_backward::call(grad_output, self, target, reduction);
  }
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
at::Tensor elu_generated_plumbing(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::elu::call(self, alpha, scale, input_scale);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, alpha, scale, input_scale);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor glu_generated_plumbing(const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::glu::call(self, dim);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, dim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor glu_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::glu_backward::call(grad_output, self, dim);
  }
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
at::Tensor hardsigmoid_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardsigmoid::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hardsigmoid_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardsigmoid_backward::call(grad_output, self);
  }
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
at::Tensor hardtanh_generated_plumbing(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardtanh::call(self, min_val, max_val);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, min_val, max_val);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hardtanh_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardtanh_backward::call(grad_output, self, min_val, max_val);
  }
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
at::Tensor hardswish_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardswish::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor hardswish_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::hardswish_backward::call(grad_output, self);
  }
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
at::Tensor leaky_relu_generated_plumbing(const at::Tensor & self, const at::Scalar & negative_slope) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::leaky_relu::call(self, negative_slope);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, negative_slope);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor leaky_relu_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::leaky_relu_backward::call(grad_output, self, negative_slope, self_is_result);
  }
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
at::Tensor log_sigmoid_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::log_sigmoid::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor rrelu_with_noise_generated_plumbing(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(noise, cur_level)) {
    return at::_ops::rrelu_with_noise::call(self, noise, lower, upper, training, generator);
  }
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
at::Tensor softplus_generated_plumbing(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::softplus::call(self, beta, threshold);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, beta, threshold);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor softshrink_generated_plumbing(const at::Tensor & self, const at::Scalar & lambd) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::softshrink::call(self, lambd);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, lambd);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor softshrink_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::softshrink_backward::call(grad_output, self, lambd);
  }
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
at::Tensor _adaptive_avg_pool2d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_adaptive_avg_pool2d::call(self, output_size);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor _adaptive_avg_pool3d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::_adaptive_avg_pool3d::call(self, output_size);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor avg_pool2d_generated_plumbing(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::avg_pool2d::call(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor avg_pool3d_generated_plumbing(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::avg_pool3d::call(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
::std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices_generated_plumbing(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::max_pool2d_with_indices::call(self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, stride, padding, dilation, ceil_mode);
  return std::make_tuple(makeBatched(std::get<0>(results), std::get<1>(results), cur_level), makeBatched(std::get<2>(results), std::get<3>(results), cur_level));
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor reflection_pad1d_generated_plumbing(const at::Tensor & self, at::IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::reflection_pad1d::call(self, padding);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor reflection_pad2d_generated_plumbing(const at::Tensor & self, at::IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::reflection_pad2d::call(self, padding);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor reflection_pad3d_generated_plumbing(const at::Tensor & self, at::IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::reflection_pad3d::call(self, padding);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor replication_pad1d_generated_plumbing(const at::Tensor & self, at::IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::replication_pad1d::call(self, padding);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor replication_pad2d_generated_plumbing(const at::Tensor & self, at::IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::replication_pad2d::call(self, padding);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor replication_pad3d_generated_plumbing(const at::Tensor & self, at::IntArrayRef padding) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::replication_pad3d::call(self, padding);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, padding);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_linear1d_vec_generated_plumbing(const at::Tensor & input, at::OptionalIntArrayRef output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::upsample_linear1d_vec::call(input, output_size, align_corners, scale_factors);
  }
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_linear1d_backward_vec_generated_plumbing(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::upsample_linear1d_backward_vec::call(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_bilinear2d_vec_generated_plumbing(const at::Tensor & input, at::OptionalIntArrayRef output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::upsample_bilinear2d_vec::call(input, output_size, align_corners, scale_factors);
  }
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_bilinear2d_backward_vec_generated_plumbing(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::upsample_bilinear2d_backward_vec::call(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_trilinear3d_vec_generated_plumbing(const at::Tensor & input, at::OptionalIntArrayRef output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::upsample_trilinear3d_vec::call(input, output_size, align_corners, scale_factors);
  }
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_trilinear3d_backward_vec_generated_plumbing(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::upsample_trilinear3d_backward_vec::call(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_bicubic2d_vec_generated_plumbing(const at::Tensor & input, at::OptionalIntArrayRef output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::upsample_bicubic2d_vec::call(input, output_size, align_corners, scale_factors);
  }
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_bicubic2d_backward_vec_generated_plumbing(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::upsample_bicubic2d_backward_vec::call(grad_output, output_size, input_size, align_corners, scale_factors);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, align_corners, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest1d_vec_generated_plumbing(const at::Tensor & input, at::OptionalIntArrayRef output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::upsample_nearest1d_vec::call(input, output_size, scale_factors);
  }
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest1d_backward_vec_generated_plumbing(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::upsample_nearest1d_backward_vec::call(grad_output, output_size, input_size, scale_factors);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest2d_vec_generated_plumbing(const at::Tensor & input, at::OptionalIntArrayRef output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::upsample_nearest2d_vec::call(input, output_size, scale_factors);
  }
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest2d_backward_vec_generated_plumbing(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::upsample_nearest2d_backward_vec::call(grad_output, output_size, input_size, scale_factors);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest3d_vec_generated_plumbing(const at::Tensor & input, at::OptionalIntArrayRef output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level)) {
    return at::_ops::upsample_nearest3d_vec::call(input, output_size, scale_factors);
  }
  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  auto results = batch_rule(input_value, input_bdim, output_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest3d_backward_vec_generated_plumbing(const at::Tensor & grad_output, at::OptionalIntArrayRef output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::upsample_nearest3d_backward_vec::call(grad_output, output_size, input_size, scale_factors);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, output_size, input_size, scale_factors);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_linear1d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::upsample_linear1d::call(self, output_size, align_corners, scales);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, align_corners, scales);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_bilinear2d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::upsample_bilinear2d::call(self, output_size, align_corners, scales_h, scales_w);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, align_corners, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_bicubic2d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::upsample_bicubic2d::call(self, output_size, align_corners, scales_h, scales_w);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, align_corners, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_trilinear3d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::upsample_trilinear3d::call(self, output_size, align_corners, scales_d, scales_h, scales_w);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, align_corners, scales_d, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest1d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::upsample_nearest1d::call(self, output_size, scales);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, scales);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest2d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::upsample_nearest2d::call(self, output_size, scales_h, scales_w);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor upsample_nearest3d_generated_plumbing(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::upsample_nearest3d::call(self, output_size, scales_d, scales_h, scales_w);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, output_size, scales_d, scales_h, scales_w);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor sigmoid_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & output) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::sigmoid_backward::call(grad_output, output);
  }
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
at::Tensor logit_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logit_backward::call(grad_output, self, eps);
  }
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
at::Tensor tanh_backward_generated_plumbing(const at::Tensor & grad_output, const at::Tensor & output) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level) && !isBatchedAtLevel(output, cur_level)) {
    return at::_ops::tanh_backward::call(grad_output, output);
  }
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
at::Tensor im2col_generated_plumbing(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::im2col::call(self, kernel_size, dilation, padding, stride);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, kernel_size, dilation, padding, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor im2col_backward_generated_plumbing(const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(grad_output, cur_level)) {
    return at::_ops::im2col_backward::call(grad_output, input_size, kernel_size, dilation, padding, stride);
  }
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  auto results = batch_rule(grad_output_value, grad_output_bdim, input_size, kernel_size, dilation, padding, stride);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isfinite_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isfinite::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isinf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isinf::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isposinf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isposinf::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor isneginf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::isneginf::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_entr_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_entr::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_ndtri_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_ndtri::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_expm1_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_expm1::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_exp2_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_exp2::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_psi_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_psi::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_digamma_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_digamma::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_gammaln_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_gammaln::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_erf_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_erf::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_erfc_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_erfc::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_erfcx_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_erfcx::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_erfinv_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_erfinv::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_ndtr_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_ndtr::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_xlog1py_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::special_xlog1py::call(self, other);
  }
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
at::Tensor special_xlog1py_other_scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_xlog1py_other_scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_xlogy_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::special_xlogy::call(self, other);
  }
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
at::Tensor special_xlogy_other_scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_xlogy_other_scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_zeta_generated_plumbing(const at::Tensor & self, const at::Tensor & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(other, cur_level)) {
    return at::_ops::special_zeta::call(self, other);
  }
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
at::Tensor special_zeta_other_scalar_generated_plumbing(const at::Tensor & self, const at::Scalar & other) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_zeta_other_scalar::call(self, other);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, other);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_i0_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_i0::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_i0e_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_i0e::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_i1_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_i1::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_i1e_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_i1e::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_expit_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_expit::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_sinc_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_sinc::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_round_generated_plumbing(const at::Tensor & self, int64_t decimals) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_round::call(self, decimals);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, decimals);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor special_log1p_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::special_log1p::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor logdet_generated_plumbing(const at::Tensor & self) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::logdet::call(self);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor linalg_householder_product_generated_plumbing(const at::Tensor & input, const at::Tensor & tau) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(input, cur_level) && !isBatchedAtLevel(tau, cur_level)) {
    return at::_ops::linalg_householder_product::call(input, tau);
  }
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
at::Tensor linalg_pinv_generated_plumbing(const at::Tensor & self, double rcond, bool hermitian) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::linalg_pinv::call(self, rcond, hermitian);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, rcond, hermitian);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}
template <typename batch_rule_t, batch_rule_t batch_rule>
at::Tensor expand_copy_generated_plumbing(const at::Tensor & self, at::IntArrayRef size, bool implicit) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level)) {
    return at::_ops::expand_copy::call(self, size, implicit);
  }
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  auto results = batch_rule(self_value, self_bdim, size, implicit);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

}} // namespace at::functorch
