#pragma once

#include <functorch/csrc/PlumbingHelper.h>

namespace at { namespace functorch {

template <typename F, F BatchRule, typename... ExtraArgs>
static Tensor& inplacePlumbing1(Tensor& self, ExtraArgs... extra_args) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  BatchRule(self_value, self_bdim, std::forward<ExtraArgs>(extra_args)...);
  return self;
}

template <typename BR, BR BatchRule, typename... ExtraArgs>
Tensor& inplacePlumbing2(Tensor& self, const Tensor& other, ExtraArgs... extra_args) {
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
  BatchRule(
      self_value, self_bdim, other_value, other_bdim,
      std::forward<ExtraArgs>(extra_args)...);
  return self;
}

}}
