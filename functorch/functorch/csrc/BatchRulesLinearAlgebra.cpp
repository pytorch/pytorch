#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
slogdet_batch_rule(const Tensor& self, optional<int64_t> self_bdim) {
  if (!self_bdim.has_value()) {
    auto result = at::slogdet(self);
    return {
      std::move(std::get<0>(result)), nullopt,
      std::move(std::get<1>(result)), nullopt
    };
  }

  // slogdet supports arbitrary dims at the front
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto result = at::slogdet(self_);
  return {
    std::move(std::get<0>(result)), 0,
    std::move(std::get<1>(result)), 0
  };
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("slogdet", slogdet_batch_rule);
}

}}

