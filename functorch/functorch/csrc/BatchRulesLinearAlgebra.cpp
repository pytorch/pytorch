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

std::tuple<Tensor, optional<int64_t>> dot_batch_rule(const Tensor& A, optional<int64_t> A_bdim, const Tensor& B, optional<int64_t> B_bdim) {
  auto A_ = moveBatchDimToFront(A, A_bdim);
  auto B_ = moveBatchDimToFront(B, B_bdim);
  if (A_bdim && B_bdim) {
    return {at::matmul(A_.unsqueeze(-2), B_.unsqueeze(-1)).squeeze(-1).squeeze(-1), 0};
  } else if (!A_bdim && !B_bdim) {
    return {at::dot(A_, B_), nullopt};
  } else {
    return {at::matmul(A_, B_.t()), 0};
  }
}


TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("slogdet", slogdet_batch_rule);
  VMAP_SUPPORT("dot", dot_batch_rule);
}
}}

