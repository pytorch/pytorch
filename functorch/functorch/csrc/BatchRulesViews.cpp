#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

std::tuple<Tensor, optional<int64_t>> flatten_batch_rule(
    const Tensor& self, 
    optional<int64_t> self_bdim,
    int64_t start_dim, int64_t end_dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  start_dim = getPhysicalDim(self_, self_bdim.has_value(), start_dim);
  end_dim = getPhysicalDim(self_, self_bdim.has_value(), end_dim);
  return { at::flatten(self_, start_dim, end_dim), valIfNonempty(self_bdim, 0) };
}

std::tuple<Tensor,optional<int64_t>> unsqueeze_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim); 
  auto rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, rank + 1) + 1;
  return { self.unsqueeze(dim), valIfNonempty(self_bdim, 0) };
}

TORCH_LIBRARY_IMPL(aten, BatchedOutOfTree, m) {
#define VMAP_SUPPORT(op, batch_rule) \
  m.impl(op, PrimBatchRule7< \
      decltype(&batch_rule), &batch_rule, to_operator_t<decltype(batch_rule)> \
      >::apply);

  VMAP_SUPPORT("flatten.using_ints", flatten_batch_rule);
  VMAP_SUPPORT("unsqueeze", unsqueeze_batch_rule);

#undef VMAP_SUPPORT
}

}}
