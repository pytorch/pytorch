#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

// [start, start + 1, ..., stop - 1]
static VmapDimVector range(int64_t start, int64_t stop) {
  TORCH_INTERNAL_ASSERT(stop > start);
  VmapDimVector dims;
  dims.reserve(stop - start);
  for (int64_t i = start; i < stop; i++) {
    dims.emplace_back(i);
  }
  return dims;
}

std::tuple<Tensor,optional<int64_t>> sum_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<ScalarType> dtype) {
  if (!self_bdim.has_value()) {
    return { self.sum(dtype), nullopt };
  }
  auto self_dim = self.dim();
  if (self_dim == 1) {
    return { self.clone(), 0 };
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto dims = range(1, self_dim);
  auto result = at::sum(self_, dims, /*keepdim*/false, dtype);
  return { result, 0 };
}

std::tuple<Tensor,optional<int64_t>> mean_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, optional<ScalarType> dtype) {
  if (!self_bdim.has_value()) {
    return { self.sum(dtype), nullopt };
  }
  auto self_dim = self.dim();
  if (self_dim == 1) {
    return { self.clone(), 0 };
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto dims = range(1, self_dim);
  auto result = at::mean(self_, dims, /*keepdim*/false, dtype);
  return { result, 0 };
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("sum", sum_batch_rule);
  VMAP_SUPPORT("mean", mean_batch_rule);
}

}}
