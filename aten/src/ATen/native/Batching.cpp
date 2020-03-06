#include <ATen/Batching.h>

namespace at { namespace native {

Tensor _make_batched(const Tensor& self, optional<int64_t> batch_dim, int64_t level) {
  return makeBatched(self, batch_dim, level);
}

Tensor _unwrap_batched(const Tensor& self, int64_t level) {
  // TODO: it probably matters if the level is different
  return unwrapBatched(self);
}

int64_t _batch_dim(const Tensor& self) {
  if (!isBatched(self)) {
    return -1;
  }
  auto* batch_tensor = getBatched(self);
  if (batch_tensor->batch_dim_) {
    return *batch_tensor->batch_dim_;
  }
  return -1;
}

bool _is_batched(const Tensor& self) {
  return isBatched(self);
}

}}
