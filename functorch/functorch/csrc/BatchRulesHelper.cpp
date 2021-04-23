#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

Tensor moveBatchDimToFront(const Tensor& tensor, optional<int64_t> maybe_batch_dim) {
  if (!maybe_batch_dim.has_value()) {
    return tensor;
  }
  return tensor.movedim(maybe_batch_dim.value(), 0);
}

int64_t rankWithoutBatchDim(const Tensor& tensor, optional<int64_t> maybe_batch_dim) {
  int64_t result = tensor.dim();
  if (maybe_batch_dim.has_value()) {
    result -= 1;
  }
  return result;
}

optional<int64_t> valIfNonempty(optional<int64_t> maybe_empty, int64_t new_val) {
  if (maybe_empty.has_value()) {
    return new_val;
  }
  return nullopt;
}

int64_t getPhysicalDim(const Tensor& tensor, bool has_batch_dim, int64_t logical_dim) {
  optional<int64_t> bdim = has_batch_dim ? optional<int64_t>(0) : nullopt;
  auto rank = rankWithoutBatchDim(tensor, bdim);
  return maybe_wrap_dim(rank, logical_dim) + 1;
}

}}
