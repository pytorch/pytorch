#include <ATen/BatchingRules.h>
#include <ATen/ATen.h>

namespace at {

std::pair<Tensor,BatchDims> sum_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef dims, bool keepdim, c10::optional<ScalarType> dtype) {
  // NB: We don't really need to move the batch dims to the front.
  // One alternative way to do this is to keep them where they are and compute
  // the required `dims` to reduce over. However, assuming that the batch
  // dims are at front greatly simplifies the `dims` calculation and moving
  // them there is relatively cheap.
  auto self_ = moveBatchDimsToFront(self, self_bdims);
  auto result_bdims = moveBatchDimsToFront(self_bdims);
  auto tensor_dims = self_.dim() - self_bdims.size();

  // Real dims to reduce over
  std::vector<int64_t> actual_dims;
  actual_dims.reserve(dims.size());
  for (int64_t dim : dims) {
    dim = maybe_wrap_dim(dim, tensor_dims);
    actual_dims.push_back(dim + self_bdims.size());
  }

  auto result = at::sum(self_, actual_dims, keepdim, dtype);
  return { result, result_bdims };
}

}

