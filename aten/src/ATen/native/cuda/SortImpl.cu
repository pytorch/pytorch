#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace at::native {

std::vector<int64_t> infer_dense_strides_dim_last(const Tensor & self, int64_t dim) {
  int64_t ndim = self.dim();
  // sort the strides in descending order according to its value,
  // keeping dim the last.
  std::vector<int64_t> strides = self.strides().vec();
  strides[dim] = -1;
  std::vector<int64_t> original_dim(ndim);
  for (int64_t i = 0; i < ndim; i++) {
    original_dim[i] = i;
  }
  thrust::stable_sort_by_key(
    thrust::host, strides.data(), strides.data() + ndim, original_dim.data(),
    thrust::greater<int64_t>()
  );
  // generate contiguous strides on permuted dims
  std::vector<int64_t> new_strides(ndim);
  std::vector<int64_t> new_strides_unsort(ndim);
  int64_t cumprod = 1;
  for (int64_t i = 0; i < ndim; i++) {
    new_strides[ndim - 1 - i] = cumprod;
    cumprod *= self.sizes()[original_dim[ndim - 1 - i]];
  }
  // unsort new strides
  for (int64_t i = 0; i < ndim; i++) {
    new_strides_unsort[original_dim[i]] = new_strides[i];
  }
  return new_strides_unsort;
}

} // namespace at::native
