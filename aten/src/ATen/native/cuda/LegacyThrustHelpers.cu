#include <ATen/ATen.h>

#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace at { namespace native {

void index_put_with_sort_kernel_thrust_helper(Tensor &linearIndex, Tensor &orig_indices, Tensor &sorted_indices, int64_t num_indices) {
  sorted_indices.copy_(linearIndex);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  using device_ptr = thrust::device_ptr<int64_t>;

  // Fill sortedOrigIndices with sequential indices
  const auto count_iter = thrust::counting_iterator<int64_t>(0);
  auto orig_data = device_ptr(orig_indices.data_ptr<int64_t>());
  thrust::copy(policy, count_iter, count_iter + num_indices, orig_data);

  // Sort the inputs into sorted with the corresponding indices; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
  // Sort; a stable sort is not required
  // NB - not passing comparator causes thrust to use radix sort, and it hurts perf A LOT, at least for medium (few K) sized indices
  auto sorted_data = device_ptr(sorted_indices.data_ptr<int64_t>());
  thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, orig_data, ThrustLTOp<int64_t>());
}

}}
