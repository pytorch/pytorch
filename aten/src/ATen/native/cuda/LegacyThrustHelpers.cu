#include <ATen/ATen.h>
#include <ATen/native/cuda/SortingCommon.cuh>

#include <THC/THCThrustAllocator.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>

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
  thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, orig_data, LTOp<int64_t>());
}

template<typename index_t>
void embedding_dense_backward_cuda_scan(Tensor &sorted_indices, Tensor &count) {
  using device_ptr = thrust::device_ptr<index_t>;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto num_indices = count.numel();

  // Compute an increasing sequence per unique item in sortedIndices:
  // sorted: 2 5 5 5 7 7 8 9 9
  //  count: 1 1 2 3 1 2 1 1 2
  auto sorted_data = device_ptr(sorted_indices.data_ptr<index_t>());
  auto count_data = device_ptr(count.data_ptr<index_t>());
  thrust::inclusive_scan_by_key(
    policy,
    sorted_data,
    sorted_data + num_indices,
    thrust::make_constant_iterator(1),
    count_data
  );

  // Take the maximum of each count per unique key in reverse:
  // sorted: 2 5 5 5 7 7 8 9 9
  //  count: 1 3 3 3 2 2 1 2 2
  thrust::inclusive_scan_by_key(
    policy,
    thrust::make_reverse_iterator(sorted_data + num_indices),
    thrust::make_reverse_iterator(sorted_data),
    thrust::make_reverse_iterator(count_data + num_indices),
    thrust::make_reverse_iterator(count_data + num_indices),
    thrust::equal_to<index_t>(),
    thrust::maximum<index_t>()
  );
}

template
void embedding_dense_backward_cuda_scan<int>(Tensor &sorted_indices, Tensor &count);
template
void embedding_dense_backward_cuda_scan<int64_t>(Tensor &sorted_indices, Tensor &count);

}}
