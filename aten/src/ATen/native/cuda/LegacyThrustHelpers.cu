#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/cuda/cub_definitions.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>

namespace at::native {

template<typename index_t>
int64_t embedding_backward_cuda_kernel_unique_by_key(const Tensor &sorted_indices, Tensor &segment_offsets) {
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);
  const ptrdiff_t numel = sorted_indices.numel();
  auto sorted_indices_dev = thrust::device_ptr<const index_t>(sorted_indices.const_data_ptr<index_t>());
  auto dummy = at::empty_like(sorted_indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto dummy_dev = thrust::device_ptr<index_t>(dummy.mutable_data_ptr<index_t>());
  auto ends = thrust::unique_by_key_copy(
          policy,
          sorted_indices_dev,
          sorted_indices_dev + numel,
          thrust::make_counting_iterator(0),
          dummy_dev,
          thrust::device_ptr<index_t>(segment_offsets.mutable_data_ptr<index_t>()));
  return thrust::get<0>(ends) - dummy_dev;
}

template
int64_t embedding_backward_cuda_kernel_unique_by_key<int>(const Tensor &sorted_indices, Tensor &segment_offsets);
template
int64_t embedding_backward_cuda_kernel_unique_by_key<int64_t>(const Tensor &sorted_indices, Tensor &segment_offsets);

} // namespace at::native
