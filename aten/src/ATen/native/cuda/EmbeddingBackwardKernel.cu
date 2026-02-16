#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/EmbeddingBackwardKernel.cuh>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/OpMathType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/cuda/SortingCommon.cuh>

#include <c10/macros/Macros.h>

#include <thrust/iterator/counting_iterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

namespace {

/* This code computes the sum of the weights in two-steps:
  1) Each GPU warp sums `NROWS_PER_THREAD` number of row given by `indices`
  2) Each partial-sum from 1) are summed and scatter into `grad_weight`

  Notice, `NROWS_PER_THREAD` impacts the Achieved Occupancy of the
  kernel execution. If it is high, the size of the thread blocks will be
  too small to achieve good occupancy. Similarly, a very low value will
  make the size of the thread blocks in the final sum in step 2) too small.
*/
constexpr int NROWS_PER_THREAD = 10;

// If the number of blocks processed by each SM is larger than this value,
// we will use the two-step sum and scatter approach.
constexpr int32_t MAX_ATOMIC_ACCUM_BLOCKS_PER_SM = 4;

// Fast ceil division (no overflow checking)
__host__ __device__ __forceinline__
int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename index_t>
__global__
void krn_partials_per_segment(index_t *ret, const index_t *segment_offsets,
                              const int64_t *num_of_segments_ptr, int64_t numel) {
  int64_t num_of_segments = *num_of_segments_ptr;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    const int64_t idx_start = segment_offsets[id];
    const int64_t idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];
    const int64_t size = idx_end - idx_start;
    ret[id] = ceil_div(size, NROWS_PER_THREAD);
  }
}

template <typename index_t>
__global__
void krn_partial_segment_offset(
        index_t *ret,
        const index_t *partials_per_segment,
        const index_t *partials_per_segment_offset,
        const index_t *segment_offsets,
        const int64_t *num_of_segments_ptr) {
  int64_t num_of_segments = *num_of_segments_ptr;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    index_t idx = partials_per_segment_offset[id];
    const index_t num_partials = partials_per_segment[id];
    const index_t segment_offset = segment_offsets[id];
    for (int64_t i=0; i<num_partials; ++i) {
      ret[idx++] = segment_offset + i * NROWS_PER_THREAD;
    }
  }
}


template <typename scalar_t, typename index_t>
__global__ void compute_grad_weight_bags(
    const index_t *indices, const scalar_t *gradOutput,
    const index_t *offset2bag, const index_t *count, ptrdiff_t numel,
    int64_t stride, int mode_mean, const index_t *bag_size,
    const scalar_t* per_sample_weights, int64_t per_sample_weights_stride,
    const index_t* segment_offsets, const int64_t *num_of_segments_ptr,
    acc_type<scalar_t, true> *grad_weight_per_segment,
    const int64_t stride_warped) {

  int64_t num_of_segments = *num_of_segments_ptr;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t id = gid / stride_warped;
  const int64_t startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_begin = segment_offsets[id];
  const int idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];

  acc_type<scalar_t, true> weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    const int origRow = indices[idx];
    const int seq_number = offset2bag[origRow];
    const int gradOutputRow = seq_number * stride;

    acc_type<scalar_t, true> scale = count ? 1.0 / count[idx] : 1.0;
    if (per_sample_weights) {
      scale *= per_sample_weights[origRow * per_sample_weights_stride];
    }

    acc_type<scalar_t, true> gradient = gradOutput[gradOutputRow + startFeature];
    if (mode_mean) {
      gradient /= bag_size[seq_number];
    }
    weight += gradient * scale;
  }
  grad_weight_per_segment[id * stride + startFeature] = weight;
}

template <typename scalar_t, typename index_t>
__global__ void compute_grad_weight(
    const index_t *indices,
    const scalar_t *gradOutput,
    const index_t *count,
    ptrdiff_t numel,
    int64_t stride,
    const index_t* segment_offsets,
    const int64_t *num_of_segments_ptr,
    acc_type<scalar_t, true> *grad_weight_per_segment,
    const int64_t stride_warped) {

  int64_t num_of_segments = *num_of_segments_ptr;
  using accscalar_t = acc_type<scalar_t, true>;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t id = gid / stride_warped;
  const int64_t startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_begin = segment_offsets[id];
  const int idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];

  accscalar_t weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    const index_t target_row = indices[idx];
    const accscalar_t scale = count ? (accscalar_t)1.0 / count[idx] : 1.0;
    weight += gradOutput[target_row * stride + startFeature] * scale;
  }
  grad_weight_per_segment[id * stride + startFeature] = weight;
}

// Fused kernel that combines compute_grad_weight and sum_and_scatter using atomic adds.
// This eliminates the serialization bottleneck when num_of_segments is small.
// Each partial segment atomically adds its contribution directly to grad_weight.
// Template parameters:
//   scalar_t: input gradient type
//   output_t: output gradient weight type (float for half/bf16 atomic accumulation)
template <typename scalar_t, typename output_t, typename index_t>
__global__ void compute_grad_weight_atomic_accumulate(
    const index_t *orig_indices,
    const scalar_t *gradOutput,
    const index_t *count,
    ptrdiff_t numel,
    int64_t stride,
    const index_t *partial_segment_offsets,
    const int64_t *num_of_partial_segments_ptr,
    const index_t *sorted_indices,
    const index_t *partial_to_segment_idx,
    const index_t *segment_offsets,
    output_t *grad_weight,
    const int64_t padding_idx,
    const int64_t stride_warped) {

  int64_t num_of_partial_segments = *num_of_partial_segments_ptr;
  using accscalar_t = acc_type<scalar_t, true>;
  const int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t partial_id = gid / stride_warped;
  const int32_t startFeature = gid % stride_warped;

  if (startFeature >= stride) {
    return;
  }
  if (partial_id >= num_of_partial_segments) {
    return;
  }

  const index_t idx_begin = partial_segment_offsets[partial_id];
  const index_t idx_end = (partial_id == num_of_partial_segments - 1)
      ? numel
      : partial_segment_offsets[partial_id + 1];

  accscalar_t weight = 0;
  for (index_t idx = idx_begin; idx < idx_end; ++idx) {
    const index_t target_row = orig_indices[idx];
    const accscalar_t scale = count ? (accscalar_t)1.0 / count[idx] : 1.0;
    weight += gradOutput[target_row * stride + startFeature] * scale;
  }

  // Get the target row for this partial segment from the segment info
  const index_t segment_id = partial_to_segment_idx[partial_id];
  const index_t target_row = sorted_indices[segment_offsets[segment_id]];

  if (target_row != padding_idx) {
    gpuAtomicAddNoReturn(
        &grad_weight[target_row * stride + startFeature],
        static_cast<output_t>(weight));
  }
}

// Kernel to build mapping from partial segment ID to segment ID
template <typename index_t>
__global__ void krn_partial_to_segment_idx(
    index_t *partial_to_segment_idx,
    const index_t *partials_per_segment,
    const index_t *partials_per_segment_offset,
    const int64_t *num_of_segments_ptr) {
  int64_t num_of_segments = *num_of_segments_ptr;
  const int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num_of_segments) {
    index_t idx = partials_per_segment_offset[id];
    const index_t num_partials = partials_per_segment[id];
    for (index_t i = 0; i < num_partials; ++i) {
      partial_to_segment_idx[idx++] = id;
    }
  }
}

// This kernel assumes that all input tensors are contiguous.
template <typename scalar_t, typename index_t>
__global__ void sum_and_scatter(
    const index_t *input, scalar_t *gradWeight, int64_t stride,
    const index_t* segment_offsets, const int64_t *num_of_segments_ptr,
    const acc_type<scalar_t, true> *grad_weight_per_segment,
    const index_t *segment_sizes_offsets, const int64_t *num_of_partial_segments_ptr,
    const int64_t padding_idx,
    const int64_t stride_warped) {

  int64_t num_of_segments = *num_of_segments_ptr;
  int64_t num_of_partial_segments = *num_of_partial_segments_ptr;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t id = gid / stride_warped;
  const int64_t startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }

  const int idx_begin = segment_sizes_offsets[id];
  const int idx_end = (id == num_of_segments-1)?num_of_partial_segments:segment_sizes_offsets[id+1];
  acc_type<scalar_t, true> weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    weight += grad_weight_per_segment[idx*stride + startFeature];
  }
  int64_t target_row = input[segment_offsets[id]];
  if (target_row != padding_idx) {
    gradWeight[target_row * stride + startFeature] = weight;
  }
}

template<typename index_t>
__global__ void compute_num_of_partial_segments(const index_t *partials_per_segment, const index_t *partials_per_segment_offset, const int64_t *num_of_segments_ptr, int64_t *output) {
  int64_t num_of_segments = *num_of_segments_ptr;
  *output = partials_per_segment[num_of_segments-1] +
            partials_per_segment_offset[num_of_segments-1];
}


} // anon namespace


Tensor embedding_backward_cuda_kernel(
        const Tensor &grad,
        const Tensor &orig_indices,
        const Tensor &sorted_indices,
        const Tensor &count,
        int64_t num_weights,
        int padding_idx,
        bool mode_mean,
        const Tensor &offset2bag,
        const Tensor &bag_size,
        const Tensor &per_sample_weights) {

  auto stream = at::cuda::getCurrentCUDAStream();
  const ptrdiff_t numel = sorted_indices.numel();

  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());
  const int64_t stride = grad_weight.stride(0);

  // Compute the number of segments and their start position so that we do not have to
  // spawn a warp per index. In this context, a segment is a number of rows that should
  // be summarized.
  // Unit: index in `sorted_indices` and `orig_indices`
  auto segment_offsets = at::empty({numel}, orig_indices.options());
  auto num_of_segments_tensor = at::empty({}, grad.options().dtype(kLong));
  int64_t *num_of_segments_ptr = num_of_segments_tensor.mutable_data_ptr<int64_t>();
  AT_DISPATCH_INDEX_TYPES(orig_indices.scalar_type(), "embedding_backward_cuda_kernel", [&] () {
    cuda::cub::unique_by_key(
      sorted_indices.const_data_ptr<index_t>(), thrust::make_counting_iterator(0),
      segment_offsets.mutable_data_ptr<index_t>(),
      num_of_segments_ptr, sorted_indices.numel());
  });

  int64_t max_segments = std::min<int64_t>(numel, num_weights);

  AT_DISPATCH_INDEX_TYPES(orig_indices.scalar_type(), "embedding_backward_cuda_kernel", [&] () {
    // We split the segments up into sizes of `NROWS_PER_THREAD`
    // Compute the number partial-segments per segment (some partial-segments
    // may not be the full `NROWS_PER_THREAD` number of rows)
    auto partials_per_segment = at::empty({max_segments}, orig_indices.options());
    {
      krn_partials_per_segment<<<ceil_div(max_segments, 32), 32, 0, stream>>> (
              partials_per_segment.mutable_data_ptr<index_t>(),
              segment_offsets.const_data_ptr<index_t>(),
              num_of_segments_ptr,
              numel);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // In order to compute `partial_segment_offset`, which is the start index
    // of each partial-segment in `sorted_indices`, we need to compute the
    // start position of each _segment_ in `partial_segment_offset`.
    // Unit: index in `partial_segment_offset`
    auto partials_per_segment_offset = at::empty({max_segments}, orig_indices.options());
    cuda::cub::exclusive_sum(
        partials_per_segment.const_data_ptr<index_t>(),
        partials_per_segment_offset.mutable_data_ptr<index_t>(),
        max_segments);

    // The total number of partial-segments is the sum of `partials_per_segment_offset`
    auto num_of_partial_segments_tensor = at::empty({}, grad.options().dtype(kLong));
    int64_t *num_of_partial_segments_ptr = num_of_partial_segments_tensor.mutable_data_ptr<int64_t>();
    compute_num_of_partial_segments<index_t><<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
      partials_per_segment.const_data_ptr<index_t>(),
      partials_per_segment_offset.const_data_ptr<index_t>(),
      num_of_segments_ptr, num_of_partial_segments_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto max_partial_segment = numel / NROWS_PER_THREAD + max_segments;

    // Now we can compute the start position of each partial-segment
    // Unit: index in `sorted_indices` and `orig_indices`
    auto partial_segment_offset = at::empty({max_partial_segment}, orig_indices.options());
    {
      krn_partial_segment_offset<<<ceil_div(max_segments, 32), 32, 0, stream>>> (
              partial_segment_offset.mutable_data_ptr<index_t>(),
              partials_per_segment.const_data_ptr<index_t>(),
              partials_per_segment_offset.const_data_ptr<index_t>(),
              segment_offsets.const_data_ptr<index_t>(),
              num_of_segments_ptr);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int warp_size = at::cuda::warp_size();
    const int stride_warped = ceil_div(stride, warp_size)*warp_size;
    const int block = std::min(stride_warped, MAX_BLOCK_SIZE);
    const int grid = ceil_div(max_partial_segment*stride_warped, block);

    // Heuristic: Use fused kernel when sum_and_scatter would have poor parallelism
    // This happens when max_segments * stride_warped is small relative to GPU capacity
    // The fused kernel uses atomic adds but has grid size proportional to max_partial_segment
    // instead of max_segments, giving much better parallelism when segments are few but large
    // Note: Fused kernel uses atomics which are non-deterministic, so we skip it
    // when deterministic algorithms are requested
    const int32_t sum_scatter_grid = ceil_div(max_segments * stride_warped, block);
    const int32_t num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int32_t min_blocks_for_good_occupancy = num_sms * MAX_ATOMIC_ACCUM_BLOCKS_PER_SM;
    const bool use_fused_kernel = !offset2bag.defined() &&
        sum_scatter_grid < min_blocks_for_good_occupancy &&
        max_partial_segment > sum_scatter_grid * 4 &&
        !at::globalContext().deterministicAlgorithms();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      grad.scalar_type(), "embedding_bag_backward_cuda_compute_grad_weight", [&] {
        // For numerical stability, the dtype of `grad_weight_per_segment`
        // should match `acc_type`
        using partial_weight_t = acc_type<scalar_t, true>;

        if (use_fused_kernel) {
          // This eliminates the serialization bottleneck in sum_and_scatter
          // by directly accumulating into grad_weight from all partial segments

          // Build mapping from partial segment ID to segment ID
          auto partial_to_segment_idx = at::empty({max_partial_segment}, orig_indices.options());
          {
            krn_partial_to_segment_idx<<<ceil_div(max_segments, 32), 32, 0, stream>>>(
                partial_to_segment_idx.mutable_data_ptr<index_t>(),
                partials_per_segment.const_data_ptr<index_t>(),
                partials_per_segment_offset.const_data_ptr<index_t>(),
                num_of_segments_ptr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }

          // For half/bfloat16 types, use opmath_type (float) intermediate buffer for atomic adds
          // (float atomics are faster and more precise than half/bf16 atomics)
          // For float/double, accumulate directly into grad_weight
          if constexpr (!std::is_same_v<scalar_t, opmath_type<scalar_t>>) {
            auto grad_weight_acc = at::zeros(
                {num_weights, stride},
                grad.options().dtype(toOpMathType(grad.scalar_type())));
            compute_grad_weight_atomic_accumulate<scalar_t, partial_weight_t><<<grid, block, 0, stream>>>(
                orig_indices.const_data_ptr<index_t>(),
                grad.const_data_ptr<scalar_t>(),
                count.defined() ? count.const_data_ptr<index_t>() : nullptr,
                numel, stride,
                partial_segment_offset.const_data_ptr<index_t>(),
                num_of_partial_segments_ptr,
                sorted_indices.const_data_ptr<index_t>(),
                partial_to_segment_idx.const_data_ptr<index_t>(),
                segment_offsets.const_data_ptr<index_t>(),
                grad_weight_acc.mutable_data_ptr<partial_weight_t>(),
                padding_idx,
                stride_warped);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            // Convert back to original dtype
            grad_weight.copy_(grad_weight_acc);
          } else {
            // For float/double, accumulate directly into grad_weight
            compute_grad_weight_atomic_accumulate<scalar_t, scalar_t><<<grid, block, 0, stream>>>(
                orig_indices.const_data_ptr<index_t>(),
                grad.const_data_ptr<scalar_t>(),
                count.defined() ? count.const_data_ptr<index_t>() : nullptr,
                numel, stride,
                partial_segment_offset.const_data_ptr<index_t>(),
                num_of_partial_segments_ptr,
                sorted_indices.const_data_ptr<index_t>(),
                partial_to_segment_idx.const_data_ptr<index_t>(),
                segment_offsets.const_data_ptr<index_t>(),
                grad_weight.mutable_data_ptr<scalar_t>(),
                padding_idx,
                stride_warped);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        } else {

          // Two-pass path: compute_grad_weight + sum_and_scatter
          auto grad_weight_per_segment = at::empty(
              {max_partial_segment, stride},
              grad.options().dtype(toOpMathType(grad.scalar_type())));
          // Compute the sum of each partial-segment and handle bags
          if (offset2bag.defined()) {
                compute_grad_weight_bags<scalar_t><<<grid, block, 0, stream>>>(
                  orig_indices.const_data_ptr<index_t>(),
                  grad.const_data_ptr<scalar_t>(),
                  offset2bag.const_data_ptr<index_t>(),
                  count.defined() ? count.const_data_ptr<index_t>() : nullptr, numel, stride,
                  mode_mean, bag_size.const_data_ptr<index_t>(),
                  per_sample_weights.defined() ? per_sample_weights.const_data_ptr<scalar_t>() : NULL,
                  per_sample_weights.defined() ? per_sample_weights.stride(0) : 0,
                  partial_segment_offset.const_data_ptr<index_t>(),
                  num_of_partial_segments_ptr, grad_weight_per_segment.mutable_data_ptr<partial_weight_t>(),
                  stride_warped);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
                compute_grad_weight<scalar_t><<<grid, block, 0, stream>>>(
                  orig_indices.const_data_ptr<index_t>(),
                  grad.const_data_ptr<scalar_t>(),
                  count.defined() ? count.const_data_ptr<index_t>() : nullptr,
                  numel, stride,
                  partial_segment_offset.const_data_ptr<index_t>(),
                  num_of_partial_segments_ptr,
                  grad_weight_per_segment.mutable_data_ptr<partial_weight_t>(),
                  stride_warped);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
          }

          // Finally, we sum all the partial-sums and scatter them
          // into `grad_weight`.
          const int grid2 = ceil_div(max_segments*stride_warped, block);
              sum_and_scatter<scalar_t><<<grid2, block, 0, stream>>>(
                sorted_indices.const_data_ptr<index_t>(),
                grad_weight.mutable_data_ptr<scalar_t>(),
                stride,
                segment_offsets.const_data_ptr<index_t>(),
                num_of_segments_ptr, grad_weight_per_segment.const_data_ptr<partial_weight_t>(),
                partials_per_segment_offset.const_data_ptr<index_t>(),
                num_of_partial_segments_ptr,
                padding_idx,
                stride_warped);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
  });
  return grad_weight;
}

} // namespace at::native
