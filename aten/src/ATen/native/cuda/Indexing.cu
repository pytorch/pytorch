#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/Resize.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/DeviceUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_assert_async.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_backward_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/scatter_reduce_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.h>
#if defined(USE_ROCM)
#include <ATen/cuda/cub.cuh>
#endif
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>
#include <c10/core/QScheme.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>

#include <limits>
#include <type_traits>

#include <c10/macros/Macros.h>

namespace {
constexpr uint64_t getDefaultMaxThreadsPerBlock() {
#ifndef USE_ROCM
  return 128;
#else
  // bigger default
  return 512;
#endif
}

#ifdef USE_ROCM
#define SKIP_SORTED_INDICES 32
template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_many_indices(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  extern __shared__ unsigned char smem[];
  auto smem_dups_cache = reinterpret_cast<int64_t*>(smem);

  int smem_offset = threadIdx.y * C10_WARP_SIZE;

  int laneIdx = threadIdx.x % C10_WARP_SIZE;

  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z) {
    // Init duplicates every time we compute a new set of entries:
    smem_dups_cache[smem_offset + laneIdx] = 0;
    WARP_SYNC();

    int64_t base_idx = blockIdx.x * blockDim.y * C10_WARP_SIZE + threadIdx.y * C10_WARP_SIZE;
    int64_t idx = base_idx + laneIdx;

    if (idx < numel) {
      int64_t crnt_sorted_idx = sorted_indices[idx];

      if (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]) {
        // Determine the number of duplicates in advance:
        int64_t num_duplicates = 1;

        // Lookahead in case there is a large number of duplicates. Once that is done, handle the tail.
        while ((idx + num_duplicates + SKIP_SORTED_INDICES - 1) < numel) {
          if (sorted_indices[idx + num_duplicates + SKIP_SORTED_INDICES - 1] != crnt_sorted_idx) break;
            num_duplicates += SKIP_SORTED_INDICES;
        }
        while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
          num_duplicates++;
        }

        smem_dups_cache[smem_offset + laneIdx] = num_duplicates;
      }
    }

    WARP_SYNC();

    // All lanes in the warp are still active here. Use them all to reduce duplicates when
    // large number of duplicates are present:
    for (int subwarp = 0; subwarp < C10_WARP_SIZE; subwarp++) {
      // All lanes read the shared memory entry for number of duplicates
      int64_t new_num_duplicates = smem_dups_cache[smem_offset + subwarp];

      // Check if the original sub-warp had duplicates to eliminate, if not skip.
      if (new_num_duplicates == 0)
        continue;

      // There are duplicates that need eliminating:
      int64_t new_idx = base_idx + subwarp;
      int64_t new_crnt_sorted_idx = sorted_indices[new_idx];
      const int64_t new_weight_row = new_crnt_sorted_idx * stride + z * stride_before;

      if (!accumulate) {
        const int64_t grad_row = ((int64_t)indices[new_idx + new_num_duplicates - 1]) * stride + z * numel * stride;
        int64_t feature_dim = blockIdx.y * blockDim.x + threadIdx.x;
        while (feature_dim < stride) {
          grad_weight[new_weight_row + feature_dim] = grad_output[grad_row + feature_dim];
          feature_dim += gridDim.y * blockDim.x;
        }
        continue;
      }

      for (int dup = 0; dup < new_num_duplicates; dup++) {
        const int64_t grad_row = ((int64_t) indices[new_idx + dup]) * stride + z * numel * stride;

        // All lanes do the same thing up to here.
        int64_t feature_dim = blockIdx.y * blockDim.x + threadIdx.x;

        // Each lane has a different feature_dim.
        while (feature_dim < stride) {
          grad_weight[new_weight_row + feature_dim] += grad_output[grad_row + feature_dim];
          feature_dim += gridDim.y * blockDim.x;
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void indexing_backward_kernel_stride_1(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  int laneIdx = threadIdx.x % C10_WARP_SIZE;

  const opmath_t scale = (opmath_t)1.0;
  int64_t grad_row = 0;

  extern __shared__ unsigned char smem[];
  auto smem_dups_cache = reinterpret_cast<int64_t*>(smem);

  // Each warp gets a different section of the share memory allocation:
  int smem_offset = threadIdx.y * C10_WARP_SIZE;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z) {
    // Init duplicates every time we compute a new set of entries:
    smem_dups_cache[smem_offset + laneIdx] = 0;

    int64_t base_idx = blockIdx.x * blockDim.y * C10_WARP_SIZE + threadIdx.y * C10_WARP_SIZE;
    int64_t idx = base_idx + laneIdx;

    // Each lane calculates the number of duplicates:
    if (idx < numel) {
      int64_t crnt_sorted_idx = sorted_indices[idx];

      if (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]) {
        // Determine the number of duplicates in advance:
        int64_t num_duplicates = 1;

        // Lookahead in case there is a large number of duplicates. Once that is done, handle the tail.
        while ((idx + num_duplicates + SKIP_SORTED_INDICES - 1) < numel) {
          if (sorted_indices[idx + num_duplicates + SKIP_SORTED_INDICES - 1] != crnt_sorted_idx) break;
            num_duplicates += SKIP_SORTED_INDICES;
        }
        while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
          num_duplicates++;
        }

        if (!accumulate) {
          const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
          grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
          grad_weight[weight_row] =
            static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row]) * scale);
          continue;
        }

        // Each lane sequentially handles the duplicate elimination:
        if (num_duplicates < C10_WARP_SIZE) {
          opmath_t gradient = (opmath_t)0.0;
          const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
          for (int64_t i = 0; i < num_duplicates; ++i) {
            grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
          }

          grad_weight[weight_row] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row]) + gradient);
        } else {
          // Add duplicate to the cache:
          smem_dups_cache[smem_offset + laneIdx] = num_duplicates;
        }
      }
    }

    WARP_SYNC();

    // All lanes in the warp are still active here. Use them all to reduce duplicates when
    // large number of duplicates are present:
    for (int subwarp = 0; subwarp < C10_WARP_SIZE; subwarp++) {
      // All lanes read the shared memory entry for number of duplicates
      int64_t new_num_duplicates = smem_dups_cache[smem_offset + subwarp];

      // Check if the original sub-warp had duplicates to eliminate, if not skip.
      if (new_num_duplicates == 0)
        continue;

      // There are duplicates that need eliminating:
      int64_t new_idx = base_idx + subwarp;
      int64_t new_crnt_sorted_idx = sorted_indices[new_idx];
      const int64_t new_weight_row = new_crnt_sorted_idx * stride + z * stride_before;

      // Result of the reduction will be in this variable:
      opmath_t gradient = (opmath_t)0.0;

      int64_t num_warp_passes = new_num_duplicates / C10_WARP_SIZE;
      // Parallel reduction across the array of duplicates using all the lanes in the warp:
      for (int64_t i = 0; i < num_warp_passes; ++i) {
        grad_row = ((int64_t) indices[new_idx + i * C10_WARP_SIZE + laneIdx]) * stride + z * numel * stride;
        gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
      }

      // Reduce across the lanes of the warp:
      WARP_SYNC();
      for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
        gradient += WARP_SHFL_DOWN(gradient, offset);
      }

      if (laneIdx == 0) {
        for (int64_t i = num_warp_passes * C10_WARP_SIZE; i < new_num_duplicates; ++i) {
          grad_row = ((int64_t) indices[new_idx + i]) * stride + z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }

        grad_weight[new_weight_row] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[new_weight_row]) + gradient);
      }
    }
  }
}
#endif

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
//numel is total number of flattened indices, not expanded to dimensions that are not indexed.
//stride is the cumulative size of the not-indexed last dimensions
//stride_before is the stride of the dimension immediately preceding first indexed dimension
//if indexing starts from the 0th dimension, stride_before does not matter because blockIdx.z will be 0 in this case
//outer_dim is number of elements in the first unindexed dimensions
  using opmath_t = at::opmath_type<scalar_t>;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same destination index as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        // if not accumulate, we only keep the last duplicate index so skip those before it
        if (!accumulate && (idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
              if (accumulate) {
                weight[ii] = static_cast<opmath_t>(grad_weight[weight_row + feature_dim]);
              }
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            if (accumulate) {
              weight[ii] += gradient[ii] * scale;
            } else {
              weight[ii] = gradient[ii] * scale;
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}

#ifndef USE_ROCM
template <typename scalar_t>
__global__ void indexing_backward_kernel_stride_1(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t crnt_sorted_idx = sorted_indices[idx];

    if ((idx < numel) &&
        (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]))
    {
      // Determine the number of duplicates in advance
      int64_t num_duplicates = 1;
      while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
        num_duplicates++;
      }

      // Continue computing weights
      const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
      int64_t grad_row = 0;
      const opmath_t scale = (opmath_t)1.0;

      if (!accumulate) {
        grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
        grad_weight[weight_row] =
          static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row]) * scale);
      } else {
        opmath_t gradient = (opmath_t)0.0;

        int laneIdx = threadIdx.x % C10_WARP_SIZE;
        int64_t num_warp_passes = num_duplicates / C10_WARP_SIZE;
        for (int64_t i = 0; i < num_warp_passes; ++i) {
            grad_row = ((int64_t) indices[idx + i * C10_WARP_SIZE + laneIdx]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }
        WARP_SYNC();
        for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
          gradient += WARP_SHFL_DOWN(gradient, offset);
        }

        if (laneIdx == 0) {
          for (int64_t i = num_warp_passes * C10_WARP_SIZE; i < num_duplicates; ++i) {
            grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
          }

          grad_weight[weight_row] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row]) + gradient);
        }
      }
    }
  }
}
#endif

template <typename scalar_t>
__global__ void indexing_backward_kernel_small_stride(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t tidx = threadIdx.x;
    int64_t crnt_sorted_idx = sorted_indices[idx];

    if ((idx < numel) &&
        (tidx < stride) &&
        (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]))
    {
      // Determine the number of duplicates in advance
      int64_t num_duplicates = 1;
      while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
        num_duplicates++;
      }

      // Continue computing weights
      const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
      int64_t grad_row = 0;
      const opmath_t scale = (opmath_t)1.0;

      if (!accumulate) {
        grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
        grad_weight[weight_row + tidx] =
          static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale);
      } else {
        opmath_t gradient = (opmath_t)0.0;
        for (int64_t i = 0; i < num_duplicates; ++i) {
          grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale;
        }

        grad_weight[weight_row + tidx] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row + tidx]) + gradient);
      }
    }
  }
}

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_quantized(
  const int64_t* sorted_indices, const int64_t* indices, const float* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim,
  float inv_scale, int zero_point, int64_t qmin, int64_t qmax) {

  // This implementation is adopted from indexing_backward_kernel above.
  using opmath_t = at::opmath_type<float>;
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        // we only keep the last duplicate index so skip those before it
        if ((idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            weight[ii] = gradient[ii] * scale;
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                // we do quantization here
                int64_t qvalue = static_cast<int64_t>(zero_point + nearbyintf(weight[ii]* inv_scale));
                qvalue = min(max(qvalue, qmin), qmax);
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(qvalue);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}


}


namespace at::native {

namespace {

#if defined(USE_ROCM)
constexpr int64_t INDEX_SELECT_BACKWARD_CHUNK_SIZE = 512;
constexpr int64_t INDEX_SELECT_BACKWARD_LONG_RUN_SIZE = 8192;
constexpr int64_t INDEX_SELECT_BACKWARD_MIN_SORT_SIZE = 1000000;
constexpr int64_t INDEX_SELECT_BACKWARD_MAX_SCRATCH_BYTES = 64 * 1024 * 1024;

// Returns the first segment offset greater than value. Fixed chunk boundaries
// use this device-side upper_bound to find the duplicate run that owns them.
template <typename offset_t>
__device__ __forceinline__ int64_t index_select_backward_upper_bound(
    const offset_t* offsets,
    int64_t count,
    int64_t value) {
  int64_t first = 0;
  while (count > 0) {
    const int64_t step = count / 2;
    const int64_t current = first + step;
    if (static_cast<int64_t>(offsets[current]) <= value) {
      first = current + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

// Classifies runs that need two-pass reduction and records their fixed-size
// chunk counts. Short and unused segments write zero so an exclusive scan can
// turn this array into compact long-run scratch offsets.
__global__ void index_select_backward_long_chunk_counts_kernel(
    const int32_t* segment_offsets,
    const int64_t* num_segments_ptr,
    int32_t* long_chunk_counts,
    int64_t num_indices,
    int64_t max_segments) {
  const int64_t num_segments = *num_segments_ptr;
  for (int64_t segment = blockIdx.x * blockDim.x + threadIdx.x;
       segment < max_segments;
       segment += blockDim.x * gridDim.x) {
    int32_t count = 0;
    if (segment < num_segments) {
      const int64_t begin = static_cast<int64_t>(segment_offsets[segment]);
      const int64_t end = segment + 1 < num_segments
          ? static_cast<int64_t>(segment_offsets[segment + 1])
          : num_indices;
      const int64_t length = end - begin;
      if (length >= INDEX_SELECT_BACKWARD_LONG_RUN_SIZE) {
        count = static_cast<int32_t>(
            (length + INDEX_SELECT_BACKWARD_CHUNK_SIZE - 1) /
            INDEX_SELECT_BACKWARD_CHUNK_SIZE);
      }
    }
    long_chunk_counts[segment] = count;
  }
}

// Generic compact reducer for supported scalar types and feature widths. One
// block resolves a segment-start or fixed-boundary candidate, skips long runs,
// and directly stores unique chunks or atomically combines split chunks.
template <
    typename scalar_t,
    typename acc_t,
    typename index_t,
    int values_per_thread>
__global__ void index_select_backward_compact_kernel(
    const scalar_t* grad,
    const index_t* sorted_indices,
    const int32_t* sorted_positions,
    const int32_t* segment_offsets,
    const int64_t* num_segments_ptr,
    const int32_t* long_chunk_counts,
    acc_t* grad_input,
    int64_t num_indices,
    int64_t num_rows,
    int64_t outer_size,
    int64_t inner_size,
    int64_t max_segments) {
  static_assert(values_per_thread == 1 || values_per_thread == 2);
  __shared__ int64_t chunk_begin;
  __shared__ int64_t chunk_end;
  __shared__ int64_t output_row;
  __shared__ bool single_chunk;

  const int64_t num_boundaries =
      (num_indices - 1) / INDEX_SELECT_BACKWARD_CHUNK_SIZE;
  const int64_t candidates_per_outer = max_segments + num_boundaries;
  const int64_t max_work = outer_size * candidates_per_outer;
  const int64_t num_segments = *num_segments_ptr;

  for (int64_t work = blockIdx.x; work < max_work; work += gridDim.x) {
    const int64_t candidate = work % candidates_per_outer;
    const int64_t outer = work / candidates_per_outer;
    if (candidate < max_segments && candidate >= num_segments) {
      continue;
    }

    if (threadIdx.x == 0) {
      int64_t begin;
      int64_t segment;
      int64_t run_end = num_indices;
      const bool starts_run = candidate < max_segments;
      if (starts_run) {
        segment = candidate;
        begin = static_cast<int64_t>(segment_offsets[segment]);
        if (segment + 1 < num_segments) {
          run_end = static_cast<int64_t>(segment_offsets[segment + 1]);
        }
      } else {
        begin =
            (candidate - max_segments + 1) * INDEX_SELECT_BACKWARD_CHUNK_SIZE;
        if (sorted_indices[begin] != sorted_indices[begin - 1]) {
          begin = -1;
          segment = -1;
        } else {
          const int64_t next_segment = index_select_backward_upper_bound(
              segment_offsets, num_segments, begin);
          segment = next_segment - 1;
          if (next_segment < num_segments) {
            run_end = static_cast<int64_t>(segment_offsets[next_segment]);
          }
        }
      }

      if (begin >= 0 && long_chunk_counts != nullptr &&
          long_chunk_counts[segment] != 0) {
        begin = -1;
      }
      if (begin >= 0) {
        const index_t row = sorted_indices[begin];
        const int64_t boundary_end =
            ((begin / INDEX_SELECT_BACKWARD_CHUNK_SIZE) + 1) *
            INDEX_SELECT_BACKWARD_CHUNK_SIZE;
        const int64_t end = boundary_end < run_end ? boundary_end : run_end;
        chunk_begin = begin;
        chunk_end = end;
        output_row = static_cast<int64_t>(row);
        single_chunk = starts_run && end == run_end;
      } else {
        chunk_begin = -1;
      }
    }
    __syncthreads();

    if (chunk_begin >= 0) {
      if constexpr (values_per_thread == 2) {
        static_assert(std::is_same_v<scalar_t, c10::BFloat16>);
        static_assert(std::is_same_v<acc_t, c10::BFloat16>);
        const int64_t feature = threadIdx.x * 2;
        acc_t sum0 = acc_t(0);
        acc_t sum1 = acc_t(0);
        for (int64_t current = chunk_begin; current < chunk_end; ++current) {
          const int64_t source_row =
              static_cast<int64_t>(sorted_positions[current]);
          const int64_t source_offset =
              (outer * num_indices + source_row) * inner_size;
          const auto values = memory::load_vector<2>(
              grad + source_offset, static_cast<uint32_t>(threadIdx.x));
          sum0 += static_cast<acc_t>(values.val[0]);
          sum1 += static_cast<acc_t>(values.val[1]);
        }
        const int64_t output_offset =
            (outer * num_rows + output_row) * inner_size + feature;
        if (single_chunk) {
          grad_input[output_offset] = sum0;
          grad_input[output_offset + 1] = sum1;
        } else {
          gpuAtomicAddNoReturn(grad_input + output_offset, sum0);
          gpuAtomicAddNoReturn(grad_input + output_offset + 1, sum1);
        }
      } else {
        for (int64_t feature = threadIdx.x; feature < inner_size;
             feature += blockDim.x) {
          acc_t sum = acc_t(0);
          for (int64_t current = chunk_begin; current < chunk_end; ++current) {
            const int64_t source_row =
                static_cast<int64_t>(sorted_positions[current]);
            const int64_t source_offset =
                (outer * num_indices + source_row) * inner_size + feature;
            sum += static_cast<acc_t>(grad[source_offset]);
          }
          const int64_t output_offset =
              (outer * num_rows + output_row) * inner_size + feature;
          if (single_chunk) {
            grad_input[output_offset] = sum;
          } else {
            gpuAtomicAddNoReturn(grad_input + output_offset, sum);
          }
        }
      }
    }
    __syncthreads();
  }
}

// Broadcasts 64-bit metadata from logical lane zero by shuffling its two
// 32-bit halves. The explicit width isolates subgroups packed into one wave64.
__device__ __forceinline__ int64_t index_select_backward_wave_broadcast_int64(
    int64_t value,
    int width) {
  union Bits {
    int64_t value;
    uint32_t words[2];
  } bits = {.value = value};
  bits.words[0] = WARP_SHFL(bits.words[0], 0, width);
  bits.words[1] = WARP_SHFL(bits.words[1], 0, width);
  return bits.value;
}

// Describes whether a compact candidate is skipped, uniquely owns its output
// row, or contributes one of multiple chunks through an atomic add.
enum class IndexSelectBackwardCompactMode : uint32_t {
  Invalid = 0,
  Direct = 1,
  Atomic = 2,
};

// Atomically adds two adjacent BF16 values with the gfx942/gfx950 packed
// instruction. Callers provide a 4-byte-aligned pair; unsupported targets use
// two scalar atomics as a correctness fallback.
__device__ __forceinline__ void index_select_backward_atomic_add_bfloat16_pair(
    c10::BFloat16* output,
    c10::BFloat16 value0,
    c10::BFloat16 value1) {
  using packed_t = short __attribute__((ext_vector_type(2)));
  union PackedBFloat16 {
    c10::BFloat16 values[2];
    packed_t packed;
  } value = {};
  value.values[0] = value0;
  value.values[1] = value1;
  if (__builtin_amdgcn_is_invocable(
          __builtin_amdgcn_flat_atomic_fadd_v2bf16)) {
    __builtin_amdgcn_flat_atomic_fadd_v2bf16(
        reinterpret_cast<packed_t*>(output), value.packed);
  } else {
    gpuAtomicAddNoReturn(output, value0);
    gpuAtomicAddNoReturn(output + 1, value1);
  }
}

// Reduces non-long sorted runs with one logical subgroup per segment start or
// fixed chunk boundary. Each lane accumulates eight BF16 features; uniquely
// owned rows store directly, while split rows commit packed atomic pairs.
template <int64_t feature_size, typename index_t>
__global__ void index_select_backward_compact_bfloat16_kernel(
    const c10::BFloat16* grad,
    const index_t* sorted_indices,
    const int32_t* sorted_positions,
    const int32_t* segment_offsets,
    const int64_t* num_segments_ptr,
    const int32_t* long_chunk_counts,
    c10::BFloat16* grad_input,
    int64_t num_indices,
    int64_t num_rows,
    int64_t outer_size) {
  static_assert(
      feature_size == 64 || feature_size == 128 || feature_size == 256);
  constexpr int64_t kValuesPerLane = 8;
  constexpr int64_t kThreadsPerBlock = 256;
  constexpr int64_t kFeatureSize = feature_size;
  constexpr int64_t kGroupSize = kFeatureSize / kValuesPerLane;
  constexpr int64_t kGroupsPerBlock = kThreadsPerBlock / kGroupSize;
  static_assert(kFeatureSize % kValuesPerLane == 0);
  static_assert((kGroupSize & (kGroupSize - 1)) == 0);
  static_assert(64 % kGroupSize == 0);
  static_assert(kGroupSize * kGroupsPerBlock == kThreadsPerBlock);
  const int64_t lane = threadIdx.x;
  const int64_t wave = threadIdx.y;
  int64_t num_segments = lane == 0 ? *num_segments_ptr : 0;
  num_segments =
      index_select_backward_wave_broadcast_int64(num_segments, kGroupSize);
  const int64_t num_boundaries =
      (num_indices - 1) / INDEX_SELECT_BACKWARD_CHUNK_SIZE;
  const int64_t candidates_per_outer = num_segments + num_boundaries;
  const int64_t actual_work = outer_size * candidates_per_outer;
  const int64_t work_stride =
      static_cast<int64_t>(gridDim.x) * kGroupsPerBlock;

  for (int64_t work =
           static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + wave;
       work < actual_work;
       work += work_stride) {
    const int64_t candidate = work % candidates_per_outer;
    const int64_t outer = work / candidates_per_outer;
    int64_t chunk_begin = -1;
    int64_t chunk_end = -1;
    int64_t output_row = -1;
    auto mode = IndexSelectBackwardCompactMode::Invalid;

    if (lane == 0) {
      int64_t segment;
      int64_t run_end = num_indices;
      const bool starts_run = candidate < num_segments;
      if (starts_run) {
        segment = candidate;
        chunk_begin = static_cast<int64_t>(segment_offsets[segment]);
        if (segment + 1 < num_segments) {
          run_end = static_cast<int64_t>(segment_offsets[segment + 1]);
        }
      } else {
        chunk_begin =
            (candidate - num_segments + 1) * INDEX_SELECT_BACKWARD_CHUNK_SIZE;
        if (sorted_indices[chunk_begin] != sorted_indices[chunk_begin - 1]) {
          chunk_begin = -1;
          segment = -1;
        } else {
          const int64_t next_segment = index_select_backward_upper_bound(
              segment_offsets, num_segments, chunk_begin);
          segment = next_segment - 1;
          if (next_segment < num_segments) {
            run_end = static_cast<int64_t>(segment_offsets[next_segment]);
          }
        }
      }

      if (chunk_begin >= 0 && long_chunk_counts != nullptr &&
          long_chunk_counts[segment] != 0) {
        chunk_begin = -1;
      }
      if (chunk_begin >= 0) {
        const int64_t boundary_end =
            ((chunk_begin / INDEX_SELECT_BACKWARD_CHUNK_SIZE) + 1) *
            INDEX_SELECT_BACKWARD_CHUNK_SIZE;
        chunk_end = boundary_end < run_end ? boundary_end : run_end;
        output_row = static_cast<int64_t>(sorted_indices[chunk_begin]);
        mode = starts_run && chunk_end == run_end
            ? IndexSelectBackwardCompactMode::Direct
            : IndexSelectBackwardCompactMode::Atomic;
      }
    }

    chunk_begin =
        index_select_backward_wave_broadcast_int64(chunk_begin, kGroupSize);
    chunk_end =
        index_select_backward_wave_broadcast_int64(chunk_end, kGroupSize);
    output_row =
        index_select_backward_wave_broadcast_int64(output_row, kGroupSize);
    mode = static_cast<IndexSelectBackwardCompactMode>(WARP_SHFL(
        static_cast<uint32_t>(mode), 0, kGroupSize));
    if (mode == IndexSelectBackwardCompactMode::Invalid) {
      continue;
    }

    const int64_t feature = lane * kValuesPerLane;
    const int64_t output_offset =
        (outer * num_rows + output_row) * kFeatureSize + feature;
    if constexpr (kValuesPerLane == 4) {
      c10::BFloat16 sum0 = c10::BFloat16(0.0f);
      c10::BFloat16 sum1 = c10::BFloat16(0.0f);
      c10::BFloat16 sum2 = c10::BFloat16(0.0f);
      c10::BFloat16 sum3 = c10::BFloat16(0.0f);
      for (int64_t current = chunk_begin; current < chunk_end; ++current) {
        const int64_t source_row =
            static_cast<int64_t>(sorted_positions[current]);
        const int64_t source_offset =
            (outer * num_indices + source_row) * kFeatureSize;
        const uint32_t vector_offset = static_cast<uint32_t>(lane * 2);
        const auto values01 =
            memory::load_vector<2>(grad + source_offset, vector_offset);
        const auto values23 =
            memory::load_vector<2>(grad + source_offset, vector_offset + 1);
        sum0 += values01.val[0];
        sum1 += values01.val[1];
        sum2 += values23.val[0];
        sum3 += values23.val[1];
      }
      if (mode == IndexSelectBackwardCompactMode::Direct) {
        grad_input[output_offset] = sum0;
        grad_input[output_offset + 1] = sum1;
        grad_input[output_offset + 2] = sum2;
        grad_input[output_offset + 3] = sum3;
      } else {
        index_select_backward_atomic_add_bfloat16_pair(
            grad_input + output_offset, sum0, sum1);
        index_select_backward_atomic_add_bfloat16_pair(
            grad_input + output_offset + 2, sum2, sum3);
      }
    } else {
      static_assert(kValuesPerLane == 8);
      c10::BFloat16 sum0 = c10::BFloat16(0.0f);
      c10::BFloat16 sum1 = c10::BFloat16(0.0f);
      c10::BFloat16 sum2 = c10::BFloat16(0.0f);
      c10::BFloat16 sum3 = c10::BFloat16(0.0f);
      c10::BFloat16 sum4 = c10::BFloat16(0.0f);
      c10::BFloat16 sum5 = c10::BFloat16(0.0f);
      c10::BFloat16 sum6 = c10::BFloat16(0.0f);
      c10::BFloat16 sum7 = c10::BFloat16(0.0f);
      for (int64_t current = chunk_begin; current < chunk_end; ++current) {
        const int64_t source_row =
            static_cast<int64_t>(sorted_positions[current]);
        const int64_t source_offset =
            (outer * num_indices + source_row) * kFeatureSize;
        const uint32_t vector_offset = static_cast<uint32_t>(lane * 4);
        const auto values01 =
            memory::load_vector<2>(grad + source_offset, vector_offset);
        const auto values23 =
            memory::load_vector<2>(grad + source_offset, vector_offset + 1);
        const auto values45 =
            memory::load_vector<2>(grad + source_offset, vector_offset + 2);
        const auto values67 =
            memory::load_vector<2>(grad + source_offset, vector_offset + 3);
        sum0 += values01.val[0];
        sum1 += values01.val[1];
        sum2 += values23.val[0];
        sum3 += values23.val[1];
        sum4 += values45.val[0];
        sum5 += values45.val[1];
        sum6 += values67.val[0];
        sum7 += values67.val[1];
      }
      if (mode == IndexSelectBackwardCompactMode::Direct) {
        grad_input[output_offset] = sum0;
        grad_input[output_offset + 1] = sum1;
        grad_input[output_offset + 2] = sum2;
        grad_input[output_offset + 3] = sum3;
        grad_input[output_offset + 4] = sum4;
        grad_input[output_offset + 5] = sum5;
        grad_input[output_offset + 6] = sum6;
        grad_input[output_offset + 7] = sum7;
      } else {
        index_select_backward_atomic_add_bfloat16_pair(
            grad_input + output_offset, sum0, sum1);
        index_select_backward_atomic_add_bfloat16_pair(
            grad_input + output_offset + 2, sum2, sum3);
        index_select_backward_atomic_add_bfloat16_pair(
            grad_input + output_offset + 4, sum4, sum5);
        index_select_backward_atomic_add_bfloat16_pair(
            grad_input + output_offset + 6, sum6, sum7);
      }
    }
  }
}

// First pass for exceptionally long runs. Each logical work item reduces at
// most one fixed-size source chunk into bounded scratch, tiled over the
// flattened outer and feature dimensions.
template <
    typename scalar_t,
    typename acc_t,
    typename index_t,
    int values_per_thread>
__global__ void index_select_backward_long_first_pass_kernel(
    const scalar_t* grad,
    const int32_t* sorted_positions,
    const int32_t* segment_offsets,
    const int64_t* num_segments_ptr,
    const int32_t* long_chunk_counts,
    const int32_t* long_chunk_offsets,
    acc_t* scratch,
    int64_t num_indices,
    int64_t max_segments,
    int64_t max_long_chunks,
    int64_t inner_size,
    int64_t feature_offset,
    int64_t feature_count) {
  static_assert(values_per_thread == 1 || values_per_thread == 2);
  for (int64_t chunk = blockIdx.x; chunk < max_long_chunks;
       chunk += gridDim.x) {
    const int64_t segment = index_select_backward_upper_bound(
                                long_chunk_offsets, max_segments, chunk) -
        1;
    if (segment < 0 || segment >= max_segments) {
      continue;
    }
    const int64_t first_chunk =
        static_cast<int64_t>(long_chunk_offsets[segment]);
    const int64_t chunk_count =
        static_cast<int64_t>(long_chunk_counts[segment]);
    const int64_t chunk_in_segment = chunk - first_chunk;
    if (chunk_count == 0 || chunk_in_segment < 0 ||
        chunk_in_segment >= chunk_count) {
      continue;
    }

    const int64_t begin =
        static_cast<int64_t>(segment_offsets[segment]) +
        chunk_in_segment * INDEX_SELECT_BACKWARD_CHUNK_SIZE;
    const int64_t num_segments = *num_segments_ptr;
    const int64_t run_end = segment + 1 < num_segments
        ? static_cast<int64_t>(segment_offsets[segment + 1])
        : num_indices;
    const int64_t unbounded_end = begin + INDEX_SELECT_BACKWARD_CHUNK_SIZE;
    const int64_t end = unbounded_end < run_end ? unbounded_end : run_end;

    if constexpr (values_per_thread == 2) {
      static_assert(std::is_same_v<scalar_t, c10::BFloat16>);
      static_assert(std::is_same_v<acc_t, c10::BFloat16>);
      for (int64_t local_feature = threadIdx.x * 2;
           local_feature + 1 < feature_count;
           local_feature += blockDim.x * 2) {
        const int64_t flat_feature = feature_offset + local_feature;
        const int64_t outer = flat_feature / inner_size;
        const int64_t feature = flat_feature % inner_size;
        acc_t sum0 = acc_t(0);
        acc_t sum1 = acc_t(0);
        for (int64_t current = begin; current < end; ++current) {
          const int64_t source_row =
              static_cast<int64_t>(sorted_positions[current]);
          const int64_t source_offset =
              (outer * num_indices + source_row) * inner_size;
          const auto values = memory::load_vector<2>(
              grad + source_offset, static_cast<uint32_t>(feature / 2));
          sum0 += static_cast<acc_t>(values.val[0]);
          sum1 += static_cast<acc_t>(values.val[1]);
        }
        const int64_t scratch_offset =
            chunk * feature_count + local_feature;
        scratch[scratch_offset] = sum0;
        scratch[scratch_offset + 1] = sum1;
      }
    } else {
      for (int64_t local_feature = threadIdx.x;
           local_feature < feature_count;
           local_feature += blockDim.x) {
        const int64_t flat_feature = feature_offset + local_feature;
        const int64_t outer = flat_feature / inner_size;
        const int64_t feature = flat_feature % inner_size;
        acc_t sum = acc_t(0);
        for (int64_t current = begin; current < end; ++current) {
          const int64_t source_row =
              static_cast<int64_t>(sorted_positions[current]);
          const int64_t source_offset =
              (outer * num_indices + source_row) * inner_size + feature;
          sum += static_cast<acc_t>(grad[source_offset]);
        }
        scratch[chunk * feature_count + local_feature] = sum;
      }
    }
  }
}

// Finalizes each long run by summing its first-pass scratch chunks in order and
// storing the destination row once. Long runs are excluded from compact
// reduction, so this pass needs no output atomics.
template <typename acc_t, typename index_t, int values_per_thread>
__global__ void index_select_backward_long_final_pass_kernel(
    const index_t* sorted_indices,
    const int32_t* segment_offsets,
    const int32_t* long_chunk_counts,
    const int32_t* long_chunk_offsets,
    const acc_t* scratch,
    acc_t* grad_input,
    int64_t num_rows,
    int64_t max_segments,
    int64_t max_long_chunks,
    int64_t inner_size,
    int64_t feature_offset,
    int64_t feature_count) {
  static_assert(values_per_thread == 1 || values_per_thread == 2);
  for (int64_t chunk = blockIdx.x; chunk < max_long_chunks;
       chunk += gridDim.x) {
    const int64_t segment = index_select_backward_upper_bound(
                                long_chunk_offsets, max_segments, chunk) -
        1;
    if (segment < 0 || segment >= max_segments) {
      continue;
    }
    const int64_t first_chunk =
        static_cast<int64_t>(long_chunk_offsets[segment]);
    const int64_t chunk_count =
        static_cast<int64_t>(long_chunk_counts[segment]);
    if (chunk != first_chunk || chunk_count == 0) {
      continue;
    }

    const int64_t output_row = static_cast<int64_t>(
        sorted_indices[static_cast<int64_t>(segment_offsets[segment])]);
    if constexpr (values_per_thread == 2) {
      static_assert(std::is_same_v<acc_t, c10::BFloat16>);
      for (int64_t local_feature = threadIdx.x * 2;
           local_feature + 1 < feature_count;
           local_feature += blockDim.x * 2) {
        const int64_t flat_feature = feature_offset + local_feature;
        const int64_t outer = flat_feature / inner_size;
        const int64_t feature = flat_feature % inner_size;
        acc_t sum0 = acc_t(0);
        acc_t sum1 = acc_t(0);
        for (int64_t current = 0; current < chunk_count; ++current) {
          const int64_t scratch_offset =
              (first_chunk + current) * feature_count + local_feature;
          sum0 += scratch[scratch_offset];
          sum1 += scratch[scratch_offset + 1];
        }
        const int64_t output_offset =
            (outer * num_rows + output_row) * inner_size + feature;
        grad_input[output_offset] = sum0;
        grad_input[output_offset + 1] = sum1;
      }
    } else {
      for (int64_t local_feature = threadIdx.x;
           local_feature < feature_count;
           local_feature += blockDim.x) {
        const int64_t flat_feature = feature_offset + local_feature;
        const int64_t outer = flat_feature / inner_size;
        const int64_t feature = flat_feature % inner_size;
        acc_t sum = acc_t(0);
        for (int64_t current = 0; current < chunk_count; ++current) {
          sum += scratch[(first_chunk + current) * feature_count +
                         local_feature];
        }
        const int64_t output_offset =
            (outer * num_rows + output_row) * inner_size + feature;
        grad_input[output_offset] = sum;
      }
    }
  }
}
#endif

class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMul(self_data_start + index, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
#if defined(USE_ROCM)
    // TODO: this check is too coarse, revisit, we should only be checking for
    //       the availability of the builtins required by the implementation, at
    //       most.
    if(__builtin_amdgcn_processor_is("gfx942") ||
       __builtin_amdgcn_processor_is("gfx950"))
      return opportunistic_fastAtomicAdd(self_data_start, index, numel, *src_data);
    fastAtomicAdd(self_data_start, index, numel, *src_data, true);
#else
    fastAtomicAdd(self_data_start, index, numel, *src_data, true);
#endif
  }
};
static ReduceAdd reduce_add;

class ReduceMinimum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMin(self_data_start + index, *src_data);
  }
};
static ReduceMinimum reduce_minimum;

class ReduceMaximum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMax(self_data_start + index, *src_data);
  }
};
static ReduceMaximum reduce_maximum;

}

static Tensor wrapIndexOnce(const Tensor & index, int64_t dim, int64_t dim_size, bool check_range=true) {
//we don't need to check range in backward - if there were out of bounds indices forward should already have errored out
  if (index.numel() != 0 && check_range) {
    auto [index_min, index_max] = at::aminmax(index);
    at::_assert_async(index_max < dim_size);
    at::_assert_async(index_min >= -dim_size);
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  if (stride.empty()) {
    return stride;
  }
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t, int64_t, int64_t>
computeLinearIndex(const Tensor & src, TensorList indices, bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& device = src.options().device();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t nElemBefore = 1, nElemAfter = 1, strideBefore =0;
  int64_t dims_before = 0, dims_indexed = 0;
  for (const auto i: c10::irange(src.dim())) {
    if (indices[i].defined()) {
      dims_indexed++;
      // Cast index to the longType matching src's device
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i]).to(device);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
        if (i>0) {
           strideBefore = src.stride(i-1); // stride after undefined dimensions
        }
      }
    } else if (linearIndex.defined()) {
      nElemAfter *= src.size(i);
    } else {
      dims_before++;
      nElemBefore *= src.size(i);
    }
  }

  return std::make_tuple(std::move(linearIndex), nElemBefore, strideBefore, nElemAfter, dims_before, dims_indexed);
}


static std::tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>, int64_t, int64_t>
makeLinearIndex(Tensor self, IOptTensorListRef orig, bool check_range) {
  checkIndexTensorTypes(orig, /*allow_int*/true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  for (auto & i : indices) {
    if (i.defined() && i.dtype() == at::kInt) {
      i = i.to(at::kLong);
    }
  }
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  std::vector<int64_t> inversePerm;
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices, inversePerm) = transposeToFrontAndInvPerm(self, indices);
  }
  auto [linearIndex, nElemBefore, strideBefore, nElemAfter, dims_before, dims_indexed] =
    computeLinearIndex(self, indices, check_range);
  return std::make_tuple(std::move(linearIndex), std::move(self), nElemBefore, strideBefore, nElemAfter, std::move(inversePerm),
                         dims_before, dims_indexed);
}
namespace {

int64_t largestIndex(const Tensor &self) {
  int64_t result = 0;
  for (const auto i: c10::irange(self.dim())) {
    result += (self.sizes()[i] - 1) * self.strides()[i];
  }
  return result;
}

DimVector valsShape(IntArrayRef self_sizes,
                              int64_t dims_before,
                              int64_t dims_indexed,
                              IntArrayRef replacement_shape) {
  auto shape = DimVector(self_sizes);
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  shape.insert(
    shape.begin() + dims_before,
    replacement_shape.begin(),
    replacement_shape.end());
  return shape;
}

void index_put_with_sort_kernel(Tensor & self, const c10::List<std::optional<Tensor>>& indices, const Tensor & value, bool accumulate, bool unsafe) {
  TORCH_CHECK(!indices.empty() || is_expandable_to(value.sizes(), self.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", self.sizes());
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize, dims_before, dims_indexed;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm,
  dims_before, dims_indexed) = makeLinearIndex(self_, indices, !unsafe);
  auto vals_shape = valsShape(src.sizes(), dims_before, dims_indexed, linearIndex.sizes());
  int64_t num_indices = linearIndex.numel();
  expandedValue = expandedValue.expand(vals_shape).contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.const_data_ptr<int64_t>(), sorted_indices.mutable_data_ptr<int64_t>(),
        range.const_data_ptr<int64_t>(), orig_indices.mutable_data_ptr<int64_t>(),
        num_indices, false, 0, nbits);


      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());

      const int UNROLL = 4;
      const int indices_per_block = 4;
      const int warp_size = at::cuda::warp_size();
      dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
           std::clamp<int>(nElemBefore, 1, at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(warp_size, indices_per_block);

#ifdef USE_ROCM
      dim3 new_grid_many_indices(ceil_div(num_indices, (int64_t) (indices_per_block * warp_size)),
      grid.y == 1 ? std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size))) : grid.y,
      grid.z);
      dim3 new_grid(ceil_div(num_indices, (int64_t) (indices_per_block * warp_size)), grid.y, grid.z);
      size_t smem_dups_size = indices_per_block * warp_size * sizeof(int64_t);
#define KERNEL_GRID new_grid
#define KERNEL_SMEM smem_dups_size
#else
#define KERNEL_GRID grid
#define KERNEL_SMEM 0
#endif

      if (sliceSize == 1) {
        // This implementation is faster with high amounts of duplicates but could overflow
        // if FP16 / BF16 is used
        AT_DISPATCH_V2(
          expandedValue.scalar_type(),
          "indexing_backward_kernel_stride_1",
          AT_WRAP([&] {
            indexing_backward_kernel_stride_1<scalar_t><<<KERNEL_GRID, block, KERNEL_SMEM, stream>>>
            (
              sorted_indices.const_data_ptr<int64_t>(),
              orig_indices.const_data_ptr<int64_t>(),
              expandedValue.const_data_ptr<scalar_t>(),
              src_.mutable_data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }),
          AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
          // AT_EXPAND(AT_FLOAT8_TYPES),
          // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
          // should not be supported here, then reenable AT_FLOAT8_DTYPES
          kFloat8_e4m3fn,
          kFloat8_e5m2,
          kFloat8_e4m3fnuz,
          kFloat8_e5m2fnuz,
          kComplexHalf,
          kBComplex32,
          kHalf,
          kBool,
          kBFloat16);
      } else {
        if (sliceSize <= warp_size) {
          AT_DISPATCH_V2(
            expandedValue.scalar_type(),
            "indexing_backward_kernel_small_stride",
            AT_WRAP([&] {
              indexing_backward_kernel_small_stride<scalar_t><<<grid, block, 0, stream>>>(
                sorted_indices.const_data_ptr<int64_t>(),
                orig_indices.const_data_ptr<int64_t>(),
                expandedValue.const_data_ptr<scalar_t>(),
                src_.mutable_data_ptr<scalar_t>(),
                num_indices,
                sliceSize,
                strideBefore,
                nElemBefore,
                accumulate);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }),
            AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
            // AT_EXPAND(AT_FLOAT8_TYPES),
            // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
            // should not be supported here, then reenable AT_FLOAT8_DTYPES
            kFloat8_e4m3fn,
            kFloat8_e5m2,
            kFloat8_e4m3fnuz,
            kFloat8_e5m2fnuz,
            kComplexHalf,
            kBComplex32,
            kHalf,
            kBool,
            kBFloat16);
        } else {
#ifdef USE_ROCM
          if (num_indices >= 200000)
            AT_DISPATCH_V2(
              expandedValue.scalar_type(),
              "indexing_backward_many_indices",
              AT_WRAP([&] {
                indexing_backward_kernel_many_indices<scalar_t, UNROLL><<<new_grid_many_indices, block, smem_dups_size, stream>>>(
                  sorted_indices.const_data_ptr<int64_t>(),
                  orig_indices.const_data_ptr<int64_t>(),
                  expandedValue.const_data_ptr<scalar_t>(),
                  src_.mutable_data_ptr<scalar_t>(),
                  num_indices,
                  sliceSize,
                  strideBefore,
                  nElemBefore,
                  accumulate);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }),
              AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
              // AT_EXPAND(AT_FLOAT8_TYPES),
              // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
              // should not be supported here, then reenable AT_FLOAT8_DTYPES
              kFloat8_e4m3fn,
              kFloat8_e5m2,
              kFloat8_e4m3fnuz,
              kFloat8_e5m2fnuz,
              kComplexHalf,
              kBComplex32,
              kHalf,
              kBool,
              kBFloat16);
          else
#endif
          AT_DISPATCH_V2(
            expandedValue.scalar_type(),
            "indexing_backward",
            AT_WRAP([&] {
              indexing_backward_kernel<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
                sorted_indices.const_data_ptr<int64_t>(),
                orig_indices.const_data_ptr<int64_t>(),
                expandedValue.const_data_ptr<scalar_t>(),
                src_.mutable_data_ptr<scalar_t>(),
                num_indices,
                sliceSize,
                strideBefore,
                nElemBefore,
                accumulate);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }),
            AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
            // AT_EXPAND(AT_FLOAT8_TYPES),
            // TODO(#113663): clean up accumulation behavior in float8 dtypes, accumulate=True
            // should not be supported here, then reenable AT_FLOAT8_DTYPES
            kFloat8_e4m3fn,
            kFloat8_e5m2,
            kFloat8_e4m3fnuz,
            kFloat8_e5m2fnuz,
            kComplexHalf,
            kBComplex32,
            kHalf,
            kBool,
            kBFloat16);
        }
      }

#undef KERNEL_GRID
#undef KERNEL_SMEM

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_stub, &index_put_with_sort_kernel)

void index_put_with_sort_quantized(Tensor & self, const c10::List<std::optional<Tensor>>& indices, const Tensor & value, double scale, int zero_point, bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize, dims_before, dims_indexed;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm,
  dims_before, dims_indexed) = makeLinearIndex(self_, indices, !unsafe);
  auto vals_shape = valsShape(src.sizes(), dims_before, dims_indexed, linearIndex.sizes());
  int64_t num_indices = linearIndex.numel();
  expandedValue = expandedValue.expand(vals_shape).contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.const_data_ptr<int64_t>(), sorted_indices.mutable_data_ptr<int64_t>(),
        range.const_data_ptr<int64_t>(), orig_indices.mutable_data_ptr<int64_t>(),
        num_indices, false, 0, nbits);


      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());
      const int UNROLL = 4;
      const int indices_per_block = 4;
      const int warp_size = at::cuda::warp_size();
      dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
           std::clamp<int>(nElemBefore, 1, at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(warp_size, indices_per_block);

      AT_DISPATCH_QINT_TYPES(
        src.scalar_type(), "indexing_backward_quantized", [&] {
        constexpr int64_t qmin = std::numeric_limits<typename scalar_t::underlying>::min();
        constexpr int64_t qmax = std::numeric_limits<typename scalar_t::underlying>::max();
        float inv_scale = 1.0f / static_cast<float>(scale);

        indexing_backward_kernel_quantized<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
          sorted_indices.const_data_ptr<int64_t>(),
          orig_indices.const_data_ptr<int64_t>(),
          expandedValue.const_data_ptr<float>(),
          src_.mutable_data_ptr<scalar_t>(),
          num_indices,
          sliceSize,
          strideBefore,
          nElemBefore,
          inv_scale,
          zero_point,
          qmin,
          qmax);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_quantized_stub, &index_put_with_sort_quantized)
} //anonymous


// Check tensor dimensions for index operations, and return the slice size.
static size_t getSliceSize(const Tensor & dst,
                              int dim,
                              const Tensor & index,
                              const Tensor & src)
{
  const auto dstDims = dst.dim();
  const auto srcDims = src.dim();

  TORCH_CHECK(index.dim() <= 1, "Index must be vector or scalar");

  size_t dstSliceSize = 1;
  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim ", dim, " is out of bounds");
  for (const auto d: c10::irange(dstDims)) {
    if (d != dim) {
      dstSliceSize *= dst.size(d);
    }
  }

  TORCH_CHECK(dim < srcDims, "Indexing dim ", dim, " is out of bounds");
  TORCH_CHECK(index.numel() == src.size(dim),
             "length of src.size[dim] is not equal to length of indices");

  size_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (const auto d: c10::irange(srcDims)) {
    if (d != dim) {
      srcSliceSize *= src.size(d);
      if (!mismatch && dst.size(d) != src.size(d)) mismatch = true;
    }
  }

  TORCH_CHECK(dstSliceSize == srcSliceSize,
             "Source/destination tensor have different slice sizes (",
             dstSliceSize, " vs ", srcSliceSize, ")");

  if (mismatch) {
    TORCH_WARN_ONCE(
        "Warning: source/destination slices have same size but different "
        "shape for an index operation.  This behavior is deprecated.\n");
  }

  return dstSliceSize;
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFuncLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          typename func_t>
__global__ void indexFuncSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }

  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFuncSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    T val = src.data[srcOffset] * alpha;
    op(dst.data, dstOffset, dstNumel, &val);
  }
}

// Compare the stride between adjacent slices (sliceStride) with strides in the
// other dimensions (i.e., strides *inside* each slice).
//
// - Returns true if some dimension inside the slice has lower stride than
//   sliceStride.  The simplest example is a 2-D contiguous tensor with sliceDim
//   == 0 (that is, each slice is a row).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "index-major order".  For example, if thread count equals slice size, then
//   all threads process slice #0 in lockstep, and then slice #1, and so on.
//
// - Otherwise (i.e., sliceStride has the lowest value), this function returns
//   false.  The simplest example is a 2-D contiguous tensor with sliceDim == 1
//   (each slice is a column).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "elementInSlice-major order".  For example, each thread can process element
//   #0 of every slice, and then element #1 of every slice, and so on.
template <typename scalar_t>
bool indexShouldBeMajor(cuda::detail::TensorInfo<scalar_t, unsigned int> &info,
                                    int sliceDim)
{
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (const auto i: c10::irange(info.dims)) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

void index_add_cuda_impl(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  // Scalars are treated as 1-d tensor
  const Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(result.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims" );
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");

  if (globalContext().deterministicAlgorithms()){
    torch::List<std::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for ([[maybe_unused]] const auto i : c10::irange(dim)) {
      indices.emplace_back();
    }
    indices.emplace_back(index.to(at::kLong));
    result.index_put_(indices, source * alpha, true);
    return;
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  const uint64_t sliceSize = getSliceSize(self_, dim, index, source_);
  const uint64_t sourceTotalSize = source.numel();
  const uint64_t selfAddDimSize = self_.size(dim);
  const uint64_t numIndex = index.numel();
  const uint64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool indContig = index.is_contiguous();

  const int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  // Fast path: index_add_(0, idx, src) with alpha == 1 is equivalent to
  // self.scatter_add_(0, idx.view({n, 1, ...}).expand_as(src), src). Delegate
  // so scatter_add's own TMA/vectorized eligibility check + dispatch is the
  // single source of truth (see PR #182675). Pattern from
  // pytorch/pytorch#180430.
  // Gated on CUDA >= 12.8: pre-12.8 builds compile out the TMA branch in
  // scatter_add and fall back to its vectorized atomicAdd path, which
  // regresses skewed/high-contention workloads vs indexFunc{Small,Large}Index
  // (warp-per-entry scheduling concentrates atomic contention on hot rows).
  // Older builds therefore stay on the existing indexFunc dispatch.
  // index_add supports {complex64, complex128, ComplexHalf, Bool} that
  // scatter_add does not, so exclude those and let them use indexFunc.
  // The dtype check is ordered FIRST so short-circuit evaluation skips
  // alpha.equal(1) for complex `self`, where alpha may itself be a
  // complex Scalar and the equality comparison would be ill-defined.
  const auto stype = self_.scalar_type();
  const bool dtype_supported_by_scatter_add =
      !c10::isComplexType(stype) && stype != at::kBool;
  if (dtype_supported_by_scatter_add && dim == 0 &&
      alpha.equal(1) && numIndex > 0 &&
      index.dim() <= 1 && indContig) {
    std::vector<int64_t> idx_shape(source_.dim(), 1);
    idx_shape[0] = static_cast<int64_t>(numIndex);
    self_.scatter_add_(0, index.view(idx_shape).expand_as(source_), source_);
    return;
  }
#endif

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)     \
  indexFuncSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>   \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      selfInfo, sourceInfo, indexInfo,                                                  \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize,                              \
      selfNumel, reduce_add, alpha_value);                                              \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                        \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)            \
  indexFuncLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                      \
                      SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>          \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                       \
      selfInfo, sourceInfo, indexInfo,                                      \
      selfAddDim, sourceAddDim, sourceTotalSize,                            \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                \
      selfAddDimSize, selfNumel, reduce_add, alpha_value);                  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  uint64_t defaultMaxBlockThreads = getDefaultMaxThreadsPerBlock();
  const dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (uint64_t)128), (uint64_t)(mpc * 8)));
  const dim3 smallIndexBlock(std::min(sliceSize, (uint64_t)128));

  const dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (uint64_t)128), (uint64_t)(mpc * 8)));
  //On ROCm, std::min -> ::min did not work as expected on when outTotalSize>=2147483648
  dim3 largeIndexBlock( (sourceTotalSize < defaultMaxBlockThreads) ? sourceTotalSize : defaultMaxBlockThreads );

  if (cuda::detail::canUse32BitIndexMath(result) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::ComplexHalf, result.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, unsigned int> selfInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(self_);
      const int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);
      const auto alpha_value = alpha.to<scalar_t>();
      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        auto sourceInfo =
          cuda::detail::getTensorInfo<const scalar_t, unsigned int>(source_);
        const int sourceAddDim = sourceInfo.collapseDims(dim);
        sourceInfo.reduceDim(sourceAddDim);

        auto indexInfo =
        cuda::detail::getTensorInfo<const index_t, unsigned int>(index);
        indexInfo.collapseDims();

        // A reasonable choice for when to have each thread iterate over
        // index to choose
        if (numIndex <= 16) {
          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
          } else {
            SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
          }
        } else {
          const bool indexIsMajor = indexShouldBeMajor(selfInfo, selfAddDim);

          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
            }
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, uint64_t> selfInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(self_);
      const int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);
      const auto alpha_value = alpha.to<scalar_t>();

      cuda::detail::TensorInfo<const scalar_t, uint64_t> sourceInfo =
        cuda::detail::getTensorInfo<const scalar_t, uint64_t>(source_);
      const int sourceAddDim = sourceInfo.collapseDims(dim);
      sourceInfo.reduceDim(sourceAddDim);

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        cuda::detail::TensorInfo<const index_t, uint64_t> indexInfo =
          cuda::detail::getTensorInfo<const index_t, uint64_t>(index);
        indexInfo.collapseDims();

        LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
      });
    });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

template <typename func_t>
void index_reduce_func_cuda_impl(
  const Tensor& self,
  int64_t dim,
  const Tensor& index,
  const Tensor& source,
  bool include_self,
  const ReductionType& reduce,
  const func_t& reduce_func,
  const Tensor& result) {
  globalContext().alertNotDeterministic("index_reduce_cuda");

  if (!result.is_same(self)) result.copy_(self);

  // Scalars are treated as 1-d tensor
  Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(result.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims" );
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");

  if (!include_self) {
    AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      self.scalar_type(), "index_reduce_func_cuda_exclude_input_init", [&] {
      scalar_t init_val;
      switch (reduce) {
        case ReductionType::PROD:
          init_val = (scalar_t)1;
          break;
        case ReductionType::MAX:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? -std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::lowest();
          break;
        case ReductionType::MIN:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::max();
          break;
        default:
          init_val = (scalar_t)0;
          break;
      }
      // index_fill_ requires index to be a LongTensor
      self_.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
    });
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  uint64_t sliceSize = getSliceSize(self_, dim, index, source_);
  uint64_t sourceTotalSize = source.numel();
  uint64_t selfReduceDimSize = self_.size(dim);
  uint64_t numIndex = index.numel();
  uint64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bool indContig = index.is_contiguous();

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // Fast path: index_reduce_(0, idx, src, amin/amax) is equivalent to
  // scatter_reduce_(0, idx.view({n, 1, ...}).expand_as(src), src). Reuse the
  // scatter path so eligibility and architecture-specific dispatch remain in one place.
  // For include_self=False, the identity initialization above has already
  // replaced the indexed rows, so the delegated reduction includes that state.
  const auto stype = self_.scalar_type();
  const bool dtype_supported_by_scatter_reduce =
      stype == at::kHalf || stype == at::kBFloat16;
  const bool minmax_reduce =
      reduce == ReductionType::MAX || reduce == ReductionType::MIN;
  const bool contiguous_dim0_rows = self_.is_contiguous() && source_.is_contiguous();
  const int device_major = at::cuda::getCurrentDeviceProperties()->major;
  const bool row_size_supported = sliceSize * self_.element_size() >= 16 &&
      (sliceSize * self_.element_size()) % 16 == 0;
  if (dtype_supported_by_scatter_reduce && minmax_reduce && contiguous_dim0_rows &&
      row_size_supported && device_major >= 8 && dim == 0 && numIndex > 0 && index.dim() == 1 &&
      indContig && source_.dim() > 0 &&
      source_.size(0) == static_cast<int64_t>(numIndex)) {
    self_.scatter_reduce_(
        0, index.view({static_cast<int64_t>(numIndex), 1}).expand_as(source_), source_,
        reduce == ReductionType::MAX ? "amax" : "amin", /*include_self=*/true);
    return;
  }
#endif

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)                  \
  indexFuncSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>                \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                                \
      selfInfo, sourceInfo, indexInfo,                                                               \
      selfReduceDim, sourceReduceDim, sliceSize, selfReduceDimSize,                                  \
      selfNumel, reduce_func, alpha_value);                                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                                     \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)                         \
  indexFuncLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                                   \
                     SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>                        \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                                    \
      selfInfo, sourceInfo, indexInfo,                                                   \
      selfReduceDim, sourceReduceDim, sourceTotalSize,                                   \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                             \
      selfReduceDimSize, selfNumel, reduce_func, alpha_value);                           \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  uint64_t defaultMaxBlockThreads = getDefaultMaxThreadsPerBlock();
  dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (uint64_t)128), (uint64_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (uint64_t)128));

  dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (uint64_t)128), (uint64_t)(mpc * 8)));
  //On ROCm, std::min -> ::min did not work as expected on when outTotalSize>=2147483648
  dim3 largeIndexBlock( (sourceTotalSize < defaultMaxBlockThreads) ? sourceTotalSize : defaultMaxBlockThreads );

  if (cuda::detail::canUse32BitIndexMath(result) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, result.scalar_type(), "index_reduce", [&] {
      cuda::detail::TensorInfo<scalar_t, unsigned int> selfInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(self_);
      int selfReduceDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfReduceDim);
      auto alpha_value = (scalar_t) 1;
      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_reduce_cuda", [&] () {
        auto sourceInfo =
          cuda::detail::getTensorInfo<const scalar_t, unsigned int>(source_);
        int sourceReduceDim = sourceInfo.collapseDims(dim);
        sourceInfo.reduceDim(sourceReduceDim);

        auto indexInfo =
        cuda::detail::getTensorInfo<const index_t, unsigned int>(index);
        indexInfo.collapseDims();

        // A reasonable choice for when to have each thread iterate over
        // index to choose
        if (numIndex <= 16) {
          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
          } else {
            SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
          }
        } else {
          bool indexIsMajor = indexShouldBeMajor(selfInfo, selfReduceDim);

          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
            }
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_reduce", [&] {
      cuda::detail::TensorInfo<scalar_t, uint64_t> selfInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(self_);
      int selfReduceDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfReduceDim);
      auto alpha_value = (scalar_t) 1;

      cuda::detail::TensorInfo<const scalar_t, uint64_t> sourceInfo =
        cuda::detail::getTensorInfo<const scalar_t, uint64_t>(source_);
      int sourceReduceDim = sourceInfo.collapseDims(dim);
      sourceInfo.reduceDim(sourceReduceDim);

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_reduce_cuda", [&] () {
        cuda::detail::TensorInfo<const index_t, uint64_t> indexInfo =
          cuda::detail::getTensorInfo<const index_t, uint64_t>(index);
        indexInfo.collapseDims();

        LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
      });
    });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

TORCH_IMPL_FUNC(index_add_cuda_out)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  index_add_cuda_impl(self, dim, index, source, alpha, result);
}

TORCH_IMPL_FUNC(index_reduce_cuda_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const std::string_view reduce,
 bool include_self,
 const Tensor& result) {
  TORCH_WARN_ONCE("index_reduce() is in beta and the API may change at any time.");

  if (reduce == "prod") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::PROD, reduce_multiply, result);
  } else if (reduce == "mean") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MEAN, reduce_add, result);
    auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
    counts.index_add_(dim, index, at::ones_like(source));
    counts.masked_fill_(counts == 0, 1);
    if (result.is_floating_point() || result.is_complex()) {
      result.div_(counts);
    } else {
      result.div_(counts, "floor");
    }
  } else if (reduce == "amax") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MAX, reduce_maximum, result);
  } else if (reduce == "amin") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MIN, reduce_minimum, result);
  } else {
    TORCH_CHECK(false, "reduce argument must be either prod, mean, amax or amin, got ", reduce, ".");
  }
}

namespace {
// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<const T, IndexType> src,
                                      cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}


namespace {

// When using a 0-dim scalar tensor, we need the legacy (THC) semantics of
// TensorInfo: Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
cuda::detail::TensorInfo<T, IndexType>
tensorInfoLegacyIfScalar(cuda::detail::TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}


}


template <typename scalar_t>
void index_select_out_cuda_impl(
    Tensor& out,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  uint64_t numIndices = index.numel();
  auto selfDims = self.dim() == 0 ? 1 : self.dim();

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      index.dim() <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(
      !(self.dim() == 0 && numIndices != 1), "index_select(): Index to scalar can have only 1 value, got ", numIndices, " value(s)");
  TORCH_CHECK(dim < selfDims, "Indexing dim is out of bounds");

  std::vector<int64_t> newSize = self.sizes().vec();
  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }

  if (self.is_quantized()){
      out = at::empty_quantized(newSize, out);
  } else {
    at::native::resize_output(out, newSize);
  }

  uint64_t outTotalSize = out.numel();
  if (outTotalSize == 0) {
    return;
  }

  bool indContig = index.is_contiguous();

  // The `self` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  uint64_t selfSelectDimSize = self.dim() == 0 ? 1 : self.size(dim);
  uint64_t sliceSize = outTotalSize / numIndices;

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)         \
  indexSelectSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      outInfo, selfInfo, indicesInfo,                                                   \
      outSelectDim, selfSelectDim, static_cast<TYPE>(sliceSize),                        \
      selfSelectDimSize);                                                               \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  uint64_t defaultMaxBlockThreads = getDefaultMaxThreadsPerBlock();
  dim3 smallIndexGrid(std::min(ceil_div(sliceSize, defaultMaxBlockThreads), (uint64_t) (mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, defaultMaxBlockThreads));

  // SmallIndexKernel is more performant when the number of indices is small, and pre-loading
  // the index reduces memory accesses. When the number of indices is large, we avoid that
  // and increase parallelism by calling gather_out which is a generalization of index_select
  if (cuda::detail::canUse32BitIndexMath(out) &&
      cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(index) &&
      numIndices <= 16
      ) {
    auto outInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, unsigned int>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto  selfInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<const scalar_t, unsigned int>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_out_cuda_impl", [&] () {
      auto indicesInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<const index_t, unsigned int>(index));
      indicesInfo.collapseDims();

      // A reasonable choice for when to have each thread iterate over
      // indices to choose
      if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
        SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
      } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
        SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
      } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
        SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
      }
    });
  } else {
    std::vector<int64_t> tmpSize(newSize.size(), 1);
    if (self.dim() > 0) {
      tmpSize[dim] = numIndices;
    }
    at::gather_out(out, self, dim, index.view(tmpSize).expand(newSize));
    return;
  }
#undef SMALL_INDEX
}
} // anonymous namespace

Tensor& index_select_out_cuda(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  static constexpr std::string_view DIM_WARNING =
      "Tensor too large or too many (> 25) dimensions";
  TORCH_CHECK(
      at::cuda::check_device({out, self, index}),
      "Input, output and indices must be on the current device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  if (self.is_quantized()) {
    TORCH_CHECK(
        self.qscheme() == kPerTensorAffine,
        "Only per_tensor quantized quantized tensors are supported by index_select.")
    AT_DISPATCH_QINT_TYPES(out.scalar_type(), "index_select_quant_cuda", [&] {
      index_select_out_cuda_impl<scalar_t>(out, self, dim, index);
    });
  } else {
    AT_DISPATCH_V2(
        out.scalar_type(),
        "index_select_cuda",
        AT_WRAP([&] {
          index_select_out_cuda_impl<scalar_t>(out, self, dim, index);
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
        AT_EXPAND(AT_FLOAT8_TYPES),
        kComplexHalf,
        kBComplex32,
        kHalf,
        kBool,
        kBFloat16);
  }

  return out;
}

Tensor index_select_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor out = at::empty({0}, self.options());
  at::native::index_select_out_cuda(self, dim, index, out);
  return out;
}

// Dispatches index_select backward to the ROCm sorted-reduction path only for
// large supported workloads on gfx942/gfx950. Small, unsupported, misaligned,
// subclass, deterministic, and non-ROCm cases retain native index_add_ behavior.
Tensor index_select_backward_cuda(
    const Tensor& grad,
    IntArrayRef self_sizes,
    int64_t dim,
    const Tensor& index) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(grad));
  Tensor result;
  {
    result = grad.new_zeros(self_sizes, grad.options());
  }
  if (isTensorSubclassLike(index)) {
    return result.index_add(dim, index, grad);
  }
  dim = at::maybe_wrap_dim(dim, result.dim());

#if defined(USE_ROCM)
  const auto dtype = grad.scalar_type();
  const bool supported_dtype =
      dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat ||
      dtype == at::kDouble;
  const bool use_sorted_backward =
      at::detail::getCUDAHooks().isGPUArch({"gfx942", "gfx950"}) &&
      !globalContext().deterministicAlgorithms() && grad.is_contiguous() &&
      index.device() == grad.device() && index.is_contiguous() &&
      index.dim() <= 1 &&
      (index.scalar_type() == at::kInt || index.scalar_type() == at::kLong) &&
      supported_dtype && index.numel() <= std::numeric_limits<int>::max();

  if (use_sorted_backward) {
    const int64_t num_indices = index.numel();
    const int64_t num_rows = result.dim() == 0 ? 1 : result.size(dim);
    int64_t outer_size = 1;
    int64_t inner_size = 1;
    for (const auto d : c10::irange(result.dim())) {
      if (static_cast<int64_t>(d) < dim) {
        outer_size *= result.size(d);
      } else if (static_cast<int64_t>(d) > dim) {
        inner_size *= result.size(d);
      }
    }

    bool valid_grad_shape = grad.dim() == result.dim();
    if (valid_grad_shape) {
      for (const auto d : c10::irange(result.dim())) {
        const auto expected_size =
            static_cast<int64_t>(d) == dim ? num_indices : result.size(d);
        valid_grad_shape = valid_grad_shape && grad.size(d) == expected_size;
      }
    }
    if (!valid_grad_shape) {
      return result.index_add_(dim, index, grad);
    }
    if (num_indices == 0 || outer_size == 0 || inner_size == 0) {
      return result;
    }
    const bool supported_inner_size =
        inner_size == 64 || inner_size == 128 || inner_size == 256;
    if (num_indices < INDEX_SELECT_BACKWARD_MIN_SORT_SIZE ||
        !supported_inner_size) {
      return result.index_add_(dim, index, grad);
    }

    if (num_rows == 0 || num_rows > std::numeric_limits<int>::max()) {
      return result.index_add_(dim, index, grad);
    }
    const auto position_options = index.options().dtype(at::kInt);
    const int64_t max_segments = std::min(num_indices, num_rows);
    const int64_t num_boundaries =
        (num_indices - 1) / INDEX_SELECT_BACKWARD_CHUNK_SIZE;
    const int64_t max_candidates = max_segments + num_boundaries;
    if (max_candidates > std::numeric_limits<int64_t>::max() / outer_size) {
      return result.index_add_(dim, index, grad);
    }

    Tensor sorted_indices;
    Tensor original_positions;
    Tensor positions;
    Tensor segment_offsets;
    Tensor num_segments;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    {
      sorted_indices = at::empty_like(index, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      original_positions = at::empty({num_indices}, position_options);
      positions = at::arange(num_indices, position_options);
      segment_offsets = at::empty({max_segments}, position_options);
      num_segments = at::empty({}, index.options().dtype(at::kLong));
      auto [index_min, index_max] = at::aminmax(index);
      at::_assert_async(index_min >= 0);
      at::_assert_async(index_max < num_rows);
    }

    {
      AT_DISPATCH_INDEX_TYPES(
          index.scalar_type(), "index_select_backward_sort", [&] {
            {
              cuda::cub::radix_sort_pairs(
                  index.const_data_ptr<index_t>(),
                  sorted_indices.mutable_data_ptr<index_t>(),
                  positions.const_data_ptr<int32_t>(),
                  original_positions.mutable_data_ptr<int32_t>(),
                  num_indices,
                  false,
                  0,
                  cuda::cub::get_num_bits(num_rows));
            }

            {
              cuda::cub::unique_by_key(
                  sorted_indices.const_data_ptr<index_t>(),
                  cccl_counting_iterator<int32_t>{0},
                  segment_offsets.mutable_data_ptr<int32_t>(),
                  num_segments.mutable_data_ptr<int64_t>(),
                  num_indices);
            }
          });
    }

    const int64_t multiprocessor_count =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const bool use_long_reduction =
        num_indices >= INDEX_SELECT_BACKWARD_LONG_RUN_SIZE;
    int64_t max_long_chunks = 0;
    Tensor long_chunk_counts;
    Tensor long_chunk_offsets;
    if (use_long_reduction) {
      max_long_chunks =
          num_indices / INDEX_SELECT_BACKWARD_CHUNK_SIZE +
          num_indices / INDEX_SELECT_BACKWARD_LONG_RUN_SIZE;
      TORCH_INTERNAL_ASSERT(max_long_chunks > 0);
      long_chunk_counts = positions;
      long_chunk_offsets = at::empty({max_segments}, position_options);
      constexpr int metadata_threads = 256;
      const int metadata_grid = static_cast<int>(std::min<int64_t>(
          ceil_div(max_segments, static_cast<int64_t>(metadata_threads)),
          multiprocessor_count * 8));
      index_select_backward_long_chunk_counts_kernel
          <<<metadata_grid, metadata_threads, 0, stream>>>(
              segment_offsets.const_data_ptr<int32_t>(),
              num_segments.const_data_ptr<int64_t>(),
              long_chunk_counts.mutable_data_ptr<int32_t>(),
              num_indices,
              max_segments);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      cuda::cub::exclusive_sum(
          long_chunk_counts.const_data_ptr<int32_t>(),
          long_chunk_offsets.mutable_data_ptr<int32_t>(),
          max_segments);
    }

    const bool packed_scratch_fits =
        !use_long_reduction ||
        max_long_chunks <=
            INDEX_SELECT_BACKWARD_MAX_SCRATCH_BYTES /
                (2 * static_cast<int64_t>(sizeof(c10::BFloat16)));
    const bool use_packed_bfloat16 =
        dtype == at::kBFloat16 && supported_inner_size && packed_scratch_fits &&
        memory::can_vectorize_up_to<c10::BFloat16>(reinterpret_cast<const char*>(
            grad.const_data_ptr<c10::BFloat16>())) >= 2;
    const int threads = use_packed_bfloat16
        ? 64
        : inner_size <= 64 ? 64 : inner_size <= 128 ? 128 : 256;
    const int64_t max_blocks = multiprocessor_count * 8;
    const int64_t max_work = outer_size * max_candidates;
    const int grid =
        static_cast<int>(std::min<int64_t>(max_work, max_blocks));
    const int32_t* long_chunk_counts_ptr = use_long_reduction
        ? long_chunk_counts.const_data_ptr<int32_t>()
        : nullptr;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grad.scalar_type(),
        "index_select_backward_sorted",
        [&] {
          using acc_t = std::conditional_t<
              std::is_same_v<scalar_t, c10::BFloat16>,
              c10::BFloat16,
              at::opmath_type<scalar_t>>;
          Tensor accumulation;
          const int64_t feature_plane = outer_size * inner_size;
          int64_t scratch_feature_tile = 0;
          Tensor long_scratch;
          accumulation = result;
          if constexpr (!std::is_same_v<scalar_t, acc_t>) {
            accumulation = at::zeros(
                result.sizes(), result.options().dtype(toOpMathType(dtype)));
          }

          if (use_long_reduction) {
            TORCH_INTERNAL_ASSERT(
                max_long_chunks <=
                std::numeric_limits<int64_t>::max() /
                    static_cast<int64_t>(sizeof(acc_t)));
            const int64_t bytes_per_feature =
                max_long_chunks * static_cast<int64_t>(sizeof(acc_t));
            int64_t max_scratch_features =
                INDEX_SELECT_BACKWARD_MAX_SCRATCH_BYTES / bytes_per_feature;
            if (use_packed_bfloat16) {
              max_scratch_features -= max_scratch_features % 2;
              TORCH_INTERNAL_ASSERT(max_scratch_features >= 2);
            } else {
              TORCH_INTERNAL_ASSERT(max_scratch_features >= 1);
            }
            scratch_feature_tile =
                std::min<int64_t>(feature_plane, max_scratch_features);
            if (use_packed_bfloat16) {
              scratch_feature_tile -= scratch_feature_tile % 2;
            }
            TORCH_INTERNAL_ASSERT(
                scratch_feature_tile > 0 &&
                max_long_chunks <=
                    std::numeric_limits<int64_t>::max() /
                        scratch_feature_tile);
            long_scratch = at::empty(
                {max_long_chunks, scratch_feature_tile}, accumulation.options());
          }

          AT_DISPATCH_INDEX_TYPES(
              index.scalar_type(), "index_select_backward_sorted_index", [&] {
                {
                  if (use_packed_bfloat16) {
                    if constexpr (
                        std::is_same_v<scalar_t, c10::BFloat16> &&
                        std::is_same_v<acc_t, c10::BFloat16>) {
                      const auto launch_packed_bfloat16 =
                          [&](auto feature_size_constant) {
                            constexpr int64_t packed_feature_size =
                                decltype(feature_size_constant)::value;
                            constexpr int64_t packed_values_per_lane = 8;
                            constexpr int64_t packed_group_size =
                                packed_feature_size / packed_values_per_lane;
                            constexpr int64_t packed_groups_per_block =
                                256 / packed_group_size;
                            const int packed_grid = static_cast<int>(
                                std::min<int64_t>(
                                    ceil_div(
                                        max_work, packed_groups_per_block),
                                    max_blocks));
                            const dim3 packed_block(
                                packed_group_size, packed_groups_per_block);
                            index_select_backward_compact_bfloat16_kernel<
                                packed_feature_size,
                                index_t><<<packed_grid, packed_block, 0, stream>>>(
                                grad.const_data_ptr<c10::BFloat16>(),
                                sorted_indices.const_data_ptr<index_t>(),
                                original_positions.const_data_ptr<int32_t>(),
                                segment_offsets.const_data_ptr<int32_t>(),
                                num_segments.const_data_ptr<int64_t>(),
                                long_chunk_counts_ptr,
                                accumulation.mutable_data_ptr<c10::BFloat16>(),
                                num_indices,
                                num_rows,
                                outer_size);
                          };
                      switch (inner_size) {
                        case 64:
                          launch_packed_bfloat16(
                              std::integral_constant<int64_t, 64>{});
                          break;
                        case 128:
                          launch_packed_bfloat16(
                              std::integral_constant<int64_t, 128>{});
                          break;
                        case 256:
                          launch_packed_bfloat16(
                              std::integral_constant<int64_t, 256>{});
                          break;
                        default:
                          TORCH_INTERNAL_ASSERT(false);
                      }
                      C10_CUDA_KERNEL_LAUNCH_CHECK();
                    } else {
                      TORCH_INTERNAL_ASSERT(false);
                    }
                } else {
                  index_select_backward_compact_kernel<
                      scalar_t,
                      acc_t,
                      index_t,
                      1><<<grid, threads, 0, stream>>>(
                      grad.const_data_ptr<scalar_t>(),
                      sorted_indices.const_data_ptr<index_t>(),
                      original_positions.const_data_ptr<int32_t>(),
                      segment_offsets.const_data_ptr<int32_t>(),
                      num_segments.const_data_ptr<int64_t>(),
                      long_chunk_counts_ptr,
                      accumulation.mutable_data_ptr<acc_t>(),
                      num_indices,
                      num_rows,
                      outer_size,
                      inner_size,
                      max_segments);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                  }
                }

                if (use_long_reduction) {
                  const int64_t long_max_blocks =
                      multiprocessor_count *
                      (use_packed_bfloat16 ? 16 : 8);
                  const int long_grid = static_cast<int>(
                      std::min<int64_t>(max_long_chunks, long_max_blocks));
                  for (int64_t feature_offset = 0;
                       feature_offset < feature_plane;
                       feature_offset += scratch_feature_tile) {
                    const int64_t feature_count = std::min<int64_t>(
                        scratch_feature_tile, feature_plane - feature_offset);
                      if (use_packed_bfloat16) {
                        if constexpr (
                            std::is_same_v<scalar_t, c10::BFloat16> &&
                            std::is_same_v<acc_t, c10::BFloat16>) {
                          {
                            index_select_backward_long_first_pass_kernel<
                                scalar_t,
                                acc_t,
                                index_t,
                                2><<<long_grid, 64, 0, stream>>>(
                                grad.const_data_ptr<scalar_t>(),
                                original_positions.const_data_ptr<int32_t>(),
                                segment_offsets.const_data_ptr<int32_t>(),
                                num_segments.const_data_ptr<int64_t>(),
                                long_chunk_counts.const_data_ptr<int32_t>(),
                                long_chunk_offsets.const_data_ptr<int32_t>(),
                                long_scratch.mutable_data_ptr<acc_t>(),
                                num_indices,
                                max_segments,
                                max_long_chunks,
                                inner_size,
                                feature_offset,
                                feature_count);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                          }
                          {
                            index_select_backward_long_final_pass_kernel<
                                acc_t,
                                index_t,
                                2><<<long_grid, 64, 0, stream>>>(
                                sorted_indices.const_data_ptr<index_t>(),
                                segment_offsets.const_data_ptr<int32_t>(),
                                long_chunk_counts.const_data_ptr<int32_t>(),
                                long_chunk_offsets.const_data_ptr<int32_t>(),
                                long_scratch.const_data_ptr<acc_t>(),
                                accumulation.mutable_data_ptr<acc_t>(),
                                num_rows,
                                max_segments,
                                max_long_chunks,
                                inner_size,
                                feature_offset,
                                feature_count);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                          }
                        } else {
                          TORCH_INTERNAL_ASSERT(false);
                        }
                      } else {
                        {
                          index_select_backward_long_first_pass_kernel<
                              scalar_t,
                              acc_t,
                              index_t,
                              1><<<long_grid, threads, 0, stream>>>(
                              grad.const_data_ptr<scalar_t>(),
                              original_positions.const_data_ptr<int32_t>(),
                              segment_offsets.const_data_ptr<int32_t>(),
                              num_segments.const_data_ptr<int64_t>(),
                              long_chunk_counts.const_data_ptr<int32_t>(),
                              long_chunk_offsets.const_data_ptr<int32_t>(),
                              long_scratch.mutable_data_ptr<acc_t>(),
                              num_indices,
                              max_segments,
                              max_long_chunks,
                              inner_size,
                              feature_offset,
                              feature_count);
                          C10_CUDA_KERNEL_LAUNCH_CHECK();
                        }
                        {
                          index_select_backward_long_final_pass_kernel<
                              acc_t,
                              index_t,
                              1><<<long_grid, threads, 0, stream>>>(
                              sorted_indices.const_data_ptr<index_t>(),
                              segment_offsets.const_data_ptr<int32_t>(),
                              long_chunk_counts.const_data_ptr<int32_t>(),
                              long_chunk_offsets.const_data_ptr<int32_t>(),
                              long_scratch.const_data_ptr<acc_t>(),
                              accumulation.mutable_data_ptr<acc_t>(),
                              num_rows,
                              max_segments,
                              max_long_chunks,
                              inner_size,
                              feature_offset,
                              feature_count);
                          C10_CUDA_KERNEL_LAUNCH_CHECK();
                        }
                    }
                  }
                }
              });
          if constexpr (!std::is_same_v<scalar_t, acc_t>) {
            result.copy_(accumulation);
          }
        });
    return result;
  }
#endif

  return result.index_add_(dim, index, grad);
}

Tensor index_select_quantized_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  TORCH_CHECK(
    self.qscheme() == kPerTensorAffine,
    "Only per_tensor quantized quantized tensors are supported by index_select.")
  Tensor out = at::empty_quantized({0}, self);
  at::native::index_select_out_cuda(self, dim, index, out);
  return out;
}

namespace {

void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5(
      kBool, kHalf, kBFloat16, kComplexHalf, kBComplex32, iter.common_dtype(), "masked_fill_", [&]() {
        const auto value_ = value.to<scalar_t>();
        gpu_kernel(
            iter, [value_] GPU_LAMBDA(scalar_t self, bool mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      });
}

template <typename scalar_t>
void cuda_masked_fill_kernel_quantized(TensorIterator& iter, scalar_t quantized_val) {
    gpu_kernel(
        iter, [quantized_val] GPU_LAMBDA(scalar_t self, bool mask) -> scalar_t {
          if (mask) {
            return quantized_val;
          }
          return self;
    });
}

void masked_fill_kernel_quantized(TensorIterator& iter, const Scalar& value, double scale, int zero_point) {
  TORCH_CHECK(iter.input_dtype(1) == at::ScalarType::Bool, "masked_fill only supports boolean masks, ",
    "but got dtype ", iter.input_dtype(1));
  AT_DISPATCH_QINT_TYPES(
      iter.common_dtype(), "masked_fill_", [&]() {
        float float_val = value.to<float>();
        const auto quantized_val = quantize_val<scalar_t>(scale, zero_point, float_val);

        cuda_masked_fill_kernel_quantized<scalar_t>(iter, quantized_val);
    });
}

REGISTER_CUDA_DISPATCH(masked_fill_kernel_quantized_stub, &masked_fill_kernel_quantized)

} // anonymous namespace

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());
  TORCH_CHECK(mask.scalar_type() == kBool,
    "masked_fill only supports boolean masks, but got dtype ", mask.scalar_type());
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_const_input(self)
      .add_const_input(*b_mask)
      .build();

  masked_fill_kernel(iter, value);
  return self;
}

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  // We hit this function if either of the input tensor lives on CUDA.
  // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
  // `mask` to be CPU tensor. Check for `self` and `mask` being on same device
  // exists in `masked_fill__cuda` (Scalar version).
  TORCH_CHECK(!self.device().is_cpu(), "masked_fill_: Expected inputs to be on same device")
  return masked_fill__cuda(self, mask, value.item());
}


Tensor index_select_sparse_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  const auto ndim = self.dim();
  TORCH_CHECK_INDEX(ndim, "index_select() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK_INDEX(
      index.dim() == 1 && index.dtype() == at::kLong && index.options().layout() == at::kStrided,
      "index_select() argument index must be 1-D strided (non-sparse) long-tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  const auto size = self.size(dim);
  const auto sparse_dim = self.sparse_dim();
  const auto dense_dim = self.dense_dim();
  const auto indices = self._indices();
  const auto values = self._values();
  const auto nnz = values.size(0);
  const auto index_len = index.size(0);
  auto res_sizes = self.sizes().vec();
  res_sizes[dim] = index_len;

  // If indexing into sparse dimensions
  if (dim < sparse_dim) {
    const auto make_output = [
      dim, sparse_dim, dense_dim, res_sizes, &self, &indices, &values
    ](
        const Tensor& selected_dim_indices,
        const Tensor& res_dim_indices
    ) -> Tensor {
      auto res_indices = indices.index_select(1, selected_dim_indices);
      res_indices[dim] = res_dim_indices;
      const auto res_values = values.index_select(0, selected_dim_indices);

      return at::_sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim, dense_dim, res_sizes, res_indices, res_values, self.options());
    };

    // short-circuit if index is empty
    if (!index_len) {
      return make_output(index, index);
    }

    const auto nneg_index = [&index, size]() -> Tensor {
      auto nneg_index = at::empty_like(index, at::MemoryFormat::Contiguous);

      auto iter = TensorIteratorConfig()
        .add_output(nneg_index)
        .add_input(index)
        .build();

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_sparse_cuda", [&]() {
          gpu_kernel(iter, [size] GPU_LAMBDA (index_t idx) -> index_t {
              CUDA_KERNEL_ASSERT(idx >= -size && idx < size
                  && "index_select(): index out of bounds");
              return idx < 0 ? idx + size : idx;
          });
      });
      return nneg_index;
    }();

    const auto dim_indices = indices[dim].contiguous();
    const auto idx_nneg_index = at::arange(index_len, nneg_index.options());
    auto idx_dim_indices = at::arange(nnz, dim_indices.options());

    Tensor sorted_dim_indices, argsort_dim_indices;
    std::tie(sorted_dim_indices, argsort_dim_indices) = [&]() -> std::tuple<Tensor, Tensor> {
      if (dim == 0 && self.is_coalesced()) {
        return std::make_tuple(dim_indices, std::move(idx_dim_indices));
      }
      else {
        return dim_indices.sort();
      }
    }();

    Tensor intrsc_counts_nneg_index;
    Tensor intrsc_first_match_nneg_index;
    std::tie(intrsc_counts_nneg_index, intrsc_first_match_nneg_index) = [&]() -> std::tuple<Tensor, Tensor> {
      auto intrsc_counts_nneg_index = at::empty_like(nneg_index);
      auto intrsc_first_match_nneg_index = at::empty_like(nneg_index);

      auto iter = TensorIteratorConfig()
        .add_output(intrsc_first_match_nneg_index)
        .add_input(nneg_index)
        .add_input(idx_nneg_index)
        .build();

      AT_DISPATCH_INDEX_TYPES(nneg_index.scalar_type(), "index_select_sparse_cuda", [&]() {
          index_t* ptr_intrsc_counts_nneg_index = intrsc_counts_nneg_index.mutable_data_ptr<index_t>();
          const index_t* ptr_sorted_dim_indices = sorted_dim_indices.const_data_ptr<index_t>();
          gpu_kernel(
              iter,
              [ptr_intrsc_counts_nneg_index, ptr_sorted_dim_indices, nnz] GPU_LAMBDA (
                index_t idx_val, index_t idx_idx
              ) -> index_t {
                auto* lb = at::cuda::detail::find_bound<const index_t*, index_t, true>(
                  ptr_sorted_dim_indices,
                  ptr_sorted_dim_indices + nnz,
                  idx_val
                );
                auto* ub = at::cuda::detail::find_bound<const index_t*, index_t, false>(
                  ptr_sorted_dim_indices,
                  ptr_sorted_dim_indices + nnz,
                  idx_val
                );
                const auto idx_count = ub - lb;
                ptr_intrsc_counts_nneg_index[idx_idx] = idx_count;

                return lb - ptr_sorted_dim_indices;
              }
          );
      });

      return std::make_tuple(
          std::move(intrsc_counts_nneg_index),
          std::move(intrsc_first_match_nneg_index));
    }();

    // Unavoidable sync since the shape of the result is not known in advance
    auto res_len = intrsc_counts_nneg_index.sum().item<int64_t>();
    // Short-circuit if empty intersection
    if (!res_len) {
      auto empty_idx = at::empty({0}, nneg_index.options());
      return make_output(empty_idx, empty_idx);
    }

    auto [selected_dim_indices, res_dim_indices] = [&]() -> std::tuple<Tensor, Tensor> {
      auto res_dim_indices = at::empty({res_len}, nneg_index.options());
      auto selected_dim_indices = at::empty_like(res_dim_indices);
      auto selected_dim_indices_offsets = intrsc_counts_nneg_index.cumsum(0)
        .sub_(intrsc_counts_nneg_index);

      // Need to have output as TensorIterator does not allow having void lambdas.
      auto dummy_output = at::empty({1}, dim_indices.options()).expand(IntArrayRef({index_len}));
      auto iter = TensorIteratorConfig()
        .add_output(dummy_output)
        // All iterations map to a single element in dummy_output by design,
        // hence removed output memory overlap check.
        .set_check_mem_overlap(false)
        .add_input(idx_nneg_index)
        .add_input(intrsc_counts_nneg_index)
        .add_input(selected_dim_indices_offsets)
        .add_input(intrsc_first_match_nneg_index)
        .build();

      AT_DISPATCH_INDEX_TYPES(nneg_index.scalar_type(), "index_select_sparse_cuda", [&]() {
          index_t* ptr_res_dim_indices = res_dim_indices.mutable_data_ptr<index_t>();
          index_t* ptr_selected_dim_indices = selected_dim_indices.mutable_data_ptr<index_t>();
          const index_t* ptr_argsort_dim_indices = argsort_dim_indices.const_data_ptr<index_t>();
          gpu_kernel(
              iter,
              [ptr_res_dim_indices, ptr_selected_dim_indices, ptr_argsort_dim_indices] GPU_LAMBDA (
                index_t idx_idx, index_t count, index_t offset, index_t first_match
              ) -> index_t {
                index_t* __restrict__ ptr_res_dim_indices_out = ptr_res_dim_indices + offset;
                const index_t* __restrict__ ptr_argsort_dim_indices_in = ptr_argsort_dim_indices + first_match;
                index_t* __restrict__ ptr_selected_dim_indices_out = ptr_selected_dim_indices + offset;
                for (index_t i = 0; i < count; ++i) {
                  *ptr_res_dim_indices_out++ = idx_idx;
                  *ptr_selected_dim_indices_out++ = *ptr_argsort_dim_indices_in++;
                }

                // A dummy return scalar for a dummy output
                return static_cast<index_t>(1);
              }
          );
      });

      return std::make_tuple(
          std::move(selected_dim_indices), std::move(res_dim_indices));
    }();

    return make_output(selected_dim_indices, res_dim_indices);
  }
  // If indexing into dense dimensions
  else {
    // It is sufficient to just perform `index_select` on values
    // if `dim` refers to dense dimensions.
    const auto res_values = values.index_select(dim - sparse_dim + 1, index);

    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, res_sizes, indices, res_values, self.options());
  }
}


} // at::native
