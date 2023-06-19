#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <vector>

namespace at::native {

namespace {

static constexpr int64_t kILP = 4;
static constexpr int64_t kChunkSize = 65536;
static constexpr int64_t kBlockSize = 512;

// TODO(crcrpar): Add `n>5` for `low prec params & their higher prec copy`
// TensorListMetadata has to be < 4KB - the limit for kernel launch argument
static constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
static constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};
static constexpr int depth_to_max_tensors_scalarlist[5] = {96, 64, 48, 36, 30};
static constexpr int depth_to_max_tensors_scalarlist_of_complex_double[2] = {
    72,
    60};

template <typename T>
__device__ __forceinline__ bool is_aligned(T* p) {
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(
    T* dst,
    T* src,
    int64_t dst_offset,
    int64_t src_offset) {
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <int n>
struct TensorListMetadata {
  const void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <typename scalar_vals_t, int n>
struct TensorListScalarListMetadata {
  const void* addresses[n][depth_to_max_tensors_scalarlist[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors_scalarlist[n - 1]];
  scalar_vals_t scalar_vals[depth_to_max_tensors_scalarlist[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
};

// note(mkozuki): `n` of 1&2 violate the limit of cuda kernel argument size of
// 4kb with `c10::complex<double>`
template <>
struct TensorListScalarListMetadata<c10::complex<double>, 1> {
  const void* addresses[1]
                       [depth_to_max_tensors_scalarlist_of_complex_double[0]];
  int64_t
      numel_for_tensor[depth_to_max_tensors_scalarlist_of_complex_double[0]];
  c10::complex<double>
      scalar_vals[depth_to_max_tensors_scalarlist_of_complex_double[0]];
  unsigned char block_to_tensor[depth_to_max_blocks[1 - 1]];
  int block_to_chunk[depth_to_max_blocks[1 - 1]];
};

template <>
struct TensorListScalarListMetadata<c10::complex<double>, 2> {
  const void* addresses[2]
                       [depth_to_max_tensors_scalarlist_of_complex_double[1]];
  int64_t
      numel_for_tensor[depth_to_max_tensors_scalarlist_of_complex_double[1]];
  c10::complex<double>
      scalar_vals[depth_to_max_tensors_scalarlist_of_complex_double[1]];
  unsigned char block_to_tensor[depth_to_max_blocks[2 - 1]];
  int block_to_chunk[depth_to_max_blocks[2 - 1]];
};

// NOTE(crcrpar): This is a conservative resolution to handle `state_steps`
// whose each element is `at::Tensor` of 1 element representing the number of
// `step`s called so far.
template <int n>
struct FusedOptimizerTensorListMetadata {
  const void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors[n - 1]];
  const void* state_steps_addresses[depth_to_max_tensors_scalarlist[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <typename T, typename U, typename... ArgTypes>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void multi_tensor_apply_kernel(
    T tensorListMeta,
    U callable,
    ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however
  // it likes.
  callable(kChunkSize, tensorListMeta, args...);
}

} // namespace

// NOTE(crcrpar): [Handling of zero-size tensors]
// Basically we want to skip any index whose tesor(s) are zero-size
// but if it's the last index, we have to check if there are any non-zero
// tensors in order to avoid not applying any math.
template <int depth, typename scalar_T, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();
  size_t n_zero_tensors = 0;
  using scalar_vals_t = typename T::opmath_t;
  TensorListScalarListMetadata<scalar_vals_t, depth> tensorListMeta;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (size_t t = 0; t < n_tensors; t++) {
    if (tensor_lists[0][t].numel() == 0) {
      n_zero_tensors++;
      if (t != n_tensors - 1) {
        continue;
      }
    } else {
      tensorListMeta.scalar_vals[loc_tensor_info] = scalars[t].to<scalar_T>();
      tensorListMeta.numel_for_tensor[loc_tensor_info] =
          tensor_lists[0][t].numel();
      for (int d = 0; d < depth; d++) {
        tensorListMeta.addresses[d][loc_tensor_info] =
            tensor_lists[d][t].const_data_ptr();
      }
      loc_tensor_info++;
    }

    if (n_zero_tensors == n_tensors) {
      continue;
    }
    const auto chunks = (tensor_lists[0]
                                     [t -
                                      static_cast<size_t>(
                                          (t == n_tensors - 1) &&
                                          (tensor_lists[0][t].numel() == 0))]
                                         .numel() +
                         kChunkSize - 1) /
        kChunkSize;
    for (auto chunk = 0; chunk < chunks; chunk++) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      const bool tensors_full =
          (loc_tensor_info == depth_to_max_tensors_scalarlist[depth - 1] &&
           chunk == chunks - 1);
      const bool blocks_full =
          (loc_block_info == depth_to_max_blocks[depth - 1]);
      const bool last_chunk = (t == n_tensors - 1 && chunk == chunks - 1);

      if (tensors_full || blocks_full || last_chunk) {
        multi_tensor_apply_kernel<<<
            loc_block_info,
            kBlockSize,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            tensorListMeta, callable, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Reset.
        loc_block_info = 0;
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
        } else {
          tensorListMeta.numel_for_tensor[0] =
              tensorListMeta.numel_for_tensor[loc_tensor_info - 1];
          tensorListMeta.scalar_vals[0] =
              tensorListMeta.scalar_vals[loc_tensor_info - 1];
          for (int d = 0; d < depth; d++) {
            tensorListMeta.addresses[d][0] =
                tensorListMeta.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
        }
      }
    }
  }
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();
  size_t n_zero_tensors = 0;
  TensorListMetadata<depth> tensorListMeta;
  tensorListMeta.start_tensor_this_launch = 0;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (size_t t = 0; t < n_tensors; t++) {
    if (tensor_lists[0][t].numel() == 0) {
      n_zero_tensors++;
      if (t != n_tensors - 1) {
        continue;
      }
    } else {
      tensorListMeta.numel_for_tensor[loc_tensor_info] =
          tensor_lists[0][t].numel();
      for (int d = 0; d < depth; d++) {
        tensorListMeta.addresses[d][loc_tensor_info] =
            tensor_lists[d][t].const_data_ptr();
      }
      loc_tensor_info++;
    }

    if (n_zero_tensors == n_tensors) {
      continue;
    }
    const auto chunks = (tensor_lists[0]
                                     [t -
                                      static_cast<size_t>(
                                          (t == n_tensors - 1) &&
                                          (tensor_lists[0][t].numel() == 0))]
                                         .numel() +
                         kChunkSize - 1) /
        kChunkSize;
    for (auto chunk = 0; chunk < chunks; chunk++) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      const bool tensors_full =
          (loc_tensor_info == depth_to_max_tensors[depth - 1] &&
           chunk == chunks - 1);
      const bool blocks_full =
          (loc_block_info == depth_to_max_blocks[depth - 1]);
      const bool last_chunk = (t == n_tensors - 1 && chunk == chunks - 1);

      if (tensors_full || blocks_full || last_chunk) {
        multi_tensor_apply_kernel<<<
            loc_block_info,
            kBlockSize,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            tensorListMeta, callable, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Reset.
        loc_block_info = 0;
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
          tensorListMeta.start_tensor_this_launch = t + 1;
        } else {
          tensorListMeta.numel_for_tensor[0] =
              tensorListMeta.numel_for_tensor[loc_tensor_info - 1];
          for (int d = 0; d < depth; d++) {
            tensorListMeta.addresses[d][0] =
                tensorListMeta.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
          tensorListMeta.start_tensor_this_launch = t;
        }
      }
    }
  }
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_for_fused_optimizer(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");
  const auto num_tensors = tensor_lists[0].size();
  auto num_zero_tensors = 0;
  FusedOptimizerTensorListMetadata<depth> tensorListMeta;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (const auto& tensor_index : c10::irange(num_tensors)) {
    if (tensor_lists[0][tensor_index].numel() == 0) {
      num_zero_tensors++;
      if (tensor_index != num_tensors - 1) {
        continue;
      }
    } else {
      tensorListMeta.state_steps_addresses[loc_tensor_info] =
          state_steps[tensor_index].const_data_ptr();
      tensorListMeta.numel_for_tensor[loc_tensor_info] =
          tensor_lists[0][tensor_index].numel();
      for (const auto& d : c10::irange(depth)) {
        tensorListMeta.addresses[d][loc_tensor_info] =
            tensor_lists[d][tensor_index].const_data_ptr();
      }
      loc_tensor_info++;
    }

    if (static_cast<decltype(num_tensors)>(num_zero_tensors) == num_tensors) {
      continue;
    }
    const int64_t chunks =
        (tensor_lists[0]
                     [tensor_index -
                      static_cast<decltype(tensor_index)>(
                          (tensor_index == num_tensors - 1) &&
                          (tensor_lists[0][tensor_index].numel() == 0))]
                         .numel() +
         kChunkSize - 1) /
        kChunkSize;
    TORCH_CHECK(chunks > -1);
    for (const auto& chunk : c10::irange(chunks)) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      const auto tensor_full =
          (loc_tensor_info == depth_to_max_tensors[depth - 1] &&
           chunk == chunks - 1);
      const auto blocks_full = loc_block_info == depth_to_max_blocks[depth - 1];
      const auto last_chunk =
          (tensor_index == num_tensors - 1 && chunk == chunks - 1);

      if (tensor_full || blocks_full || last_chunk) {
        multi_tensor_apply_kernel<<<
            loc_block_info,
            kBlockSize,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            tensorListMeta, callable, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Reset.
        loc_block_info = 0;
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
        } else {
          tensorListMeta.numel_for_tensor[0] =
              tensorListMeta.numel_for_tensor[loc_tensor_info - 1];
          tensorListMeta.state_steps_addresses[0] =
              tensorListMeta.state_steps_addresses[loc_tensor_info - 1];
          for (const auto& d : c10::irange(depth)) {
            tensorListMeta.addresses[d][0] =
                tensorListMeta.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
        }
      }
    }
  }
}

} // namespace at::native
