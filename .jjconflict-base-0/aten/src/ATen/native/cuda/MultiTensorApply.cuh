#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM)
#include <ATen/record_function.h>
#endif
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
// TensorListMetadata has to fit within the CUDA kernel launch argument limit.
// While CUDA 12.1, driver version R530+ and Volta+ would work with 32KB, we
// decide to be safe and only swap for CUDA 13+ during compile time. This saves
// binary size and will guarantees 32KB kernel arg space; older versions are
// still limited to 4KB. We adopt naive values for 32KB from
// https://github.com/pytorch/pytorch/pull/134373.
// TODO: The values for 32KB can very much be optimized further.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000 && !defined(USE_ROCM)

static constexpr int32_t depth_to_max_tensors[5] = {770, 448, 336, 252, 210};
static constexpr int32_t depth_to_max_blocks[5] =
    {2240, 2240, 2240, 2240, 2240};
static constexpr int32_t depth_to_max_tensors_scalarlist[5] =
    {672, 448, 336, 252, 210};
static constexpr int32_t depth_to_max_tensors_scalarlist_of_complex_double[2] =
    {504, 420};
using block_index_t = uint16_t;

#else

static constexpr int32_t depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
static constexpr int32_t depth_to_max_blocks[5] = {320, 320, 320, 320, 320};
static constexpr int32_t depth_to_max_tensors_scalarlist[5] =
    {96, 64, 48, 36, 30};
static constexpr int32_t depth_to_max_tensors_scalarlist_of_complex_double[2] =
    {72, 60};
using block_index_t = unsigned char;

#endif

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
  static constexpr int32_t max_tensors_per_launch = depth_to_max_tensors[n - 1];
  static constexpr int32_t max_blocks_per_launch = depth_to_max_blocks[n - 1];
  const void* addresses[n][max_tensors_per_launch];
  int64_t numel_for_tensor[max_tensors_per_launch];
  block_index_t block_to_tensor[max_blocks_per_launch];
  int32_t block_to_chunk[max_blocks_per_launch];
  int32_t start_tensor_this_launch;
};

template <typename scalar_vals_t, int n>
struct TensorListScalarListMetadata {
  static constexpr int32_t max_tensors_per_launch =
      depth_to_max_tensors_scalarlist[n - 1];
  static constexpr int32_t max_blocks_per_launch = depth_to_max_blocks[n - 1];
  const void* addresses[n][max_tensors_per_launch];
  int64_t numel_for_tensor[max_tensors_per_launch];
  scalar_vals_t scalar_vals[max_tensors_per_launch];
  block_index_t block_to_tensor[max_blocks_per_launch];
  int32_t block_to_chunk[max_blocks_per_launch];
};

// note(mkozuki): `n` of 1&2 violate the limit of cuda kernel argument size
// with `c10::complex<double>`
template <>
struct TensorListScalarListMetadata<c10::complex<double>, 1> {
  static constexpr int32_t max_tensors_per_launch =
      depth_to_max_tensors_scalarlist_of_complex_double[0];
  static constexpr int32_t max_blocks_per_launch = depth_to_max_blocks[0];
  const void* addresses[1][max_tensors_per_launch];
  int64_t numel_for_tensor[max_tensors_per_launch];
  c10::complex<double> scalar_vals[max_tensors_per_launch];
  block_index_t block_to_tensor[max_blocks_per_launch];
  int32_t block_to_chunk[max_blocks_per_launch];
};

template <>
struct TensorListScalarListMetadata<c10::complex<double>, 2> {
  static constexpr int32_t max_tensors_per_launch =
      depth_to_max_tensors_scalarlist_of_complex_double[1];
  static constexpr int32_t max_blocks_per_launch = depth_to_max_blocks[1];
  const void* addresses[2][max_tensors_per_launch];
  int64_t numel_for_tensor[max_tensors_per_launch];
  c10::complex<double> scalar_vals[max_tensors_per_launch];
  block_index_t block_to_tensor[max_blocks_per_launch];
  int32_t block_to_chunk[max_blocks_per_launch];
};

// NOTE(crcrpar): This is a conservative resolution to handle `state_steps`
// whose each element is `at::Tensor` of 1 element representing the number of
// `step`s called so far.
// We're aware this struct overflows the kernel arg limit at n=1 (4244 bytes),
// but our current fused optimizers only instantiate at n>=4 so it's not a
// concern (yet).
template <int n>
struct FusedOptimizerTensorListMetadata {
  static constexpr int32_t max_tensors_per_launch = depth_to_max_tensors[n - 1];
  static constexpr int32_t max_blocks_per_launch = depth_to_max_blocks[n - 1];
  const void* addresses[n][max_tensors_per_launch];
  int64_t numel_for_tensor[max_tensors_per_launch];
  const void* state_steps_addresses[max_tensors_per_launch];
  block_index_t block_to_tensor[max_blocks_per_launch];
  int32_t block_to_chunk[max_blocks_per_launch];
  int32_t start_tensor_this_launch;
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

inline void record_foreach_mta_launch() {
#if defined(USE_ROCM)
  RECORD_FUNCTION("aten::_foreach_mta_launch", {});
#endif
}

} // namespace

// multi_tensor_apply enables horizontal fusion across lists of tensors.
// For example, whereas you once had a for-loop of a + b = c, where a, b,
// and c are individual tensors in lists as, bs, and cs, you can now with
// fewer kernel launches compute as + bs = cs.
//
// You can also imagine bs to be a scalar list vs a tensor list.
//
// The function below takes in tensor lists, scalars, and a callable and
// chunks up the computation to launch as few kernels as possible by iterating
// through every "chunk" in every tensor (thus the nested for loops). In the
// simplest case, everything gets bundled into just one kernel launch, but
// due to blocksize constraints, we may need to launch multiple kernels.
// Each kernel launch is defined by one tensorListMeta construct, which we
// use to track and reset the necessary metadata for each launch.
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
  using scalar_vals_t = typename T::opmath_t;
  using metadata_t = TensorListScalarListMetadata<scalar_vals_t, depth>;
  metadata_t tensorListMeta;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (size_t t = 0; t < n_tensors; t++) {
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (tensor_lists[0][t].numel() == 0) {
      continue;
    }
    tensorListMeta.scalar_vals[loc_tensor_info] = scalars[t].to<scalar_T>();
    tensorListMeta.numel_for_tensor[loc_tensor_info] =
        tensor_lists[0][t].numel();
    for (int d = 0; d < depth; d++) {
      tensorListMeta.addresses[d][loc_tensor_info] =
          tensor_lists[d][t].const_data_ptr();
    }
    loc_tensor_info++;

    // now we enter [chunking territory].
    // we will launch a kernel when EITHER the blocks get filled up OR
    // the tensors get filled up. There will always be at least one block
    // per tensor since the zero-sized ones will not enter the loop, so
    // the nested forloop within represents iterating through the chunks
    // of a single tensor.
    const auto numel = tensor_lists[0][t].numel();
    const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    for (auto chunk = 0; chunk < chunks; chunk++) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      // a tensor is not considered full unless all its chunks have been
      // processed. Use the metadata struct's own capacities: the
      // c10::complex<double> specializations hold fewer tensors than
      // depth_to_max_tensors_scalarlist to stay within the kernel arg limit.
      const bool tensors_full =
          (loc_tensor_info == metadata_t::max_tensors_per_launch &&
           chunk == chunks - 1);
      const bool blocks_full =
          (loc_block_info == metadata_t::max_blocks_per_launch);

      if (tensors_full || blocks_full) {
        record_foreach_mta_launch();
        multi_tensor_apply_kernel<<<
            loc_block_info,
            kBlockSize,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            tensorListMeta, callable, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Reset.
        loc_block_info = 0;
        // all chunks have already been handled in the kernel
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
        } else { // blocks were full and tensor chunks remain
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

  // note: [finishing what we started]
  // if there's remaining work to be done but the tensors/blocks aren't full
  // yet we are at the end, submit the kernel to do the work!
  if (loc_block_info != 0) {
    record_foreach_mta_launch();
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tensorListMeta, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  using metadata_t = TensorListMetadata<depth>;
  metadata_t tensorListMeta;
  tensorListMeta.start_tensor_this_launch = 0;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  int processed = 0;

  for (size_t t = 0; t < n_tensors; t++) {
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (tensor_lists[0][t].numel() == 0) {
      continue;
    }
    processed++;
    tensorListMeta.numel_for_tensor[loc_tensor_info] =
        tensor_lists[0][t].numel();
    for (int d = 0; d < depth; d++) {
      tensorListMeta.addresses[d][loc_tensor_info] =
          tensor_lists[d][t].const_data_ptr();
    }
    loc_tensor_info++;

    // see note: [chunking territory].
    const auto numel = tensor_lists[0][t].numel();
    const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    for (auto chunk = 0; chunk < chunks; chunk++) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      const bool tensors_full =
          (loc_tensor_info == metadata_t::max_tensors_per_launch &&
           chunk == chunks - 1);
      const bool blocks_full =
          (loc_block_info == metadata_t::max_blocks_per_launch);

      if (tensors_full || blocks_full) {
        record_foreach_mta_launch();
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
          tensorListMeta.start_tensor_this_launch = processed;
        } else {
          tensorListMeta.numel_for_tensor[0] =
              tensorListMeta.numel_for_tensor[loc_tensor_info - 1];
          for (int d = 0; d < depth; d++) {
            tensorListMeta.addresses[d][0] =
                tensorListMeta.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
          tensorListMeta.start_tensor_this_launch = processed - 1;
        }
      }
    }
  }

  // see note: [finishing what we started]
  if (loc_block_info != 0) {
    record_foreach_mta_launch();
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tensorListMeta, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  using metadata_t = FusedOptimizerTensorListMetadata<depth>;
  metadata_t tensorListMeta;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (const auto& tensor_index : c10::irange(num_tensors)) {
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (tensor_lists[0][tensor_index].numel() == 0) {
      continue;
    }
    tensorListMeta.state_steps_addresses[loc_tensor_info] =
        state_steps[tensor_index].const_data_ptr();
    tensorListMeta.numel_for_tensor[loc_tensor_info] =
        tensor_lists[0][tensor_index].numel();
    for (const auto& d : c10::irange(depth)) {
      tensorListMeta.addresses[d][loc_tensor_info] =
          tensor_lists[d][tensor_index].const_data_ptr();
    }
    loc_tensor_info++;

    // see above note: [chunking territory]
    const auto numel = tensor_lists[0][tensor_index].numel();
    const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    TORCH_CHECK(chunks > -1);
    for (const auto& chunk : c10::irange(chunks)) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      const auto tensor_full =
          (loc_tensor_info == metadata_t::max_tensors_per_launch &&
           chunk == chunks - 1);
      const auto blocks_full =
          loc_block_info == metadata_t::max_blocks_per_launch;

      if (tensor_full || blocks_full) {
        record_foreach_mta_launch();
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

  // see above note: [finishing what we've started]
  if (loc_block_info != 0) {
    record_foreach_mta_launch();
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tensorListMeta, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace at::native
