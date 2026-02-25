#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <vector>

// __CUDA_ARCH_LIST__ has the compute capabilities sorted from lowest
// to highest. Thus, we know we are compiling only for Volta or higher
// if the first value of the list is at least 700.

// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-macros
#define __MTA_CUDA_ARCH_LIST_HEAD_IMPL(a, ...) a
#define __MTA_CUDA_ARCH_LIST_HEAD(...) \
  __MTA_CUDA_ARCH_LIST_HEAD_IMPL(__VA_ARGS__)
#if (                              \
    defined(__CUDA_ARCH_LIST__) && \
    __MTA_CUDA_ARCH_LIST_HEAD(__CUDA_ARCH_LIST__) >= 700)
#define __MTA_COMPILE_FOR_VOLTA_AND_HIGHER 1
#else
#define __MTA_COMPILE_FOR_VOLTA_AND_HIGHER 0
#endif

namespace at::native {

namespace {

static constexpr int64_t kILP = 4;
static constexpr int64_t kChunkSize = 65536;
static constexpr int64_t kBlockSize = 512;

// [NOTE: MultiTensorApply parameter size]

// Originally, users could pass only 4 KiB of data to a CUDA kernel
// via the kernel parameters. i.e., the entire of your parameters to a
// kernel must be less than 4 KiB in size (including
// alignment). multi_tensor_apply_kernel was originally written with
// this constraint in mind, but as of 2026, these constraints have
// started to cause poor performance. In particular, only 320 thread
// blocks (depth_to_max_blocks) could be launched per kernel launch,
// and each thread block could work on only kChunkSize values, causing
// poor occupancy.

// Fortunately, the kernel parameter buffer size was increased to 32
// KiB with CUDA 12.1, for Volta and newer GPU's, which allows us to
// drastically improve occupancy. To compute the new maximum number of
// blocks, I simply assumed that the max_tensors would always be 256,
// and that the largest scalar size would be 8 bytes. Then a basic
// linear programming problem can be solved to compute depth_to_max_blocks:
// https://chatgpt.com/s/t_699e2a7d21c881919a49a449de4d54b4

// sizoef(c10::complex<double>) == 16, so I compensate by reducing
// the maximum number of tensors when this causes a parameter buffer
// to be too large.

// Unfortunately, since Pascal GPU's are still supported in Pytorch as
// of today's writing, we cannot simply change the values of
// depth_to_max_blocks. Instead, we must support both code
// paths. Since you current GPU is known only at runtime, we must
// compile for both code paths at build time.

static constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
static constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};
static constexpr int depth_to_max_tensors_scalarlist[5] = {96, 64, 48, 36, 30};
static constexpr int depth_to_max_tensors_scalarlist_of_complex_double[4] = {
    72,
    60,
    60,
    60};

static constexpr int depth_to_max_tensors_large_params[5] =
    {256, 256, 256, 256, 256};
static constexpr int depth_to_max_blocks_large_params[5] =
    {5272, 4863, 4453, 4044, 3634};
static constexpr int depth_to_max_tensors_scalarlist_large_params[5] =
    {256, 256, 256, 256, 256};
static constexpr int
    depth_to_max_tensors_scalarlist_of_complex_double_large_params[4] =
        {194, 182, 170, 158};

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

// using `namespace detail` instead of mta_detail causes an error when
// compiling aten/src/ATen/native/cuda/ForeachReduceOp.cu. The proper
// fix is to get rid of the anonymous namespace, which I assume was
// intended to keep these symbols private, but anonymous namespace
// doesn't actually do that.
namespace mta_detail {

template <bool IS_VOLTA_OR_HIGHER, int depth>
struct MTAConfig {
  static constexpr int max_tensors = IS_VOLTA_OR_HIGHER
      ? depth_to_max_tensors_large_params[depth - 1]
      : depth_to_max_tensors[depth - 1];
  static constexpr int max_blocks = IS_VOLTA_OR_HIGHER
      ? depth_to_max_blocks_large_params[depth - 1]
      : depth_to_max_blocks[depth - 1];
  static constexpr int max_tensors_scalarlist = IS_VOLTA_OR_HIGHER
      ? depth_to_max_tensors_scalarlist_large_params[depth - 1]
      : depth_to_max_tensors_scalarlist[depth - 1];
};

template <bool IS_VOLTA_OR_HIGHER, int depth>
struct MTAComplexDoubleConfig {
  static constexpr int max_tensors = IS_VOLTA_OR_HIGHER
      ? depth_to_max_tensors_scalarlist_of_complex_double_large_params
            [depth - 1]
      : depth_to_max_tensors_scalarlist_of_complex_double[depth - 1];
};

} // namespace mta_detail

template <int n, bool IS_VOLTA_OR_HIGHER = __MTA_COMPILE_FOR_VOLTA_AND_HIGHER>
struct TensorListMetadata {
 private:
  using Config = mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, n>;

 public:
  const void* addresses[n][Config::max_tensors];
  int64_t numel_for_tensor[Config::max_tensors];
  unsigned char block_to_tensor[Config::max_blocks];
  int block_to_chunk[Config::max_blocks];
  int start_tensor_this_launch;
};

template <
    typename scalar_vals_t,
    int n,
    bool IS_VOLTA_OR_HIGHER = __MTA_COMPILE_FOR_VOLTA_AND_HIGHER>
struct TensorListScalarListMetadata {
 private:
  using Config = mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, n>;

 public:
  const void* addresses[n][Config::max_tensors_scalarlist];
  int64_t numel_for_tensor[Config::max_tensors_scalarlist];
  scalar_vals_t scalar_vals[Config::max_tensors_scalarlist];
  unsigned char block_to_tensor[Config::max_blocks];
  int block_to_chunk[Config::max_blocks];
};

// note(mkozuki): On toolchains with a 4 KiB launch parameter limit, `n` of 1&2
// violate the limit with `c10::complex<double>`.
template <bool IS_VOLTA_OR_HIGHER>
struct TensorListScalarListMetadata<
    c10::complex<double>,
    1,
    IS_VOLTA_OR_HIGHER> {
 private:
  using Config = mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, 1>;
  using ComplexDoubleConfig =
      mta_detail::MTAComplexDoubleConfig<IS_VOLTA_OR_HIGHER, 1>;

 public:
  const void* addresses[1][ComplexDoubleConfig::max_tensors];
  int64_t numel_for_tensor[ComplexDoubleConfig::max_tensors];
  c10::complex<double> scalar_vals[ComplexDoubleConfig::max_tensors];
  unsigned char block_to_tensor[Config::max_blocks];
  int block_to_chunk[Config::max_blocks];
};

template <bool IS_VOLTA_OR_HIGHER>
struct TensorListScalarListMetadata<
    c10::complex<double>,
    2,
    IS_VOLTA_OR_HIGHER> {
 private:
  using Config = mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, 2>;
  using ComplexDoubleConfig =
      mta_detail::MTAComplexDoubleConfig<IS_VOLTA_OR_HIGHER, 2>;

 public:
  const void* addresses[2][ComplexDoubleConfig::max_tensors];
  int64_t numel_for_tensor[ComplexDoubleConfig::max_tensors];
  c10::complex<double> scalar_vals[ComplexDoubleConfig::max_tensors];
  unsigned char block_to_tensor[Config::max_blocks];
  int block_to_chunk[Config::max_blocks];
};

template <bool IS_VOLTA_OR_HIGHER>
struct TensorListScalarListMetadata<
    c10::complex<double>,
    3,
    IS_VOLTA_OR_HIGHER> {
 private:
  using Config = mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, 3>;
  using ComplexDoubleConfig =
      mta_detail::MTAComplexDoubleConfig<IS_VOLTA_OR_HIGHER, 3>;

 public:
  const void* addresses[3][ComplexDoubleConfig::max_tensors];
  int64_t numel_for_tensor[ComplexDoubleConfig::max_tensors];
  c10::complex<double> scalar_vals[ComplexDoubleConfig::max_tensors];
  unsigned char block_to_tensor[Config::max_blocks];
  int block_to_chunk[Config::max_blocks];
};

template <bool IS_VOLTA_OR_HIGHER>
struct TensorListScalarListMetadata<
    c10::complex<double>,
    4,
    IS_VOLTA_OR_HIGHER> {
 private:
  using Config = mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, 4>;
  using ComplexDoubleConfig =
      mta_detail::MTAComplexDoubleConfig<IS_VOLTA_OR_HIGHER, 4>;

 public:
  const void* addresses[4][ComplexDoubleConfig::max_tensors];
  int64_t numel_for_tensor[ComplexDoubleConfig::max_tensors];
  c10::complex<double> scalar_vals[ComplexDoubleConfig::max_tensors];
  unsigned char block_to_tensor[Config::max_blocks];
  int block_to_chunk[Config::max_blocks];
};

// NOTE(crcrpar): This is a conservative resolution to handle `state_steps`
// whose each element is `at::Tensor` of 1 element representing the number of
// `step`s called so far.
template <int n, bool IS_VOLTA_OR_HIGHER = false>
struct FusedOptimizerTensorListMetadata {
 private:
  using Config = mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, n>;

 public:
  const void* addresses[n][Config::max_tensors];
  int64_t numel_for_tensor[Config::max_tensors];
  const void* state_steps_addresses[Config::max_tensors_scalarlist];
  unsigned char block_to_tensor[Config::max_blocks];
  int block_to_chunk[Config::max_blocks];
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

inline bool is_current_device_volta_or_higher() {
#if defined(USE_ROCM)
  return false;
#else
  return at::cuda::getCurrentDeviceProperties()->major >= 7;
#endif
}

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
template <
    bool IS_VOLTA_OR_HIGHER,
    int depth,
    typename scalar_T,
    typename T,
    typename... ArgTypes>
void multi_tensor_apply_impl(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();
  using scalar_vals_t = typename T::opmath_t;
  TensorListScalarListMetadata<scalar_vals_t, depth, IS_VOLTA_OR_HIGHER>
      tensorListMeta;

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
      // processed
      const bool tensors_full =
          (loc_tensor_info ==
               mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, depth>::
                   max_tensors_scalarlist &&
           chunk == chunks - 1);
      const bool blocks_full =
          (loc_block_info ==
           mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, depth>::max_blocks);

      if (tensors_full || blocks_full) {
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
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tensorListMeta, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <int depth, typename scalar_T, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
#if __MTA_COMPILE_FOR_VOLTA_AND_HIGHER
  multi_tensor_apply_impl<true, depth, scalar_T>(
      tensor_lists, scalars, callable, args...);
#else
  if (is_current_device_volta_or_higher()) {
    multi_tensor_apply_impl<true, depth, scalar_T>(
        tensor_lists, scalars, callable, args...);
  } else {
    multi_tensor_apply_impl<false, depth, scalar_T>(
        tensor_lists, scalars, callable, args...);
  }
#endif
}

template <bool IS_VOLTA_OR_HIGHER, int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_impl(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();
  TensorListMetadata<depth, IS_VOLTA_OR_HIGHER> tensorListMeta;
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
          (loc_tensor_info ==
               mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, depth>::max_tensors &&
           chunk == chunks - 1);
      const bool blocks_full =
          (loc_block_info ==
           mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, depth>::max_blocks);

      if (tensors_full || blocks_full) {
        multi_tensor_apply_kernel<<<
            loc_block_info, // number of blocks
            kBlockSize, // threads per block
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
#if __MTA_COMPILE_FOR_VOLTA_AND_HIGHER
  multi_tensor_apply_impl<true, depth>(tensor_lists, callable, args...);
#else
  if (is_current_device_volta_or_higher()) {
    multi_tensor_apply_impl<true, depth>(tensor_lists, callable, args...);
  } else {
    multi_tensor_apply_impl<false, depth>(tensor_lists, callable, args...);
  }
#endif
}

template <bool IS_VOLTA_OR_HIGHER, int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_for_fused_optimizer_impl(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");
  const auto num_tensors = tensor_lists[0].size();
  FusedOptimizerTensorListMetadata<depth, IS_VOLTA_OR_HIGHER> tensorListMeta;

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
          (loc_tensor_info ==
               mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, depth>::max_tensors &&
           chunk == chunks - 1);
      const auto blocks_full = loc_block_info ==
          mta_detail::MTAConfig<IS_VOLTA_OR_HIGHER, depth>::max_blocks;

      if (tensor_full || blocks_full) {
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
#if __MTA_COMPILE_FOR_VOLTA_AND_HIGHER
  multi_tensor_apply_for_fused_optimizer_impl<true, depth>(
      tensor_lists, state_steps, callable, args...);
#else
  if (is_current_device_volta_or_higher()) {
    multi_tensor_apply_for_fused_optimizer_impl<true, depth>(
        tensor_lists, state_steps, callable, args...);
  } else {
    multi_tensor_apply_for_fused_optimizer_impl<false, depth>(
        tensor_lists, state_steps, callable, args...);
  }
#endif
}

} // namespace at::native
