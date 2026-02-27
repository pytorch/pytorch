#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <torch/csrc/distributed/c10d/cuda/AsyncMM.cuh>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.hpp>

#if defined(USE_ROCM) || (defined(CUDART_VERSION) && CUDART_VERSION >= 12030)

#define INT_SWITCH_CASE(name, val, ...) \
  case val: {                           \
    constexpr int name = val;           \
    __VA_ARGS__();                      \
    break;                              \
  }

#define DISPATCH_WORLD_SIZES(world_size, ...)      \
  switch (world_size) {                            \
    INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__); \
    INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__); \
    INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__); \
    default: {                                     \
      constexpr int k_world_size = -1;             \
      __VA_ARGS__();                               \
    }                                              \
  }

#define DISPATCH_WORLD_SIZES_NO_DEFAULT(world_size, ...)                 \
  switch (world_size) {                                                  \
    INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__);                       \
    INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__);                       \
    INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__);                       \
    default: {                                                           \
      TORCH_CHECK(false, "Not implemented for world_size=", world_size); \
    }                                                                    \
  }

#define DISPATCH_ALIGNMENTS_16_8_4(alignment, ...)                    \
  switch (alignment) {                                                \
    INT_SWITCH_CASE(k_alignment, 16, __VA_ARGS__);                    \
    INT_SWITCH_CASE(k_alignment, 8, __VA_ARGS__);                     \
    INT_SWITCH_CASE(k_alignment, 4, __VA_ARGS__);                     \
    default: {                                                        \
      TORCH_CHECK(false, "Not implemented for alignment=", alignment); \
    }                                                                 \
  }

#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__));

namespace {

using namespace c10d::symmetric_memory;

size_t get_and_verify_alignment(const at::Tensor& input, const char* op_name) {
  const size_t min_alignment = std::max(4l, input.element_size());
  // Only check the offset since the multicast address is always at least
  // 128-bit aligned
  const size_t ptr_alignment = at::native::memory::get_alignment(
      static_cast<size_t>(input.storage_offset() * input.element_size()));
  TORCH_CHECK(
      ptr_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input ptr + offset must be at least ",
      min_alignment,
      "-byte aligned.");

  const size_t size_alignment =
      at::native::memory::get_alignment(static_cast<size_t>(input.numel() * input.element_size()));
  TORCH_CHECK(
      size_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input size must be at least ",
      min_alignment,
      "-byte aligned.");
  return std::min(ptr_alignment, size_alignment);
}

void init_elementwise_launch_config(
    size_t numel,
    size_t element_size,
    size_t alignment,
    size_t splits,
    size_t max_num_blocks,
    size_t max_num_threads,
    int& num_blocks,
    int& num_threads,
    int world_size) {
  // Align to preserve alignment in each split
  const size_t aligned_numel = at::round_up(numel, alignment * splits);
  const size_t numel_per_split = aligned_numel / splits;
  const size_t numel_per_thread = alignment / element_size;

  if (numel_per_split <= max_num_threads * numel_per_thread) {
    num_blocks = 1;
    num_threads = at::ceil_div(numel_per_split, numel_per_thread);
    // `sync_remote_blocks` maps threads to peers, so we need to make sure there
    // are enough threads
    num_threads = max(num_threads, world_size);
    num_threads = at::round_up(num_threads, at::cuda::warp_size());
  } else {
    num_blocks = std::min(
        at::ceil_div(numel_per_split, max_num_threads * numel_per_thread),
        max_num_blocks);
    num_threads = max_num_threads;
  }
}

#if !defined(USE_ROCM) //No multi-cast support on ROCm yet
template <typename T, int alignment>
static __global__ void multimem_all_reduce_kernel(
    T* input_mc_ptr,
    size_t numel,
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);

  sync_remote_blocks<false, true>(signal_pads, rank, world_size);
  __syncthreads();

  const size_t numel_per_rank =
      at::round_up(numel, alignment * world_size) / world_size;
  const size_t start = numel_per_rank * rank;

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  for (size_t i = offset; i < numel_per_rank; i += stride) {
    if (start + i >= numel) {
      continue;
    }
    auto vec = multimem_ld_reduce_add<alignment>(input_mc_ptr + start + i);
    multimem_st<alignment>(input_mc_ptr + start + i, vec);
  }

  __syncthreads();
  sync_remote_blocks<true, true>(signal_pads, rank, world_size);
}

at::Tensor multimem_all_reduce_(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      input.is_contiguous(), "multimem_all_reduce_: input must be contiguous.");
  TORCH_CHECK(
      reduce_op == "sum",
      "multimem_all_reduce_: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "multimem_all_reduce_: input must be allocated with empty_strided_p2p().");
  TORCH_CHECK(
      symm_mem->has_multicast_support(),
      "multimem_all_reduce_: multicast support is required.");

  const size_t alignment =
      get_and_verify_alignment(input, "multimem_all_reduce_");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      symm_mem->get_world_size(),
      8,
      1024,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      input.scalar_type(), "multimem_all_reduce_", [&]() {
        DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
          multimem_all_reduce_kernel<scalar_t, k_alignment>
              <<<num_blocks,
                 num_threads,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  reinterpret_cast<scalar_t*>(symm_mem->get_multicast_ptr()) +
                      input.storage_offset(),
                  input.numel(),
                  reinterpret_cast<uint32_t**>(
                      symm_mem->get_signal_pad_ptrs_dev()),
                  symm_mem->get_rank(),
                  symm_mem->get_world_size());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      });
  return input;
}

template <typename T, int alignment>
static __global__ void multimem_one_shot_reduce_kernel(
    T* input_mc_ptr,
    T* output_ptr,
    size_t numel,
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size,
    int64_t root) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);

  sync_remote_blocks<false, true>(signal_pads, rank, world_size);
  __syncthreads();

  if (rank == root) {
    auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
    auto stride = blockDim.x * gridDim.x * numel_per_thread;
    for (size_t i = offset; i < numel; i += stride) {
      auto vec = multimem_ld_reduce_add<alignment>(input_mc_ptr + i);
      at::native::memory::st_vec<alignment>(output_ptr + i, vec);
    }
  }

  __syncthreads();
  sync_remote_blocks<true, false>(signal_pads, rank, world_size);
}

at::Tensor multimem_one_shot_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t root,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      input.is_contiguous(),
      "multimem_one_shot_reduce: input must be contiguous.");
  TORCH_CHECK(
      reduce_op == "sum",
      "multimem_one_shot_reduce: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "multimem_one_shot_reduce: input must be allocated with empty_strided_p2p().");
  TORCH_CHECK(
      symm_mem->has_multicast_support(),
      "multimem_one_shot_reduce: requires multicast support.");

  int rank = symm_mem->get_rank();
  int world_size = symm_mem->get_world_size();
  TORCH_CHECK(
      root >= 0 && root < world_size,
      "multimem_one_shot_reduce: root must be in [0, world_size).")

  if (rank == root) {
    TORCH_CHECK(
        out.is_contiguous(),
        "multimem_one_shot_reduce: output must be contiguous.");
    TORCH_CHECK(
        out.sizes() == input.sizes(),
        "multimem_one_shot_reduce: input/output size mismatch.");
  }

  const size_t alignment =
      get_and_verify_alignment(input, "multimem_one_shot_all_reduce");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      1,
      8,
      1024,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      input.scalar_type(), "multimem_one_shot_all_reduce", [&]() {
        DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
          multimem_one_shot_reduce_kernel<scalar_t, k_alignment>
              <<<num_blocks,
                 num_threads,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  reinterpret_cast<scalar_t*>(symm_mem->get_multicast_ptr()) +
                      input.storage_offset(),
                  out.data_ptr<scalar_t>(),
                  input.numel(),
                  reinterpret_cast<uint32_t**>(
                      symm_mem->get_signal_pad_ptrs_dev()),
                  rank,
                  world_size,
                  root);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      });
  return out;
}

at::Tensor multimem_one_shot_all_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  auto group = c10d::resolve_process_group(group_name);
  int root = group->getRank();  // each rank reduces to itself
  return multimem_one_shot_reduce_out(input, reduce_op, root, group_name, out);
}

at::Tensor multimem_one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(input);
  return multimem_one_shot_all_reduce_out(input, reduce_op, group_name, out);
}

template <int alignment>
static __global__ void multimem_all_gather_kernel(
    char* input_ptr,
    char* output_mc_ptr,
    size_t bytes_per_rank,
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  sync_remote_blocks<false, true>(signal_pads, rank, world_size);
  __syncthreads();

  const size_t start = bytes_per_rank * rank;

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * alignment;
  auto stride = blockDim.x * gridDim.x * alignment;
  for (size_t i = offset; i < bytes_per_rank; i += stride) {
    auto vec = at::native::memory::ld_vec<alignment>(input_ptr + i);
    multimem_st<alignment>(output_mc_ptr + start + i, vec);
  }

  __syncthreads();
  sync_remote_blocks<true, true>(signal_pads, rank, world_size);
}

at::Tensor multimem_all_gather_out(
    const at::Tensor& input,
    std::string group_name,
    at::Tensor out) {
  auto symm_mem = c10d::symmetric_memory::rendezvous(out, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "multimem_all_gather_out: output must be allocated with empty_strided_p2p().");
  TORCH_CHECK(
      symm_mem->has_multicast_support(),
      "multimem_all_gather_out: output must have multicast support.");

  TORCH_CHECK(
      input.is_contiguous(),
      "multimem_all_gather_out: input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(),
      "multimem_all_gather_out: output must be contiguous.");

  TORCH_CHECK(
      input.dim() == out.dim(),
      "multimem_all_gather_out: input/output dimension mismatch.");

  TORCH_CHECK(
      out.sizes()[0] == input.sizes()[0] * symm_mem->get_world_size(),
      "multimem_all_gather_out: out.sizes()[0] must be equal to input.sizes[0] * world_size. (out.sizes():",
      out.sizes(),
      ", input.sizes(): ",
      input.sizes(),
      ", world_size: ",
      symm_mem->get_world_size(),
      ")");

  for (auto d = 1; d < input.dim(); ++d) {
    TORCH_CHECK(
        out.sizes()[d] == input.sizes()[d],
        "multimem_all_gather_out: all non-0th dimension of input and output must match.");
  }

  const size_t alignment =
      get_and_verify_alignment(out, "multimem_all_gather_out");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel() * input.element_size(),
      1,
      alignment,
      1,
      8,
      1024,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());

  DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
    multimem_all_gather_kernel<k_alignment>
        <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<char*>(input.data_ptr()),
            reinterpret_cast<char*>(symm_mem->get_multicast_ptr()) +
                out.storage_offset() * out.element_size(),
            input.numel() * input.element_size(),
            reinterpret_cast<uint32_t**>(symm_mem->get_signal_pad_ptrs_dev()),
            symm_mem->get_rank(),
            symm_mem->get_world_size());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return out;
}

#endif //no multi-cast support on ROCm

// One-shot all-reduce is register-intensive because it stages values loaded
// from peers in registers before performing reduction. Setting the thread
// count to 512 to prevent/alleviate register spill.
constexpr size_t one_shot_all_reduce_max_num_blocks = 24;
constexpr size_t one_shot_all_reduce_max_num_threads = 512;
template <typename T, int alignment, int k_world_size>
static __launch_bounds__(one_shot_all_reduce_max_num_threads) __global__
    void one_shot_all_reduce_kernel(
        T** input_ptrs,
        T* output_ptr,
        T* input_ptr,
        size_t input_offset,
        size_t numel,
        uint32_t** signal_pads,
        size_t rank,
        size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);
  // copy input to shared ptr
  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  if (input_ptr) {
    for (size_t i = offset; i < numel; i += stride) {
      Vec<alignment> vec_st = at::native::memory::ld_vec<alignment>(input_ptr + i);
      at::native::memory::st_vec<alignment>(input_ptrs[rank] + input_offset + i, vec_st);
    }
  }
  // TODO make it sync with one block for no-copy case
  sync_remote_blocks<true, true>(signal_pads, rank, world_size);
  __syncthreads();

  for (size_t i = offset; i < numel; i += stride) {
    auto vec = load_and_reduce<T, alignment, k_world_size>(
        input_ptrs, rank, world_size, input_offset + i);
    at::native::memory::st_vec<alignment>(output_ptr + i, vec);
  }

  __syncthreads();
  sync_remote_blocks<true, false>(signal_pads, rank, world_size);
}

at::Tensor one_shot_all_reduce_out_impl(
    const at::Tensor& input,
    const std::optional<at::Tensor>& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      input.is_contiguous(), "one_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(), "one_shot_all_reduce: output must be contiguous.");
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "one_shot_all_reduce: input/output size mismatch, input.sizes(): ",
      input.sizes(),
      ", output.sizes(): ",
      out.sizes());
  TORCH_CHECK(
      reduce_op == "sum",
      "one_shot_all_reduce: only sum is supported for now.");
  if (local_input.has_value()) {
    TORCH_CHECK(
        local_input->is_contiguous(),
        "one_shot_all_reduce: local input must be contiguous.");
    TORCH_CHECK(
        local_input->numel() <= input.numel(),
        "one_shot_all_reduce: local input size must be smaller than symm buffer size.");
  }
  if (input.numel() == 0) {
    TORCH_CHECK(input.scalar_type() == out.scalar_type());
    return out;
  }
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "one_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment =
      get_and_verify_alignment(input, "one_shot_all_reduce");
  if (local_input.has_value()) {
    const size_t local_alignment =
        get_and_verify_alignment(*local_input, "one_shot_all_reduce");
    TORCH_CHECK(
        alignment == local_alignment,
        "one_shot_all_reduce: local input and symm buffer must have the same alignment.");
  }

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      1,
      one_shot_all_reduce_max_num_blocks,
      one_shot_all_reduce_max_num_threads,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      input.scalar_type(), "one_shot_all_reduce", [&]() {
        DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
          DISPATCH_WORLD_SIZES(symm_mem->get_world_size(), [&]() {
            one_shot_all_reduce_kernel<scalar_t, k_alignment, k_world_size>
                <<<num_blocks,
                   num_threads,
                   0,
                   at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<scalar_t**>(
                        symm_mem->get_buffer_ptrs_dev()),
                    out.data_ptr<scalar_t>(),
                    local_input.has_value() ? local_input->data_ptr<scalar_t>()
                                            : nullptr,
                    input.storage_offset(),
                    input.numel(),
                    reinterpret_cast<uint32_t**>(
                        symm_mem->get_signal_pad_ptrs_dev()),
                    symm_mem->get_rank(),
                    symm_mem->get_world_size());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
        });
      });
  return out;
}

at::Tensor one_shot_all_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  return one_shot_all_reduce_out_impl(
      input, std::nullopt, reduce_op, group_name, out);
}

at::Tensor one_shot_all_reduce_copy_out(
    const at::Tensor& input,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  return one_shot_all_reduce_out_impl(
      input, local_input, reduce_op, group_name, out);
}

at::Tensor one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(input);
  return one_shot_all_reduce_out_impl(
      input, std::nullopt, reduce_op, group_name, out);
}

at::Tensor one_shot_all_reduce_copy(
    const at::Tensor& input,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(local_input);
  return one_shot_all_reduce_out_impl(
      input, local_input, reduce_op, group_name, out);
}

#if defined(USE_ROCM)
constexpr size_t two_shot_all_reduce_max_num_blocks = 64;
constexpr size_t two_shot_all_reduce_max_num_threads = 128;
#else
constexpr size_t two_shot_all_reduce_max_num_blocks = 24;
constexpr size_t two_shot_all_reduce_max_num_threads = 1024;
#endif
template <
    typename T,
    int alignment,
    int k_world_size,
    bool reduce_scatter = false,
    bool split_last_dim = false>
static __launch_bounds__(two_shot_all_reduce_max_num_threads) __global__
    void two_shot_all_reduce_kernel(
        T** input_ptrs,
        T* output_ptr,
        size_t input_offset,
        size_t numel,
        uint32_t** signal_pads,
        size_t rank,
        size_t world_size,
        size_t last_dim_size = 0) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);
  int32_t N_last_dim =
      last_dim_size / world_size; // used only for split_last_dim reduce_scatter
  sync_remote_blocks<false, true>(signal_pads, rank, world_size);
  __syncthreads();

  const size_t numel_per_rank =
      at::round_up(numel, numel_per_thread * world_size) / world_size;
  const size_t start = split_last_dim ? last_dim_size / world_size * rank
                                      : numel_per_rank * rank;

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  for (size_t i = offset; i < numel_per_rank; i += stride) {
    if constexpr (!reduce_scatter) {
      // we call reduce-scatter only with evenly divisible number of elements
      if (start + i >= numel) {
        continue;
      }
    }
    size_t idx = i;
    if constexpr (split_last_dim) {
      idx = i / N_last_dim * last_dim_size + i % N_last_dim;
    }
    auto vec = load_and_reduce<T, alignment, k_world_size>(
        input_ptrs, rank, world_size, input_offset + start + idx);
    // store to local buffer or to output
    if constexpr (reduce_scatter) {
      at::native::memory::st_vec<alignment>(output_ptr + i, vec);
    } else {
      at::native::memory::st_vec<alignment>(input_ptrs[rank] + input_offset + start + i, vec);
    }
  }

  __syncthreads();
  sync_remote_blocks<true, true>(signal_pads, rank, world_size);
  if constexpr (reduce_scatter) {
    return;
  }
  __syncthreads();
  for (size_t i = offset; i < numel_per_rank; i += stride) {
    Vec<alignment> tmp[k_world_size];
#pragma unroll k_world_size
    for (size_t step = 0; step < k_world_size; ++step) {
      size_t remote_rank = (rank + step) % k_world_size;
      size_t remote_start = numel_per_rank * remote_rank;
#if defined (USE_ROCM)
      tmp[step] = at::native::memory::ld_vec<alignment>(
          input_ptrs[remote_rank] + input_offset + min(remote_start + i, numel-1));
#else
      if (remote_start + i >= numel) {
        continue;
      }
      tmp[step] = at::native::memory::ld_vec<alignment>(
          input_ptrs[remote_rank] + input_offset + remote_start + i);
#endif
    }
#pragma unroll k_world_size
    for (size_t step = 0; step < k_world_size; ++step) {
      size_t remote_rank = (rank + step) % k_world_size;
      size_t remote_start = numel_per_rank * remote_rank;
      if (remote_start + i >= numel) {
        continue;
      }
      at::native::memory::st_vec<alignment>(output_ptr + remote_start + i, tmp[step]);
    }
  }
  // need to make sure all blocks exit simultaneously so that the data
  // is not corrupted by the subsequent kernels
  __syncthreads();
  sync_remote_blocks<true, false>(signal_pads, rank, world_size);
}

template <typename T, int alignment, int k_world_size>
static __launch_bounds__(two_shot_all_reduce_max_num_threads) __global__
    void two_shot_all_reduce_kernel_inplace(
        T** input_ptrs,
        size_t input_offset,
        size_t numel,
        uint32_t** signal_pads,
        size_t rank,
        size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);

  sync_remote_blocks<false, true>(signal_pads, rank, world_size);
  __syncthreads();

  const size_t numel_per_rank =
      at::round_up(numel, alignment * world_size) / world_size;
  const size_t start = numel_per_rank * rank;

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  for (size_t i = offset; i < numel_per_rank; i += stride) {
    if (start + i >= numel) {
      continue;
    }
    auto vec = load_and_reduce<T, alignment, k_world_size>(
        input_ptrs, rank, world_size, input_offset + start + i);
    for (size_t step = 0; step < world_size; ++step) {
      size_t remote_rank = (rank + step) % world_size;
      at::native::memory::st_vec<alignment>(
          input_ptrs[remote_rank] + input_offset + start + i, vec);
    }
  }

  __syncthreads();
  sync_remote_blocks<true, true>(signal_pads, rank, world_size);
}

at::Tensor two_shot_all_reduce_impl(
    at::Tensor input,
    std::optional<at::Tensor> output,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      input.is_contiguous(), "two_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      reduce_op == "sum",
      "two_shot_all_reduce: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "two_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment =
      get_and_verify_alignment(input, "two_shot_all_reduce");

  if (output.has_value()) {
    TORCH_CHECK(
        output->is_contiguous(),
        "two_shot_all_reduce: output must be contiguous.");
    const size_t output_alignment =
        get_and_verify_alignment(*output, "two_shot_all_reduce");
    TORCH_CHECK(
        alignment <= output_alignment,
        "two_shot_all_reduce: output alignment must be equal to or larger than input.");
    TORCH_CHECK(
        output->sizes() == input.sizes(),
        "two_shot_all_reduce: input/output size mismatch, input.sizes(): ",
        input.sizes(),
        ", output.sizes(): ",
        output->sizes());
    if (input.numel() == 0) {
      TORCH_CHECK(output->scalar_type() == input.scalar_type());
      return *output;
    }
  } else {
    if (input.numel() == 0) {
      return input;
    }
  }

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      symm_mem->get_world_size(),
      two_shot_all_reduce_max_num_blocks,
      two_shot_all_reduce_max_num_threads,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());

  if (!output.has_value()) {
    AT_DISPATCH_FLOAT_AND_BFLOAT16(
        input.scalar_type(), "two_shot_all_reduce", [&]() {
          DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
            DISPATCH_WORLD_SIZES(symm_mem->get_world_size(), [&]() {
              two_shot_all_reduce_kernel_inplace<
                  scalar_t,
                  k_alignment,
                  k_world_size>
                  <<<num_blocks,
                     num_threads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      reinterpret_cast<scalar_t**>(
                          symm_mem->get_buffer_ptrs_dev()),
                      input.storage_offset(),
                      input.numel(),
                      reinterpret_cast<uint32_t**>(
                          symm_mem->get_signal_pad_ptrs_dev()),
                      symm_mem->get_rank(),
                      symm_mem->get_world_size());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
          });
        });
    return input;
  } else {
    AT_DISPATCH_FLOAT_AND_BFLOAT16(
        input.scalar_type(), "two_shot_all_reduce", [&]() {
          DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
            DISPATCH_WORLD_SIZES_NO_DEFAULT(symm_mem->get_world_size(), [&]() {
              two_shot_all_reduce_kernel<scalar_t, k_alignment, k_world_size>
                  <<<num_blocks,
                     num_threads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      reinterpret_cast<scalar_t**>(
                          symm_mem->get_buffer_ptrs_dev()),
                      output->data_ptr<scalar_t>(),
                      input.storage_offset(),
                      input.numel(),
                      reinterpret_cast<uint32_t**>(
                          symm_mem->get_signal_pad_ptrs_dev()),
                      symm_mem->get_rank(),
                      symm_mem->get_world_size());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
          });
        });
    return *output;
  }
}

at::Tensor two_shot_all_reduce_(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  return two_shot_all_reduce_impl(input, std::nullopt, reduce_op, group_name);
}

at::Tensor two_shot_all_reduce_out(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  return two_shot_all_reduce_impl(input, output, reduce_op, group_name);
}

at::Tensor reduce_scatter_out(
    at::Tensor input,
    std::string group_name,
    bool split_last_dim,
    at::Tensor output) {
  TORCH_CHECK(
      input.is_contiguous(), "reduce_scatter: input must be contiguous.");
  TORCH_CHECK(
      output.is_contiguous(), "reduce_scatter: output must be contiguous.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "reduce_scatter: input must be allocated with empty_strided_p2p().");

  const size_t alignment = get_and_verify_alignment(input, "reduce_scatter");

  const size_t output_alignment =
      get_and_verify_alignment(input, "reduce_scatter");

  TORCH_CHECK(
      input.numel() %
              (symm_mem->get_world_size() *
               (alignment / input.element_size())) ==
          0,
      "expected number of elements to be divisible by world_size * alignment, number of elements ",
      input.numel(),
      " world size ",
      symm_mem->get_world_size(),
      "alignment ",
      alignment);

  if (split_last_dim) {
    TORCH_CHECK(input.dim() == output.dim());
    bool are_equal_except_last = std::equal(
        input.sizes().begin(), input.sizes().end() - 1, output.sizes().begin());
    TORCH_CHECK(
        are_equal_except_last,
        "reduce_scatter expected input and output to have same sizes except in the last dimension");
    TORCH_CHECK(
        output.size(-1) == input.size(-1) / symm_mem->get_world_size(),
        "reduce_scatter expected output last dim size to be input last dim size / world_size");

    TORCH_CHECK(
        input.size(-1) %
                (symm_mem->get_world_size() *
                 (alignment / input.element_size())) ==
            0,
        "expected last dimension to be divisible by world_size * alignment, last dimension ",
        input.size(-1),
        " world size ",
        symm_mem->get_world_size(),
        "alignment ",
        alignment);
  } else {
    TORCH_CHECK(input.dim() == 1, "reduce_scatter expected 1D input");
    TORCH_CHECK(output.dim() == 1, "reduce_scatter expected 1D output");
    TORCH_CHECK(output.numel() == input.numel() / symm_mem->get_world_size());
  }
  if (input.numel() == 0) {
    TORCH_CHECK(input.scalar_type() == output.scalar_type());
    return output;
  }

  TORCH_CHECK(
      output_alignment >= alignment,
      "reduce_scatter: output alignment should be not smaller than input alignment");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      symm_mem->get_world_size(),
      two_shot_all_reduce_max_num_blocks,
      two_shot_all_reduce_max_num_threads,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());
  if (split_last_dim) {
    AT_DISPATCH_FLOAT_AND_BFLOAT16(
        input.scalar_type(), "two_shot_all_reduce", [&]() {
          DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
            DISPATCH_WORLD_SIZES_NO_DEFAULT(symm_mem->get_world_size(), [&]() {
              two_shot_all_reduce_kernel<
                  scalar_t,
                  k_alignment,
                  k_world_size,
                  true,
                  true>
                  <<<num_blocks,
                     num_threads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      reinterpret_cast<scalar_t**>(
                          symm_mem->get_buffer_ptrs_dev()),
                      output.data_ptr<scalar_t>(),
                      input.storage_offset(),
                      input.numel(),
                      reinterpret_cast<uint32_t**>(
                          symm_mem->get_signal_pad_ptrs_dev()),
                      symm_mem->get_rank(),
                      symm_mem->get_world_size(),
                      input.size(-1));
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
          });
        });
  } else {
    AT_DISPATCH_FLOAT_AND_BFLOAT16(
        input.scalar_type(), "two_shot_all_reduce", [&]() {
          DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
            DISPATCH_WORLD_SIZES_NO_DEFAULT(symm_mem->get_world_size(), [&]() {
              two_shot_all_reduce_kernel<
                  scalar_t,
                  k_alignment,
                  k_world_size,
                  true,
                  false>
                  <<<num_blocks,
                     num_threads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      reinterpret_cast<scalar_t**>(
                          symm_mem->get_buffer_ptrs_dev()),
                      output.data_ptr<scalar_t>(),
                      input.storage_offset(),
                      input.numel(),
                      reinterpret_cast<uint32_t**>(
                          symm_mem->get_signal_pad_ptrs_dev()),
                      symm_mem->get_rank(),
                      symm_mem->get_world_size(),
                      input.size(-1));
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
          });
        });
  }
  return output;
}
} // namespace
#elif defined(CUDART_VERSION) && CUDART_VERSION < 12030
namespace {
at::Tensor multimem_all_reduce_(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(false, "multimem_all_reduce_: requires CUDA 12.3+.");
  return input;
}

at::Tensor multimem_one_shot_all_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(false, "multimem_one_shot_all_reduce_out: requires CUDA 12.3+.");
  return out;
}

at::Tensor multimem_one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(false, "multimem_one_shot_all_reduce: requires CUDA 12.3+.");
  return input;
}

at::Tensor multimem_all_gather_out(
    const at::Tensor& input,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(false, "multimem_all_gather_out: requires CUDA 12.3+.");
  return out;
}

at::Tensor one_shot_all_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(false, "one_shot_all_reduce_out: requires CUDA 12.3+.");
  return out;
}

at::Tensor one_shot_all_reduce_copy_out(
    const at::Tensor& input,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(false, "one_shot_all_reduce_copy_out: requires CUDA 12.3+.");
  return out;
}

at::Tensor one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(false, "one_shot_all_reduce: requires CUDA 12.3+.");
  return input;
}

at::Tensor one_shot_all_reduce_copy(
    const at::Tensor& input,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(false, "one_shot_all_reduce_copy: requires CUDA 12.3+.");
  return input;
}

at::Tensor two_shot_all_reduce_(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(false, "two_shot_all_reduce_: requires CUDA 12.3+.");
  return input;
}

at::Tensor two_shot_all_reduce_out(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  TORCH_CHECK(false, "two_shot_all_reduce_out: requires CUDA 12.3+.");
  return output;
}

at::Tensor reduce_scatter_out(
    at::Tensor input,
    std::string group_name,
    bool split_last_dim,
    at::Tensor output) {
  TORCH_CHECK(false, "reduce_scatter_out: requires CUDA 12.3+.");
  return output;
}

at::Tensor multimem_one_shot_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t root,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(false, "multimem_one_shot_reduce_out: requires CUDA 12.3+.");
  return out;
}

} // namespace
#endif // #if defined(CUDART_VERSION) && CUDART_VERSION < 12030

namespace {

at::Tensor memset32_(
    at::Tensor& input,
    int64_t offset,
    int64_t val,
    int64_t count) {
  TORCH_CHECK(
      input.dim() == 1 && input.is_contiguous() &&
          input.scalar_type() == c10::ScalarType::UInt32,
      "symm_mem::memset32_: input must be a flat, contiguous uint32 tensor.");

  TORCH_CHECK(
      offset >= 0,
      "symm_mem::memset32_: offset must be greater than or equal to 0 (got ",
      offset,
      ")");

  TORCH_CHECK(
      count > 0,
      "symm_mem::memset32_: count must be a positive integer (got ",
      count,
      ")");

  TORCH_CHECK(
      val >= 0 &&
          static_cast<size_t>(val) <= std::numeric_limits<uint32_t>::max(),
      "symm_mem::memset32_: val must be in the range of "
      "[0, 4294967295] (uint32_t).")

  TORCH_CHECK(
      offset + count <= input.numel(),
      "symm_mem::memset32_: offset + count (",
      offset + count,
      ") exceeded the numel of the input (",
      input.numel(),
      ")");

  auto addr = reinterpret_cast<uint32_t*>(input.data_ptr()) + offset;
  c10::cuda::CUDAGuard guard(input.device());

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto driver_api = c10::cuda::DriverAPI::get();
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemsetD32Async_(
      reinterpret_cast<CUdeviceptr>(addr),
      val,
      count,
      at::cuda::getCurrentCUDAStream()));
#elif defined(USE_ROCM)
  C10_CUDA_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(addr),
                                   val,
                                   count,
                                   at::cuda::getCurrentCUDAStream()));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
  return input;
}

at::Tensor stream_write_value32_(
    at::Tensor& input,
    int64_t offset,
    int64_t val) {
  TORCH_CHECK(
      input.dim() == 1 && input.is_contiguous() &&
          input.scalar_type() == c10::ScalarType::UInt32,
      "symm_mem::stream_write_value32_: input must be a flat, contiguous "
      "uint32 tensor.");

  TORCH_CHECK(
      offset >= 0,
      "symm_mem::stream_write_value32_: offset must be greater than or "
      "equal to 0 (got ",
      offset,
      ")");

  TORCH_CHECK(
      val >= 0 &&
          static_cast<size_t>(val) <= std::numeric_limits<uint32_t>::max(),
      "symm_mem::stream_write_value32_: "
      "val must be in the range of [0, 4294967295] (uint32_t).")

  TORCH_CHECK(
      offset < input.numel(),
      "symm_mem::stream_write_value32_: offset (",
      offset,
      ") exceeded the numel of the input (",
      input.numel(),
      ")");

  auto addr = reinterpret_cast<uint32_t*>(input.data_ptr()) + offset;
  c10::cuda::CUDAGuard guard(input.device());

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto driver_api = c10::cuda::DriverAPI::get();
  // According to the documentation of CUstreamWriteValue_flags,
  // cuStreamWriteValue32 will provide a memory fence before the write, which
  // has similar semantics to __threadfence_system() but is scoped to the
  // stream rather than a CUDA thread.
  C10_CUDA_DRIVER_CHECK(driver_api->cuStreamWriteValue32_(
      at::cuda::getCurrentCUDAStream(),
      reinterpret_cast<CUdeviceptr>(addr),
      val,
      0));
#elif defined(USE_ROCM)
  C10_CUDA_CHECK(hipStreamWriteValue32(
                                      at::cuda::getCurrentCUDAStream(),
                                      reinterpret_cast<void*>(addr),
                                      val,
                                      0));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
  return input;
}

} // namespace

TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
#if defined(USE_ROCM) || defined(CUDART_VERSION)
  m.impl("one_shot_all_reduce", ::one_shot_all_reduce);
  m.impl("one_shot_all_reduce_out", ::one_shot_all_reduce_out);
  m.impl("one_shot_all_reduce_copy", ::one_shot_all_reduce_copy);
  m.impl("one_shot_all_reduce_copy_out", ::one_shot_all_reduce_copy_out);
  m.impl("two_shot_all_reduce_", ::two_shot_all_reduce_);
  m.impl("two_shot_all_reduce_out", ::two_shot_all_reduce_out);
  m.impl("reduce_scatter_out", ::reduce_scatter_out);

  m.impl("_async_input_mm", c10d::cuda::detail::async_input_mm);
#endif
#if defined(CUDART_VERSION)
  m.impl("multimem_all_reduce_", ::multimem_all_reduce_);

  // NOTE: [multimem_one_shot_all_reduce]
  // multimem.ld_reduce does not guarantee a fixed accumulation order. This
  // means that while multimem_one_shot_all_reduce is faster and has higher
  // numerical accuracy than one_shot_all_reduce, it doesn't guarantee
  // identical results across ranks. There may be use cases that can take
  // advantage of this property, but it should not be used without
  // understanding the caveats.
  m.impl("multimem_one_shot_all_reduce", ::multimem_one_shot_all_reduce);
  m.impl(
      "multimem_one_shot_all_reduce_out", ::multimem_one_shot_all_reduce_out);
  m.impl(
      "multimem_one_shot_reduce_out", ::multimem_one_shot_reduce_out);
  m.impl("multimem_all_gather_out", ::multimem_all_gather_out);
#endif
  m.impl("stream_write_value32_", ::stream_write_value32_);
  m.impl("memset32_", ::memset32_);
}
