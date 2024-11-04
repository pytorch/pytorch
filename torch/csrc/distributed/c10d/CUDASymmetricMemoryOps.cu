#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <torch/csrc/distributed/c10d/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/CUDASymmetricMemory.hpp>

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

#define DISPATCH_ALIGNMENTS_16_8_4(alignment, ...)                    \
  switch (alignment) {                                                \
    INT_SWITCH_CASE(k_alignment, 16, __VA_ARGS__);                    \
    INT_SWITCH_CASE(k_alignment, 8, __VA_ARGS__);                     \
    INT_SWITCH_CASE(k_alignment, 4, __VA_ARGS__);                     \
    default: {                                                        \
      TORCH_CHECK(false, "Not implemented for aligment=", alignment); \
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
  const size_t ptr_alignment = get_alignment(
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
      get_alignment(static_cast<size_t>(input.numel() * input.element_size()));
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
    int& num_threads) {
  // Align to preserve alignment in each split
  const size_t aligned_numel = at::round_up(numel, alignment * splits);
  const size_t numel_per_split = aligned_numel / splits;
  const size_t numel_per_thread = alignment / element_size;

  if (numel_per_split <= max_num_threads * numel_per_thread) {
    num_blocks = 1;
    num_threads = at::round_up(
        at::ceil_div(numel_per_split, numel_per_thread),
        static_cast<size_t>(C10_WARP_SIZE));
  } else {
    num_blocks = std::min(
        at::ceil_div(numel_per_split, max_num_threads * numel_per_thread),
        max_num_blocks);
    num_threads = max_num_threads;
  }
}

template <typename T, int alignment>
static __global__ void multimem_all_reduce_kernel(
    T* input_mc_ptr,
    size_t numel,
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
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
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
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

  auto symm_mem = c10d::symmetric_memory::rendezvous(input);
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
      num_threads);

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
static __global__ void multimem_one_shot_all_reduce_kernel(
    T* input_mc_ptr,
    T* output_ptr,
    size_t numel,
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  for (size_t i = offset; i < numel; i += stride) {
    auto vec = multimem_ld_reduce_add<alignment>(input_mc_ptr + i);
    st_vec<alignment>(output_ptr + i, vec);
  }

  __syncthreads();
  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
}

at::Tensor multimem_one_shot_all_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      input.is_contiguous(),
      "multimem_one_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(),
      "multimem_one_shot_all_reduce: output must be contiguous.");
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "multimem_one_shot_all_reduce: input/output size mismatch.");
  TORCH_CHECK(
      reduce_op == "sum",
      "multimem_one_shot_all_reduce: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input);
  TORCH_CHECK(
      symm_mem != nullptr,
      "multimem_one_shot_all_reduce: input must be allocated with empty_strided_p2p().");
  TORCH_CHECK(
      symm_mem->has_multicast_support(),
      "multimem_one_shot_all_reduce: requires multicast support.");

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
      num_threads);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      input.scalar_type(), "multimem_one_shot_all_reduce", [&]() {
        DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
          multimem_one_shot_all_reduce_kernel<scalar_t, k_alignment>
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
                  symm_mem->get_rank(),
                  symm_mem->get_world_size());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      });
  return out;
}

at::Tensor multimem_one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(input);
  return multimem_one_shot_all_reduce_out(input, reduce_op, group_name, out);
}

// One-shot all-reduce is register-intensive because it stages values loaded
// from peers in registers before performing reduction. Setting the thread
// count to 512 to prevent/alleviate register spill.
constexpr size_t one_shot_all_reduce_max_num_blocks = 8;
constexpr size_t one_shot_all_reduce_max_num_threads = 512;

template <typename T, int alignment, int k_world_size>
static __launch_bounds__(one_shot_all_reduce_max_num_threads) __global__
    void one_shot_all_reduce_kernel(
        T** input_ptrs,
        T* output_ptr,
        size_t input_offset,
        size_t numel,
        uint32_t** signal_pads,
        size_t rank,
        size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;

  for (size_t i = offset; i < numel; i += stride) {
    auto vec = load_and_reduce<T, alignment, k_world_size>(
        input_ptrs, rank, world_size, input_offset + i);
    st_vec<alignment>(output_ptr + i, vec);
  }

  __syncthreads();
  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
}

at::Tensor one_shot_all_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      input.is_contiguous(), "one_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(), "one_shot_all_reduce: output must be contiguous.");
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "one_shot_all_reduce: input/output size mismatch.");
  TORCH_CHECK(
      reduce_op == "sum",
      "one_shot_all_reduce: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input);
  TORCH_CHECK(
      symm_mem != nullptr,
      "one_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment =
      get_and_verify_alignment(input, "one_shot_all_reduce");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      1,
      one_shot_all_reduce_max_num_blocks,
      one_shot_all_reduce_max_num_threads,
      num_blocks,
      num_threads);

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

at::Tensor one_shot_all_reduce_meta(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  return at::empty_like(input);
}

at::Tensor one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(input);
  return one_shot_all_reduce_out(input, reduce_op, group_name, out);
}

constexpr size_t two_shot_all_reduce_max_num_blocks = 24;
constexpr size_t two_shot_all_reduce_max_num_threads = 512;

template <typename T, int alignment, int k_world_size>
static __launch_bounds__(two_shot_all_reduce_max_num_threads) __global__
    void two_shot_all_reduce_kernel(
        T** input_ptrs,
        size_t input_offset,
        size_t numel,
        uint32_t** signal_pads,
        size_t rank,
        size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
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
      st_vec<alignment>(
          input_ptrs[remote_rank] + input_offset + start + i, vec);
    }
  }

  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

at::Tensor two_shot_all_reduce_(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      input.is_contiguous(), "two_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      reduce_op == "sum",
      "two_shot_all_reduce: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input);
  TORCH_CHECK(
      symm_mem != nullptr,
      "two_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment =
      get_and_verify_alignment(input, "two_shot_all_reduce");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      symm_mem->get_world_size(),
      two_shot_all_reduce_max_num_blocks,
      two_shot_all_reduce_max_num_threads,
      num_blocks,
      num_threads);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      input.scalar_type(), "two_shot_all_reduce", [&]() {
        DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
          DISPATCH_WORLD_SIZES(symm_mem->get_world_size(), [&]() {
            two_shot_all_reduce_kernel<scalar_t, k_alignment, k_world_size>
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
}

} // namespace
#endif // #if defined(CUDART_VERSION) && CUDART_VERSION >= 12030

namespace {

at::Tensor memset32_(
    at::Tensor& input,
    int64_t offset,
    int64_t val,
    int64_t count) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  TORCH_CHECK(
      input.dim() == 1 && input.is_contiguous() &&
          input.scalar_type() == c10::ScalarType::UInt32,
      "symm_mem::memset32_: input must be a flat, contiguous uint32 tensor.");

  TORCH_CHECK(
      offset > 0 && count > 0,
      "symm_mem::memset32_: offset and count must be positive integers.");

  TORCH_CHECK(
      val >= 0 &&
          static_cast<size_t>(val) <= std::numeric_limits<uint32_t>::max(),
      "symm_mem::memset32_: val must be in the range of "
      "[0, 4294967295] (uint32_t).")

  auto element_size = c10::elementSize(input.scalar_type());
  TORCH_CHECK(
      offset + count < input.numel(),
      "symm_mem::memset32_: offset + count (",
      offset + count,
      ") exceeded the numel of the input (",
      input.numel(),
      ")");

  auto addr = reinterpret_cast<uint32_t*>(input.data_ptr()) + offset;

  c10::cuda::CUDAGuard guard(input.device());
  auto driver_api = c10::cuda::DriverAPI::get();
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemsetD32Async_(
      reinterpret_cast<CUdeviceptr>(addr),
      val,
      count,
      at::cuda::getCurrentCUDAStream()));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
  return input;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
  m.def(
      "multimem_all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)",
      torch::dispatch(c10::DispatchKey::CUDA, ::multimem_all_reduce_),
      {at::Tag::pt2_compliant_tag});

  // NOTE: [multimem_one_shot_all_reduce]
  // multimem.ld_reduce does not guarantee a fixed accumulation order. This
  // means that while multimem_one_shot_all_reduce is faster and has higher
  // numerical accuracy than one_shot_all_reduce, it doesn't guarantee
  // identical results across ranks. There may be use cases that can take
  // advantage of this property, but it should not be used without
  // understanding the caveats.
  m.def(
      "multimem_one_shot_all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor",
      torch::dispatch(c10::DispatchKey::CUDA, ::multimem_one_shot_all_reduce),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "multimem_one_shot_all_reduce_out(Tensor input, str reduce_op, str group_name, Tensor(a!) out) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CUDA, ::multimem_one_shot_all_reduce_out),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "one_shot_all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor",
      {at::Tag::pt2_compliant_tag});

  m.impl(
      "one_shot_all_reduce",
      torch::dispatch(c10::DispatchKey::Meta, ::one_shot_all_reduce_meta));
  m.impl(
      "one_shot_all_reduce",
      torch::dispatch(c10::DispatchKey::CUDA, ::one_shot_all_reduce));

  m.def(
      "one_shot_all_reduce_out(Tensor input, str reduce_op, str group_name, Tensor(a!) out) -> Tensor(a!)",
      torch::dispatch(c10::DispatchKey::CUDA, ::one_shot_all_reduce_out),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "two_shot_all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)",
      torch::dispatch(c10::DispatchKey::CUDA, ::two_shot_all_reduce_),
      {at::Tag::pt2_compliant_tag});
#endif
  m.def(
      "memset32_(Tensor(a!) input, int offset, int val, int count) -> Tensor(a!)",
      torch::dispatch(c10::DispatchKey::CUDA, ::memset32_),
      {at::Tag::pt2_compliant_tag});
}
