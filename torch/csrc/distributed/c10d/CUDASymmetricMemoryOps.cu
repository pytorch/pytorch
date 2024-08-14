#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030

#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty_like.h>

#include <torch/library.h>

#include <torch/csrc/distributed/c10d/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/CUDASymmetricMemory.hpp>

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
    int& num_blocks,
    int& num_threads) {
  // Align to preserve alignment in each split
  const size_t aligned_numel = at::round_up(numel, alignment * splits);
  const size_t numel_per_split = aligned_numel / splits;
  const size_t numel_per_thread = alignment / element_size;

  if (numel_per_split <= max_num_threads_per_block * numel_per_thread) {
    num_blocks = 1;
    num_threads = at::round_up(
        at::ceil_div(numel_per_split, numel_per_thread),
        static_cast<size_t>(C10_WARP_SIZE));
  } else {
    num_blocks = std::min(
        at::ceil_div(
            numel_per_split, max_num_threads_per_block * numel_per_thread),
        max_num_blocks);
    num_threads = max_num_threads_per_block;
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

  barrier_and_acquire_previous_kernel_writes(signal_pads, rank, world_size);

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
  // Establish observation order - all writes are in-flight beyond this point.
  barrier(signal_pads, rank, world_size);
  // Establish causality order - all writes are visible to all devices beyond
  // this point.
  __threadfence_system();
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
      num_blocks,
      num_threads);

#define DISPATCH(scalar_t, kernel_alignment)                                   \
  if (alignment == kernel_alignment) {                                         \
    multimem_all_reduce_kernel<scalar_t, kernel_alignment>                     \
        <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(    \
            reinterpret_cast<scalar_t*>(symm_mem->get_multicast_ptr()) +       \
                input.storage_offset(),                                        \
            input.numel(),                                                     \
            reinterpret_cast<uint32_t**>(symm_mem->get_signal_pad_ptrs_dev()), \
            symm_mem->get_rank(),                                              \
            symm_mem->get_world_size());                                       \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                            \
  }

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "multimem_all_reduce",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        DISPATCH(scalar_t, 16);
        DISPATCH(scalar_t, 8);
        DISPATCH(scalar_t, 4);
      }) AT_DISPATCH_CASE(at::kFloat, [&] {
        DISPATCH(scalar_t, 16);
        DISPATCH(scalar_t, 8);
        DISPATCH(scalar_t, 4);
      }));

#undef DISPATCH
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

  barrier_and_acquire_previous_kernel_writes(signal_pads, rank, world_size);

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  for (size_t i = offset; i < numel; i += stride) {
    auto vec = multimem_ld_reduce_add<alignment>(input_mc_ptr + i);
    *reinterpret_cast<decltype(vec.as_scalar)*>(output_ptr + i) = vec.as_scalar;
  }
}

at::Tensor multimem_one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      input.is_contiguous(),
      "multimem_one_shot_all_reduce: input must be contiguous.");
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

  auto output = at::empty_like(input);

  const size_t alignment =
      get_and_verify_alignment(input, "multimem_one_shot_all_reduce");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      1,
      num_blocks,
      num_threads);

#define DISPATCH(scalar_t, kernel_alignment)                                   \
  if (alignment == kernel_alignment) {                                         \
    multimem_one_shot_all_reduce_kernel<scalar_t, kernel_alignment>            \
        <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(    \
            reinterpret_cast<scalar_t*>(symm_mem->get_multicast_ptr()) +       \
                input.storage_offset(),                                        \
            output.data_ptr<scalar_t>(),                                       \
            input.numel(),                                                     \
            reinterpret_cast<uint32_t**>(symm_mem->get_signal_pad_ptrs_dev()), \
            symm_mem->get_rank(),                                              \
            symm_mem->get_world_size());                                       \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                            \
  }

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "multimem_all_reduce",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        DISPATCH(scalar_t, 16);
        DISPATCH(scalar_t, 8);
        DISPATCH(scalar_t, 4);
      }) AT_DISPATCH_CASE(at::kFloat, [&] {
        DISPATCH(scalar_t, 16);
        DISPATCH(scalar_t, 8);
        DISPATCH(scalar_t, 4);
      }));

  return output;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "multimem_all_reduce_(Tensor input, str reduce_op, str group_name) -> Tensor",
      torch::dispatch(c10::DispatchKey::CUDA, ::multimem_all_reduce_),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "multimem_one_shot_all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor",
      torch::dispatch(c10::DispatchKey::CUDA, ::multimem_one_shot_all_reduce),
      {at::Tag::pt2_compliant_tag});
}

} // namespace

#endif
