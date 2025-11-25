#include <dlfcn.h>
#include <ATen/ceil_div.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/distributed/c10d/symm_mem/env.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <ATen/ceil_div.h>

namespace c10d::nccl_extension {

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32
// #define AT_DISPATCH_CASE_CONVERT(enum_type, scalar_type, ...)               \
//   case enum_type: {                                                         \
//     AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                            \
//     using scalar_t = scalar_type;                                           \
//     return __VA_ARGS__();                                                   \
//   }

#define AT_DISPATCH_NCCL_FLOATS(scalar_type, name, ...)                  \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__));

__device__ __forceinline__ char* get_remote_ptr(
    void** buffer,  // buffers_dev_
    int peer,  // peer index
    size_t byte_offset  // buffer byte offset, default 0
) {
    char* base = reinterpret_cast<char*>(buffer[peer]);
    return base + byte_offset;
}

template <typename T>
__global__ void lsa_put_kernel(
    void** buffer,  // buffers_dev_
    int dst_peer,
    size_t dst_byte_offset,
    const T* src,
    size_t nelems
) {
    // Calculate index
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Calculate remote dst pointer
    T* dst = reinterpret_cast<T*>(
        get_remote_ptr(buffer, dst_peer, dst_byte_offset));

    for (size_t i = tid; i < nelems; i += stride) {
        dst[i] = src[i];
    }
}

template <typename T>
__global__ void lsa_put_signal_kernel(
    void**  buffer,  // buffers_dev_
    void**  signal_pad,  // signal pointer table (uint64_t-based)
    int  dst_peer,
    size_t  dst_byte_offset,  // data target offset (bytes)
    const T*  src,  // local src
    size_t  nelems,
    uint64_t  signal_value     // value to write
) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // 1) data put
    T* dst = reinterpret_cast<T*>(
        get_remote_ptr(buffer, dst_peer, dst_byte_offset));

    for (size_t i = tid; i < nelems; i += stride) {
        dst[i] = src[i];
    }

    __syncthreads();

    // 2) system fence + signal
    if (tid == 0) {
        __threadfence_system();

        uint64_t* signal_pad_peer =
            reinterpret_cast<uint64_t*>(signal_pad[dst_peer]);

        // Single-writer: atomicExch is conservative but safe.
        atomicExch(
            reinterpret_cast<unsigned long long*>(signal_pad_peer),
            static_cast<unsigned long long>(signal_value));
    }
}

__global__ void nccl_wait_for_signal_table_kernel(
    void**  signal_pad,
    int  cur_rank,
    uint64_t  target_signal_value
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        volatile unsigned long long* sig_ptr =
            reinterpret_cast<volatile unsigned long long*>(signal_pad[cur_rank]);

        while (true) {
            unsigned long long val = *sig_ptr;
            if (val >= static_cast<unsigned long long>(target_signal_value)) break;
            __nanosleep(64);
        }
        __threadfence_system();
    }
}

void nccl_put(at::Tensor& tensor, const int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "put op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto symm_mem = c10d::symmetric_memory::rendezvous(tensor, "0");
  int threads = THREADS_PER_BLOCK;
  int blocks  = (tensor.numel() + threads - 1) / threads;
  c10::cuda::CUDAGuard guard(tensor.device());

  AT_DISPATCH_NCCL_FLOATS(tensor.scalar_type(), "nccl_put", [&]() {
    lsa_put_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        symm_mem->get_buffer_ptrs_dev(),
        peer,
        0,
        tensor.data_ptr<scalar_t>(),
        tensor.numel());
  });
}

void nccl_wait_for_signal(at::Tensor& sigpad, int64_t signal, int64_t cur_rank) {
  c10::cuda::CUDAGuard guard(sigpad.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto symm_mem = c10d::symmetric_memory::rendezvous(sigpad, "0");
  AT_DISPATCH_NCCL_FLOATS(sigpad.scalar_type(), "nccl_wait_for_signal", [&]() {
    nccl_wait_for_signal_table_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
        symm_mem->get_signal_pad_ptrs_dev(),
        cur_rank,
        signal);
    });
}

void nccl_put_with_signal(at::Tensor& tensor, int64_t signal, int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "put op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto symm_mem = c10d::symmetric_memory::rendezvous(tensor, "0");
  int threads = THREADS_PER_BLOCK;
  int blocks  = (tensor.numel() + threads - 1) / threads;
  c10::cuda::CUDAGuard guard(tensor.device());

  AT_DISPATCH_NCCL_FLOATS(tensor.scalar_type(), "nccl_put_with_signal", [&]() {
  lsa_put_signal_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    symm_mem->get_buffer_ptrs_dev(),
    symm_mem->get_signal_pad_ptrs_dev(),
    peer,
    0,
    tensor.data_ptr<scalar_t>(),
    tensor.numel(),
    signal);
  });
}

template <typename T>
__global__ void lsa_get_kernel(
    void**  buffer,  // buffers_dev_
    int  peer,
    size_t  src_byte_offset, // byte offset inside that peer's buffer
    T*  dst,
    size_t  nelems
) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // remote src pointer
    const T* src = reinterpret_cast<const T*>(
        get_remote_ptr(buffer, peer, src_byte_offset));

    for (size_t i = tid; i < nelems; i += stride) {
        dst[i] = src[i];
    }
}

void nccl_get(at::Tensor& tensor, const int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "get op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto symm_mem = c10d::symmetric_memory::rendezvous(tensor, "0");
  c10::cuda::CUDAGuard guard(tensor.device());
  int threads = THREADS_PER_BLOCK;
  int blocks  = (tensor.numel() + threads - 1) / threads;

  AT_DISPATCH_NCCL_FLOATS(tensor.scalar_type(), "nccl_get", [&]() {
  lsa_get_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    symm_mem->get_buffer_ptrs_dev(),
    peer,
    0,
    tensor.data_ptr<scalar_t>(),
    tensor.numel());
  });
}
} // namespace c10d::nccl_extension


TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
  m.impl("nccl_put", c10d::nccl_extension::nccl_put);
  m.impl("nccl_get", c10d::nccl_extension::nccl_get);
  m.impl("nccl_wait_for_signal", c10d::nccl_extension::nccl_wait_for_signal);
  m.impl("nccl_put_with_signal", c10d::nccl_extension::nccl_put_with_signal);
}
