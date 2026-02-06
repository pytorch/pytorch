#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace c10d::nccl_extension {

#define THREADS_PER_BLOCK 512

#ifdef NCCL_HAS_SYMMEM_SUPPORT
__device__ __forceinline__ char* get_remote_ptr(
    void** buffer,  // buffers_dev_
    int peer,  // peer index
    size_t byte_offset  // buffer byte offset, default 0
) {
    char* base = reinterpret_cast<char*>(buffer[peer]);
    return base + byte_offset;
}

__device__ inline void copy_bytes_vec16(
    const char* src_base,
    char* dst_base,
    size_t nbytes,
    size_t tid,
    size_t stride)
{
    if (nbytes == 0) return;

    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src_base);
    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_base);

    // head: try to align both to 16B
    // We can only align both with a small head copy if they share the
    // same offset modulo 16.
    size_t head = 0;
    if ((src_addr & 0xF) == (dst_addr & 0xF)) {
        size_t misalign = src_addr & 0xF;
        if (misalign != 0) {
            size_t to_align = min(nbytes, 16 - misalign);
            if (tid == 0) {
                for (size_t i = 0; i < to_align; ++i) {
                    dst_base[i] = src_base[i];
                }
            }
            head = to_align;
            src_addr += head;
            dst_addr += head;
            src_base += head;
            dst_base += head;
            nbytes -= head;
        }
    }
    if (nbytes == 0) return;

    // If either pointer is still not 16B aligned, we *must not* issue
    // 16B vector loads/stores. Fall back to scalar grid-stride copy.
    if ((src_addr & 0xF) != 0 || (dst_addr & 0xF) != 0) {
        for (size_t i = tid; i < nbytes; i += stride) {
            dst_base[i] = src_base[i];
        }
        return;
    }

    // middle: 16B vectorized copy
    size_t n_vec = nbytes / 16;

    for (size_t vec_idx = tid; vec_idx < n_vec; vec_idx += stride) {
        const char* src_ptr = src_base + vec_idx * 16;
        char* dst_ptr = dst_base + vec_idx * 16;
        auto v = at::native::memory::ld_vec<16>(src_ptr);   // load 16 bytes
        at::native::memory::st_vec<16>(dst_ptr, v);         // store 16 bytes
    }

    // tail: leftover bytes (< 16)
    size_t copied = n_vec * 16;
    size_t tail   = nbytes - copied;
    for (size_t i = tid; i < tail; i += stride) {
        dst_base[copied + i] = src_base[copied + i];
    }
}

__global__ void lsa_put_kernel(
    void** buffer,  // buffers_dev_
    int dst_peer,
    size_t dst_byte_offset,
    const void* src,
    size_t nbytes
) {
    // Calculate index
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Calculate remote dst pointer
    auto dst = get_remote_ptr(buffer, dst_peer, dst_byte_offset);
    const char* src_bytes = reinterpret_cast<const char*>(src);
    copy_bytes_vec16(src_bytes, dst, nbytes, tid, stride);
}

__global__ void lsa_put_signal_kernel(
    void**  buffer,  // buffers_dev_
    void**  signal_pad,  // signal pointer table (uint64_t-based)
    int  dst_peer,
    size_t  dst_byte_offset,  // data target offset (bytes)
    const void*  src,  // local src
    size_t  nbytes,
    unsigned int* blocks_done,  // global counter of blocks done
    uint64_t  signal_value     // value to write
) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // 1) data copy without signal set
    auto dst = get_remote_ptr(buffer, dst_peer, dst_byte_offset);
    const char* src_bytes = reinterpret_cast<const char*>(src);
    copy_bytes_vec16(src_bytes, dst, nbytes, tid, stride);

    __syncthreads();
    // Ensure all global writes from all SMs are visible system-wide
    __threadfence_system();

    // 2) system fence + signal set
    if (threadIdx.x == 0) {
        // This block is done; increment global completion counter
        unsigned int prev = atomicAdd(blocks_done, 1);

        // If this was the last block to finish:
        if (prev == gridDim.x - 1) {
            uint64_t* signal_pad_peer =
            reinterpret_cast<uint64_t*>(signal_pad[dst_peer]);

            // Single-writer: atomicExch is conservative but safe.
            atomicExch(
                reinterpret_cast<unsigned long long*>(signal_pad_peer),
                static_cast<unsigned long long>(signal_value));
        }
    }
}

__global__ void nccl_wait_for_signal_kernel(
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
            __nanosleep(64);
#endif
        }
    }
}
#endif

void nccl_put(at::Tensor& tensor, const int64_t peer) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "put op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto symm_mem = c10d::symmetric_memory::rendezvous(tensor, "0");
  int threads = THREADS_PER_BLOCK;
  int blocks  = (tensor.numel() + threads - 1) / threads;
  c10::cuda::CUDAGuard guard(tensor.device());
  size_t nbytes = tensor.numel() * c10::elementSize(tensor.scalar_type());

  lsa_put_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    symm_mem->get_buffer_ptrs_dev(),
    peer,
    0,
    tensor.data_ptr(),
    nbytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "NCCL symmetric memory is not supported. Requires NCCL >= 2.28.9");
#endif
}

void nccl_wait_for_signal(at::Tensor& sigpad, int64_t signal) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  c10::cuda::CUDAGuard guard(sigpad.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto symm_mem = c10d::symmetric_memory::rendezvous(sigpad, "0");

  // Always use device-side kernel because this function waits for a SPECIFIC signal value.
  // ncclWaitSignal only synchronizes on a channel without checking values, so it's not
  // suitable for this API which expects to wait for signal pad to reach a specific value.
  int cur_rank = symm_mem->get_rank();
  nccl_wait_for_signal_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    symm_mem->get_signal_pad_ptrs_dev(),
    cur_rank,
    signal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "NCCL symmetric memory is not supported. Requires NCCL >= 2.28.9");
#endif
}

void nccl_put_with_signal(at::Tensor& tensor, int64_t signal, int64_t peer) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "put op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto symm_mem = c10d::symmetric_memory::rendezvous(tensor, "0");
  c10::cuda::CUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Always use device-side kernel because this function writes a SPECIFIC signal value.
  // ncclPutSignal expects a channel index (0-7) for the signal parameter, not a signal value,
  // so it's not suitable for this API which needs to write specific values to the signal pad.
  int threads = THREADS_PER_BLOCK;
  int blocks = (tensor.numel() + threads - 1) / threads;
  auto opts = at::TensorOptions()
    .dtype(at::kInt)
    .device(tensor.device());
  at::Tensor blocks_done = at::zeros({1}, opts);
  unsigned int* blocks_done_dev =
    reinterpret_cast<unsigned int*>(blocks_done.data_ptr<int>());
  size_t nbytes = tensor.numel() * c10::elementSize(tensor.scalar_type());

  lsa_put_signal_kernel<<<blocks, threads, 0, stream>>>(
    symm_mem->get_buffer_ptrs_dev(),
    symm_mem->get_signal_pad_ptrs_dev(),
    peer,
    0,
    tensor.data_ptr(),
    nbytes,
    blocks_done_dev,
    signal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "NCCL symmetric memory is not supported. Requires NCCL >= 2.28.9");
#endif
}

#ifdef NCCL_HAS_SYMMEM_SUPPORT
__global__ void lsa_get_kernel(
    void**  buffer,  // buffers_dev_
    int  peer,
    size_t  src_byte_offset, // byte offset inside that peer's buffer
    void*  dst,
    size_t  nbytes
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // remote src pointer
    auto src = get_remote_ptr(buffer, peer, src_byte_offset);
    char* dst_bytes = reinterpret_cast<char*>(dst);
    copy_bytes_vec16(src, dst_bytes, nbytes, tid, stride);
}
#endif

void nccl_get(at::Tensor& tensor, const int64_t peer) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "get op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto symm_mem = c10d::symmetric_memory::rendezvous(tensor, "0");
  c10::cuda::CUDAGuard guard(tensor.device());
  int threads = THREADS_PER_BLOCK;
  int blocks  = (tensor.numel() + threads - 1) / threads;
  size_t nbytes = tensor.numel() * c10::elementSize(tensor.scalar_type());

  lsa_get_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    symm_mem->get_buffer_ptrs_dev(),
    peer,
    0,
    tensor.data_ptr(),
    nbytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "NCCL symmetric memory is not supported. Requires NCCL >= 2.28.9");
#endif
}

bool is_nccl_symmem_available() {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
    return true;
#else
    return false;
#endif
}
} // namespace c10d::nccl_extension


TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
  m.impl("nccl_put", c10d::nccl_extension::nccl_put);
  m.impl("nccl_get", c10d::nccl_extension::nccl_get);
  m.impl("nccl_wait_for_signal", c10d::nccl_extension::nccl_wait_for_signal);
  m.impl("nccl_put_with_signal", c10d::nccl_extension::nccl_put_with_signal);
}
