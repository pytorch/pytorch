#include <torch/csrc/distributed/c10d/symm_mem/NIXLSymmetricMemory.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nixl.h>
#include <nixl_descriptors.h>

// Device-side ops and CUDA kernels for the NIXL symmetric memory backend.
// Data transfers use NIXL host-side transfer requests (createXferReq /
// postXferReq / getXferStatus).  Signal polling uses CUDA kernels on the
// local signal pad which is directly addressable VRAM.

namespace c10d {
namespace symmetric_memory {

// ---- CUDA kernels --------------------------------------------------------

// Channel-based wait: spin-CAS on local signal pad (uint32_t, 0/1 protocol).
__global__ void nixl_wait_signal_kernel(
    uint32_t** signal_pads, int src_rank, int channel,
    int rank, int world_size, size_t timeout_ms) {
  if (threadIdx.x != 0) return;
  uint32_t* addr = reinterpret_cast<uint32_t*>(
      reinterpret_cast<char*>(signal_pads[rank]) + kNixlChannelSignalOffset) +
      world_size * channel + src_rank;
  size_t deadline = 0;
  if (timeout_ms != 0) {
#if !defined(USE_ROCM)
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(deadline));
#else
    deadline = clock64();
#endif
    deadline += timeout_ms * 1000000ULL;
  }
  while (true) {
    uint32_t old = atomicCAS(addr, 1u, 0u);
    if (old == 1u) break;
    if (timeout_ms != 0) {
      size_t now;
#if !defined(USE_ROCM)
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
#else
      now = clock64();
#endif
      if (now > deadline) {
        printf("[FATAL] NIXL wait_signal timeout: rank %d waiting for %d ch %d\n",
               rank, src_rank, channel);
#if !defined(USE_ROCM)
        __trap();
#else
        abort();
#endif
      }
    }
  }
}

// Value-based wait: poll first uint64_t of local signal pad for >= target.
__global__ void nixl_wait_for_signal_value_kernel(
    void** signal_pad, int cur_rank, uint64_t target) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  volatile unsigned long long* p =
      reinterpret_cast<volatile unsigned long long*>(
          reinterpret_cast<char*>(signal_pad[cur_rank]) + kNixlValueSignalOffset);
  while (true) {
    unsigned long long v = *p;
    if (v >= static_cast<unsigned long long>(target)) break;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && !defined(USE_ROCM)
    __nanosleep(64);
#endif
  }
}

// ---- kernel launcher (called from NIXLSymmetricMemory.cpp) ---------------

void nixl_launch_wait_signal_kernel(
    void** signal_pads_dev, int src_rank, int channel,
    int rank, int world_size, size_t timeout_ms, int device_idx) {
  c10::cuda::CUDAGuard guard(device_idx);
  nixl_wait_signal_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(signal_pads_dev),
      src_rank, channel, rank, world_size, timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- Op implementations --------------------------------------------------

#define THREADS 512

static void check_nixl_peer(NIXLSymmetricMemory* h, int64_t peer, const char* op) {
  TORCH_CHECK(peer >= 0 && peer < h->get_world_size() && peer != h->get_rank(),
      op, ": invalid peer ", peer);
}

static size_t checked_nbytes(const at::Tensor& t, NIXLSymmetricMemory* h, const char* op) {
  auto offset = h->get_offset();
  auto nbytes = t.numel() * c10::elementSize(t.scalar_type());
  TORCH_CHECK(
      offset <= h->get_buffer_size() && nbytes <= h->get_buffer_size() - offset,
      op,
      ": tensor range [",
      offset,
      ", ",
      offset + nbytes,
      ") exceeds NIXL symmetric buffer size ",
      h->get_buffer_size());
  return nbytes;
}

static void check_value_signal_capacity(NIXLSymmetricMemory* h, const char* op) {
  TORCH_CHECK(
      kNixlValueSignalOffset + sizeof(uint64_t) <= h->get_signal_pad_size(),
      op,
      ": signal pad is too small for NIXL value signaling");
}

void nixl_put(at::Tensor& t, int64_t peer) {
  TORCH_CHECK(t.is_contiguous(), "nixl_put requires contiguous tensor");
  auto sm = c10d::symmetric_memory::rendezvous(t, "0");
  auto* h = dynamic_cast<NIXLSymmetricMemory*>(sm.get());
  TORCH_CHECK(h, "nixl_put requires NIXL backend");
  check_nixl_peer(h, peer, "nixl_put");
  c10::cuda::CUDAGuard guard(t.device());
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  size_t nb = checked_nbytes(t, h, "nixl_put");
  auto peer_rank = static_cast<int>(peer);
  nixl_transfer(NIXL_WRITE,
      reinterpret_cast<uintptr_t>(t.data_ptr()), nb, uint64_t(h->get_local_device_idx()),
      h->get_peer_buffer_addr(peer_rank) + h->get_offset(), nb, uint64_t(h->get_peer_device_idx(peer_rank)),
      h->get_peer_agent_name(peer_rank));
}

void nixl_get(at::Tensor& t, int64_t peer) {
  TORCH_CHECK(t.is_contiguous(), "nixl_get requires contiguous tensor");
  auto sm = c10d::symmetric_memory::rendezvous(t, "0");
  auto* h = dynamic_cast<NIXLSymmetricMemory*>(sm.get());
  TORCH_CHECK(h, "nixl_get requires NIXL backend");
  check_nixl_peer(h, peer, "nixl_get");
  c10::cuda::CUDAGuard guard(t.device());
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  size_t nb = checked_nbytes(t, h, "nixl_get");
  auto peer_rank = static_cast<int>(peer);
  nixl_transfer(NIXL_READ,
      reinterpret_cast<uintptr_t>(t.data_ptr()), nb, uint64_t(h->get_local_device_idx()),
      h->get_peer_buffer_addr(peer_rank) + h->get_offset(), nb, uint64_t(h->get_peer_device_idx(peer_rank)),
      h->get_peer_agent_name(peer_rank));
}

void nixl_put_with_signal(at::Tensor& t, int64_t signal, int64_t peer) {
  TORCH_CHECK(t.is_contiguous(), "nixl_put_with_signal requires contiguous tensor");
  auto sm = c10d::symmetric_memory::rendezvous(t, "0");
  auto* h = dynamic_cast<NIXLSymmetricMemory*>(sm.get());
  TORCH_CHECK(h, "nixl_put_with_signal requires NIXL backend");
  check_nixl_peer(h, peer, "nixl_put_with_signal");
  check_value_signal_capacity(h, "nixl_put_with_signal");
  c10::cuda::CUDAGuard guard(t.device());
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

  // 1) data transfer
  size_t nb = checked_nbytes(t, h, "nixl_put_with_signal");
  auto peer_rank = static_cast<int>(peer);
  nixl_transfer(NIXL_WRITE,
      reinterpret_cast<uintptr_t>(t.data_ptr()), nb, uint64_t(h->get_local_device_idx()),
      h->get_peer_buffer_addr(peer_rank) + h->get_offset(), nb, uint64_t(h->get_peer_device_idx(peer_rank)),
      h->get_peer_agent_name(peer_rank));

  // 2) write signal value to staging buffer, then transfer
  uint64_t sig = static_cast<uint64_t>(signal);
  void* staging = h->get_signal_staging_ptr();
  AT_CUDA_CHECK(cudaMemcpy(
      static_cast<char*>(staging) + kNixlValueSignalOffset,
      &sig,
      sizeof(sig),
      cudaMemcpyHostToDevice));
  nixl_transfer(NIXL_WRITE,
      reinterpret_cast<uintptr_t>(staging) + kNixlValueSignalOffset, sizeof(uint64_t), uint64_t(h->get_local_device_idx()),
      h->get_peer_signal_pad_addr(peer_rank) + kNixlValueSignalOffset, sizeof(uint64_t), uint64_t(h->get_peer_device_idx(peer_rank)),
      h->get_peer_agent_name(peer_rank));
}

void nixl_wait_for_signal(at::Tensor& sigpad, int64_t signal) {
  c10::cuda::CUDAGuard guard(sigpad.device());
  auto sm = c10d::symmetric_memory::rendezvous(sigpad, "0");
  auto* h = dynamic_cast<NIXLSymmetricMemory*>(sm.get());
  TORCH_CHECK(h, "nixl_wait_for_signal requires NIXL backend");
  check_value_signal_capacity(h, "nixl_wait_for_signal");
  int cur = sm->get_rank();
  nixl_wait_for_signal_value_kernel<<<1, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      sm->get_signal_pad_ptrs_dev(), cur, static_cast<uint64_t>(signal));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace symmetric_memory
} // namespace c10d

TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
  m.impl("nixl_put", c10d::symmetric_memory::nixl_put);
  m.impl("nixl_get", c10d::symmetric_memory::nixl_get);
  m.impl("nixl_put_with_signal", c10d::symmetric_memory::nixl_put_with_signal);
  m.impl("nixl_wait_for_signal", c10d::symmetric_memory::nixl_wait_for_signal);
}
