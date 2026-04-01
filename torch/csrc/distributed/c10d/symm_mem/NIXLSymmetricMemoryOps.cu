#include <torch/csrc/distributed/c10d/symm_mem/NIXLSymmetricMemory.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nixl.h>
#include <nixl_descriptors.h>
#include <chrono>
#include <thread>

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
  uint32_t* addr = signal_pads[rank] + world_size * channel + src_rank;
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
      reinterpret_cast<volatile unsigned long long*>(signal_pad[cur_rank]);
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

// ---- NIXL host-side transfer helper (duplicated for TU independence) -----

static void do_nixl_xfer(nixl_xfer_op_t op,
    uintptr_t la, size_t ls, uint64_t ld,
    uintptr_t ra, size_t rs, uint64_t rd,
    const std::string& rn) {
  auto& agent = ensure_nixl_agent();
  nixl_xfer_dlist_t ll(VRAM_SEG); ll.addDesc(nixlBasicDesc(la, ls, ld));
  nixl_xfer_dlist_t rl(VRAM_SEG); rl.addDesc(nixlBasicDesc(ra, rs, rd));
  nixlXferReqH* req = nullptr;
  auto s = agent.createXferReq(op, ll, rl, rn, req);
  TORCH_CHECK(s == NIXL_SUCCESS,
      "NIXL createXferReq failed, status=", static_cast<int>(s));
  s = agent.postXferReq(req);
  TORCH_CHECK(s == NIXL_SUCCESS || s == NIXL_IN_PROG,
      "NIXL postXferReq failed, status=", static_cast<int>(s));
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (agent.getXferStatus(req) == NIXL_IN_PROG) {
    TORCH_CHECK(std::chrono::steady_clock::now() < deadline,
        "NIXL transfer timed out after 30s");
    std::this_thread::yield();
  }
  agent.releaseXferReq(req);
}

// ---- Op implementations --------------------------------------------------

#define THREADS 512

void nixl_put(at::Tensor& t, int64_t peer) {
  TORCH_CHECK(t.is_contiguous(), "nixl_put requires contiguous tensor");
  auto sm = c10d::symmetric_memory::rendezvous(t, "0");
  auto* h = dynamic_cast<NIXLSymmetricMemory*>(sm.get());
  TORCH_CHECK(h, "nixl_put requires NIXL backend");
  TORCH_CHECK(peer >= 0 && peer < h->get_world_size() && peer != h->get_rank(),
      "nixl_put: invalid peer ", peer);
  c10::cuda::CUDAGuard guard(t.device());
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  size_t nb = t.numel() * c10::elementSize(t.scalar_type());
  do_nixl_xfer(NIXL_WRITE,
      reinterpret_cast<uintptr_t>(t.data_ptr()), nb, uint64_t(h->get_local_device_idx()),
      h->get_peer_buffer_addr(peer) + h->get_offset(), nb, uint64_t(h->get_peer_device_idx(peer)),
      h->get_peer_agent_name(peer));
}

void nixl_get(at::Tensor& t, int64_t peer) {
  TORCH_CHECK(t.is_contiguous(), "nixl_get requires contiguous tensor");
  auto sm = c10d::symmetric_memory::rendezvous(t, "0");
  auto* h = dynamic_cast<NIXLSymmetricMemory*>(sm.get());
  TORCH_CHECK(h, "nixl_get requires NIXL backend");
  TORCH_CHECK(peer >= 0 && peer < h->get_world_size() && peer != h->get_rank(),
      "nixl_get: invalid peer ", peer);
  c10::cuda::CUDAGuard guard(t.device());
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  size_t nb = t.numel() * c10::elementSize(t.scalar_type());
  do_nixl_xfer(NIXL_READ,
      reinterpret_cast<uintptr_t>(t.data_ptr()), nb, uint64_t(h->get_local_device_idx()),
      h->get_peer_buffer_addr(peer) + h->get_offset(), nb, uint64_t(h->get_peer_device_idx(peer)),
      h->get_peer_agent_name(peer));
}

void nixl_put_with_signal(at::Tensor& t, int64_t signal, int64_t peer) {
  TORCH_CHECK(t.is_contiguous(), "nixl_put_with_signal requires contiguous tensor");
  auto sm = c10d::symmetric_memory::rendezvous(t, "0");
  auto* h = dynamic_cast<NIXLSymmetricMemory*>(sm.get());
  TORCH_CHECK(h, "nixl_put_with_signal requires NIXL backend");
  TORCH_CHECK(peer >= 0 && peer < h->get_world_size() && peer != h->get_rank(),
      "nixl_put_with_signal: invalid peer ", peer);
  c10::cuda::CUDAGuard guard(t.device());
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

  // 1) data transfer
  size_t nb = t.numel() * c10::elementSize(t.scalar_type());
  do_nixl_xfer(NIXL_WRITE,
      reinterpret_cast<uintptr_t>(t.data_ptr()), nb, uint64_t(h->get_local_device_idx()),
      h->get_peer_buffer_addr(peer) + h->get_offset(), nb, uint64_t(h->get_peer_device_idx(peer)),
      h->get_peer_agent_name(peer));

  // 2) write signal value to staging buffer offset 8, then transfer
  uint64_t sig = static_cast<uint64_t>(signal);
  void* staging = h->get_signal_staging_ptr();
  AT_CUDA_CHECK(cudaMemcpy(static_cast<char*>(staging) + 8, &sig, 8, cudaMemcpyHostToDevice));
  do_nixl_xfer(NIXL_WRITE,
      reinterpret_cast<uintptr_t>(staging) + 8, 8, uint64_t(h->get_local_device_idx()),
      h->get_peer_signal_pad_addr(peer), 8, uint64_t(h->get_peer_device_idx(peer)),
      h->get_peer_agent_name(peer));
}

void nixl_wait_for_signal(at::Tensor& sigpad, int64_t signal) {
  c10::cuda::CUDAGuard guard(sigpad.device());
  auto sm = c10d::symmetric_memory::rendezvous(sigpad, "0");
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
