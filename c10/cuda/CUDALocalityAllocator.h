#pragma once

#include <c10/cuda/CUDAMacros.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Locality-interleaved device memory for GPUs with two memory sides (locality
// domains).
//
// One arena per device: a 4 MiB-aligned VA reservation whose 2 MiB page P is
// backed by locality domain P & 1, so that
//
//   side(addr) = (addr >> 21) & 1.
//
// Blocks are 4 MiB aligned, so element i of same-dtype tensors from the arena
// lies on one side. Kernels can then send each chunk of work to an SM on the
// chunk's side (see ATen/native/cuda/SideAware.cuh).
//
// The arena is exposed as a pluggable allocator (raw_alloc/raw_free have the
// CUDAPluggableAllocator signatures), usable through torch.cuda.MemPool:
//
//   alloc = torch.cuda.memory.LocalityInterleavedAllocator()
//   pool = torch.cuda.MemPool(alloc.allocator())
//   with torch.cuda.use_mem_pool(pool): x = torch.empty(...)
//
// Prototype limits: blocks are rounded up to 4 MiB, freed blocks are reused
// best-fit without coalescing, pages are never unmapped, and peer access/IPC
// are not set up.
namespace c10::cuda::LocalityAllocator {

constexpr size_t kPageBytes = size_t(2) << 20; // one locality-domain page
constexpr size_t kBlockAlign = 2 * kPageBytes; // block bases start on side 0

// The first 2 * kPageBytes of each arena are reserved: page 0 (side 0) and
// page 1 (side 1) serve as probes for the SM side map.
C10_CUDA_API void* raw_alloc(size_t size, int device, cudaStream_t stream);
C10_CUDA_API void raw_free(
    void* ptr,
    size_t size,
    int device,
    cudaStream_t stream);

// True if [ptr, ptr + bytes) lies inside `device`'s arena. Lock-free.
C10_CUDA_API bool contains(int device, const void* ptr, size_t bytes);

// Base of the arena of `device` (its side-0 probe page), or nullptr if the
// arena was never created.
C10_CUDA_API void* arena_base(int device);

} // namespace c10::cuda::LocalityAllocator
