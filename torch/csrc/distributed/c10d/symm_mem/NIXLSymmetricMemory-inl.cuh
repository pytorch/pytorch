#pragma once

// Device-side helpers for the NIXL symmetric memory backend.
// These wrap nixl_device.cuh functions and are only available when
// USE_NIXL_DEVICE_API is defined (requires UCX GPU device headers).

#ifdef USE_NIXL_DEVICE_API

#include <nixl_device.cuh>

namespace c10d {
namespace symmetric_memory {

// One-sided put: copy `size` bytes from local to remote buffer.
// `level` controls the cooperation scope (THREAD/WARP/BLOCK/GRID).
template <nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ inline nixl_status_t nixl_device_put(
    nixlMemViewH local_mvh,
    size_t local_idx,
    size_t local_off,
    nixlMemViewH remote_mvh,
    size_t remote_idx,
    size_t remote_off,
    size_t size,
    unsigned channel = 0) {
  nixlMemViewElem src{local_mvh, local_idx, local_off};
  nixlMemViewElem dst{remote_mvh, remote_idx, remote_off};
  return nixlPut<level>(src, dst, size, channel);
}

// Atomic add on a remote counter (used for signaling).
template <nixl_gpu_level_t level = nixl_gpu_level_t::BLOCK>
__device__ inline nixl_status_t nixl_device_signal(
    nixlMemViewH counter_mvh,
    size_t idx,
    size_t off,
    uint64_t value = 1,
    unsigned channel = 0) {
  nixlMemViewElem counter{counter_mvh, idx, off};
  return nixlAtomicAdd<level>(value, counter, channel);
}

// Get a device-accessible pointer to a remote buffer (when mapped).
__device__ inline void* nixl_device_get_ptr(
    nixlMemViewH mvh,
    size_t index) {
  return nixlGetPtr(mvh, index);
}

// Poll transfer completion status.
template <nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t nixl_device_poll(
    nixlGpuXferStatusH& status) {
  return nixlGpuGetXferStatus<level>(status);
}

} // namespace symmetric_memory
} // namespace c10d

#endif // USE_NIXL_DEVICE_API
