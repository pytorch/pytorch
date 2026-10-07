#pragma once

#include <cstddef>
#include <cstdint>

namespace c10d::symmetric_memory {

// Launchers for the signal-pad synchronization kernels shared with the CUDA
// and NCCL backends (CUDASymmetricMemory-inl.cuh). They live in a separate
// translation unit with minimal includes because NVSHMEMSymmetricMemory.cpp
// is host-compiled: its header mix triggers nvcc frontend internal errors
// when compiled as CUDA. Each launcher performs the shared check_channel()
// validation; callers are responsible for setting the device and validating
// peer reachability.
void launch_barrier_kernel(
    uint32_t** signal_pads_dev,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms);

void launch_put_signal_kernel(
    uint32_t** signal_pads_dev,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms);

void launch_wait_signal_kernel(
    uint32_t** signal_pads_dev,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms);

} // namespace c10d::symmetric_memory
