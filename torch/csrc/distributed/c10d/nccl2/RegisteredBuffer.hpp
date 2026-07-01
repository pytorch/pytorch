// Copyright (c) Meta Platforms, Inc. and affiliates.
// RegisteredBuffer -- Lightweight handle for local registered source buffers.
//
// This header is intentionally free of heavy dependencies (no ATen, no Torch)
// so that it can be included from device-side code compiled to LLVM bitcode
// (clang device-only mode) where ATen headers are unavailable.

#pragma once

#include <cstddef>
#include <cstdint>

namespace c10d::nccl2 {

// =============================================================================
// RegisteredBuffer -- Handle for Local Registered Source Buffers
// =============================================================================
//
// Represents a registered local memory region for RMA put operations.
// Created on host via hostWindow.register_local_buffer().
//
// Used by both the host-side virtual interface (TorchCommWindow) and
// device-side kernel code (TorchCommDeviceWindow) without circular includes.

// Maximum number of IBGDA NICs per GPU surfaced through RegisteredBuffer.
// Must match NCCLX_MAX_NICS_PER_GPU in nccl.h and comms::prims::kMaxNicsPerGpu
// -- verified by static_assert at the bridge layer (PipesDeviceBackend.cpp,
// the only translation unit that pulls in all three headers). Increase
// only if a future platform supports > 2 NICs per GPU; update all three
// constants in lockstep.
inline constexpr int kMaxNicsPerGpu = 2;

// Per-NIC RDMA local keys in network byte order. `size` tracks the actual
// NIC count populated by the backend (<= kMaxNicsPerGpu); consumers MUST
// loop bounded by `size` (entries beyond `size` are zeroed but meaningless).
struct LkeyPerDevice {
  uint32_t values[kMaxNicsPerGpu]{};
  int size{0};
};

struct RegisteredBuffer {
  void* base_ptr{nullptr};
  size_t size{0};
  void* backend_window{
      nullptr}; // Backend-specific window handle (e.g., ncclWindow_t)
  // Per-NIC RDMA local keys for IBGDA puts (PipesDeviceBackend). One entry
  // per NIC up to kMaxNicsPerGpu; the device-side put selects
  // lkey_per_device.values[nic] based on the slot. `size` is the actual NIC
  // count populated by the backend (0 for backends that do not use IBGDA,
  // e.g. NCCLDeviceBackend).
  LkeyPerDevice lkey_per_device{};
};

} // namespace c10d::nccl2
