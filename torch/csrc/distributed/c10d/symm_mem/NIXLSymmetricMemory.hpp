#pragma once

#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <cstdint>
#include <nixl.h>

class nixlAgent;

namespace c10d {
namespace symmetric_memory {

inline constexpr size_t kNixlValueSignalOffset = 0;
inline constexpr size_t kNixlValueSignalBytes = sizeof(uint64_t);
inline constexpr size_t kNixlChannelSignalOffset = kNixlValueSignalBytes;
inline constexpr size_t kNixlSignalStagingBytes = 64;
inline constexpr size_t kNixlTransferTimeoutSeconds = 30;

class NIXLPeerAllocInfo;

class NIXLSymmetricMemory : public SymmetricMemory {
 public:
  NIXLSymmetricMemory(
      c10::intrusive_ptr<NIXLPeerAllocInfo> pai,
      size_t offset);

  // Shallow copy with a different offset, sharing the underlying
  // NIXLPeerAllocInfo. Used when the rendezvous pointer is not the
  // allocation base (e.g., MemPool sub-allocations).
  NIXLSymmetricMemory(const NIXLSymmetricMemory& other, size_t offset);

  NIXLSymmetricMemory(const NIXLSymmetricMemory&) = delete;
  NIXLSymmetricMemory& operator=(const NIXLSymmetricMemory&) = delete;

  ~NIXLSymmetricMemory() override = default;

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  size_t get_buffer_size() override;
  size_t get_offset() override;

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;
  int get_world_size() override;
  c10::Device get_device() override;
  bool world_within_direct_access() override;

  // NIXL-specific accessors for transfer operations (used by Ops.cu)
  const std::string& get_peer_agent_name(int rank) const;
  uintptr_t get_peer_buffer_addr(int rank) const;
  uintptr_t get_peer_signal_pad_addr(int rank) const;
  int get_peer_device_idx(int rank) const;
  int get_local_device_idx() const;

  // Registered VRAM staging buffer used as the source for small signal writes.
  // The value slot is 64-bit aligned; channel signals start after it so the two
  // protocols never alias the same signal-pad bytes.
  void* get_signal_staging_ptr() const;

 private:
  int device_idx_;
  c10::intrusive_ptr<NIXLPeerAllocInfo> pai_;
  size_t offset_{0};
};

// Global NIXL agent singleton (defined in NIXLSymmetricMemory.cpp).
nixlAgent& ensure_nixl_agent();
const std::string& nixl_agent_name();

void nixl_transfer(
    nixl_xfer_op_t op,
    uintptr_t local_addr,
    size_t local_size,
    uint64_t local_device,
    uintptr_t remote_addr,
    size_t remote_size,
    uint64_t remote_device,
    const std::string& remote_agent_name);

// CUDA kernel launcher for channel-based wait_signal.
// Defined in NIXLSymmetricMemoryOps.cu, called from NIXLSymmetricMemory.cpp.
void nixl_launch_wait_signal_kernel(
    void** signal_pads_dev,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    int device_idx);

} // namespace symmetric_memory
} // namespace c10d
