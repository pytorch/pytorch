#pragma once

#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

class nixlAgent;

namespace c10d {
namespace symmetric_memory {

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

  // Registered VRAM staging buffer for signal writes.
  // Layout: [0..3] uint32_t(1) for channel-based put_signal,
  //         [8..15] uint64_t slot for value-based put_with_signal.
  void* get_signal_staging_ptr() const;

 private:
  int device_idx_;
  c10::intrusive_ptr<NIXLPeerAllocInfo> pai_;
  size_t offset_{0};
};

// Global NIXL agent singleton (defined in NIXLSymmetricMemory.cpp).
nixlAgent& ensure_nixl_agent();
const std::string& nixl_agent_name();

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
