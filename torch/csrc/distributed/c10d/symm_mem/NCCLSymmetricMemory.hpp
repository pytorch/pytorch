#pragma once
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

namespace c10d {
namespace symmetric_memory {

class NCCLPeerAllocInfo;

class NCCLSymmetricMemory : public SymmetricMemory {
 public:
  NCCLSymmetricMemory(c10::intrusive_ptr<NCCLPeerAllocInfo> pai, size_t offset);

  ~NCCLSymmetricMemory() override = default;

  std::vector<void*> get_buffer_ptrs() override;

  std::vector<void*> get_signal_pad_ptrs() override;

  void** get_buffer_ptrs_dev() override;

  void** get_signal_pad_ptrs_dev() override;

  size_t get_buffer_size() override;

  bool has_multicast_support() override;

  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;

  int get_world_size() override;

  c10::Device get_device() override;

  ncclWindow_t get_window();

  ncclWindow_t get_signal_pad_handle();

  size_t get_offset() override;

 private:
  c10::intrusive_ptr<NCCLPeerAllocInfo> pai_;
  size_t offset_;
  int rank_;
  int world_size_;
  int device_idx_;
};

} // namespace symmetric_memory
} // namespace c10d
#endif // NCCL_HAS_SYMMEM_SUPPORT
