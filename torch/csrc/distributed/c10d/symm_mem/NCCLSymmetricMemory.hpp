#pragma once
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

namespace c10d {
namespace symmetric_memory {

struct NCCLAllocation;

class NCCLSymmetricMemory : public SymmetricMemory {
 public:
  NCCLSymmetricMemory(
      NCCLAllocation* allocation,
      const std::string& group_name,
      ncclWindow_t buffer_handle,
      ncclWindow_t signal_handle);

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

  const std::vector<int>& get_rank_to_global_rank() override;

  int* get_rank_to_global_rank_dev() override;

  ncclWindow_t get_window();

  ncclWindow_t get_signal_pad_handle();

 private:
  size_t buffer_size_;
  int device_idx_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::string group_name_;
  ncclWindow_t buffer_win_;
  ncclWindow_t signal_handle_;
  std::vector<int> rank_to_global_rank_;
};

} // namespace symmetric_memory
} // namespace c10d
#endif // NCCL_HAS_SYMMEM_SUPPORT
