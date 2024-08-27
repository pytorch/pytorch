#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

namespace c10d {
namespace symmetric_memory {

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
using HandleType = CUmemGenericAllocationHandle;
#else
using HandleType = void*;
#endif

class CUDASymmetricMemory : public SymmetricMemory {
 public:
  CUDASymmetricMemory(
      std::vector<HandleType> handles,
      size_t block_size,
      std::vector<void*> buffers,
      std::vector<void*> signal_pads,
      HandleType mc_handle,
      void* mc_addr,
      size_t buffer_size,
      int local_device_idx,
      int rank,
      int world_size);

  ~CUDASymmetricMemory() override;

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  size_t get_buffer_size() override;
  size_t get_signal_pad_size() override;

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  at::Tensor get_buffer(
      int rank,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) override;

  void barrier(int channel) override;
  void put_signal(int dst_rank, int channel) override;
  void wait_signal(int src_rank, int channel) override;

  int get_rank() override;
  int get_world_size() override;

 private:
  std::vector<HandleType> handles_;
  size_t block_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  HandleType mc_handle_;
  void* mc_addr_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::optional<std::function<void(void)>> finalizer_;
};

struct Block : public c10::intrusive_ptr_target {
  HandleType handle;
  int device_idx;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  std::string group_name;
  c10::intrusive_ptr<CUDASymmetricMemory> symm_mem = nullptr;

  Block(
      HandleType handle,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t signal_pad_offset,
      const std::string& group_name)
      : handle(handle),
        device_idx(device_idx),
        block_size(block_size),
        buffer_size(buffer_size),
        signal_pad_offset(signal_pad_offset),
        group_name(group_name),
        symm_mem(nullptr) {}
};

class CUDASymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(size_t size, int device_idx, const std::string& group_name)
      override;

  void free(void* ptr) override;
  size_t get_alloc_size(void* ptr) override;
  c10::intrusive_ptr<SymmetricMemory> rendezvous(void* ptr) override;
  bool is_rendezvous_completed(void* ptr) override;
  bool has_multicast_support() override;

 private:
  c10::intrusive_ptr<Block> find_block(void* ptr);

  std::shared_mutex mutex_;
  std::unordered_map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;
};

} // namespace symmetric_memory
} // namespace c10d
