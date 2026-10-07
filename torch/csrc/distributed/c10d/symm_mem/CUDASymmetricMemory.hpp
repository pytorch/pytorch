#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SignalPad.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <shared_mutex>

namespace c10d::symmetric_memory {

// Resource wrapper that owns a (vaddr, allocation handle) pair. Upon
// destruction, it unmaps the vaddr and releases the allocation handle.
struct AllocationRef : public c10::intrusive_ptr_target {
  void* ptr;
  HandleType handle;
  size_t block_size;
  int device_idx;
  bool is_multicast;

  AllocationRef(
      void* ptr,
      HandleType handle,
      size_t block_size,
      int device_idx,
      bool is_multicast = false);

  ~AllocationRef();
};

// Forward declaration of CUDAPeerAllocInfo
class CUDAPeerAllocInfo;

class CUDASymmetricMemory : public SymmetricMemory {
 public:
  // This is mostly a shallow copy that shares the pointer to
  // `CUDAPeerAllocInfo` which corresponds to the base Block. The
  // CUDASymmetricMemory handle is specified by the offset to the base ptr.
  CUDASymmetricMemory(
      const c10::intrusive_ptr<CUDAPeerAllocInfo>& pai,
      size_t offset);

  ~CUDASymmetricMemory() override {};

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  void** get_group_signal_pad_ptrs_dev() override;
  size_t get_group_signal_pad_size() override;
  size_t get_buffer_size() override;
  size_t get_offset() override;

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;
  std::string get_group_name() override;
  int get_world_size() override;
  c10::Device get_device() override;
  bool world_within_direct_access() override;

 private:
  int local_device_idx_;
  int rank_;
  int world_size_;
  c10::intrusive_ptr<CUDAPeerAllocInfo> pai_;
  size_t offset_{0}; // in byte
};

// A class to hold the base pointers and signal pad pointers for a group of
// peers. One `CUDAPeerAllocInfo` object can be shared by multiple
// `CUDASymmetricMemory` objects when latter reside on the same allocation
// and rendezvous over the same group. (The `CUDASymmetricMemory` objects may
// have different offsets compared to the base address.)
class CUDAPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  CUDAPeerAllocInfo(
      std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
      std::vector<void*> buffers,
      std::vector<void*> signal_pads,
      uint64_t peer_mapping_id,
      HandleType mc_handle,
      void* mc_addr,
      size_t buffer_size,
      int local_device_idx,
      int rank,
      int world_size,
      std::string group_name);

 private:
  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  // Identifies the device of every rank at rendezvous.
  uint64_t peer_mapping_id_;
  HandleType mc_handle_;
  void* mc_addr_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::string group_name_;
  // The group's internal pad, which PyTorch's own operations use, set by
  // rendezvous() before the info is shared with any handle. Unset only on the
  // info that maps a pad itself.
  std::shared_ptr<const SignalPad> pad_;

  friend class CUDASymmetricMemory;
  friend class CUDASymmetricMemoryAllocator;
};

// Metadata associated with each allocation performed by
// `CUDASymmetricMemoryAllocator`.
struct Block : public c10::intrusive_ptr_target {
  c10::intrusive_ptr<AllocationRef> alloc_ref;
  int device_idx;
  size_t block_size;
  size_t buffer_size;
  // Byte offset from the allocation base (alloc_ref->ptr) to the start of the
  // user buffer; the signal pad occupies [0, buffer_offset).
  size_t buffer_offset;
  std::optional<std::string> default_group_name;
  std::map<std::string, c10::intrusive_ptr<CUDAPeerAllocInfo>> symm_mems;

  Block(
      c10::intrusive_ptr<AllocationRef> alloc_ref,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t buffer_offset,
      const std::optional<std::string>& group_name);
};

class CUDASymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override;

  void free(void* ptr) override;
  size_t get_alloc_size(void* ptr) override;
  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override;
  bool has_multicast_support(int device_idx) override;
  bool has_allocation(void* ptr) override;
  c10::DeviceType supported_device_type() override;
  std::string name() override;

 private:
  // Allocates and maps a block of `size` bytes after a front region of
  // `front_size` bytes, without registering it, so that alloc() and the
  // groups' internal pads share one allocation path.
  c10::intrusive_ptr<Block> create_block(
      size_t size,
      size_t front_size,
      int device_idx,
      const std::optional<std::string>& group_name);
  // The internal pad of (group, device), created by the group's first
  // rendezvous on the device.
  std::shared_ptr<const SignalPad> signal_pad(
      const std::string& group_name,
      int device_idx);
  c10::intrusive_ptr<Block> find_block(void* ptr);
  c10::intrusive_ptr<Block> find_block_covering(void* ptr, size_t& offset);

  std::shared_mutex mutex_;
  std::unordered_map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;
  c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type
      handle_type_ = c10::cuda::CUDACachingAllocator::
          Expandable_Segments_Handle_Type::UNSPECIFIED;
};

} // namespace c10d::symmetric_memory
