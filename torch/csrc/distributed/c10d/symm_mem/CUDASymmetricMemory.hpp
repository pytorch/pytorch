#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <map>
#include <shared_mutex>
#include <vector>

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
      void* mc_signal_pad_addr,
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
  void* mc_signal_pad_addr_;
  HandleType mc_handle_;
  void* mc_addr_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::string group_name_;

  friend class CUDASymmetricMemory;
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
  // Handles this allocator has returned, by storage pointer and
  // group, each recorded under the block it lies in so that the block's entries
  // go with it. Ops rendezvous their inputs on every call, so a repeat is a
  // pointer hash and a name comparison, with no string copied or hashed. Holds
  // at most one entry per storage pointer and group. Not thread-safe: the
  // allocator's mutex guards it.
  class HandleCache {
   public:
    // The handle cached for (ptr, group), or null.
    c10::intrusive_ptr<SymmetricMemory> find(
        void* ptr,
        const std::string& group) const;
    // Caches `handle` for (ptr, group), which must not be cached yet; `block`
    // is the block ptr lies in.
    void insert(
        const Block* block,
        void* ptr,
        std::string group,
        c10::intrusive_ptr<SymmetricMemory> handle);
    // Drops every entry recorded under `block`.
    void erase_block(const Block* block);

   private:
    struct Entry {
      std::string group;
      c10::intrusive_ptr<SymmetricMemory> handle;
    };
    ska::flat_hash_map<void*, std::vector<Entry>> by_ptr_;
    ska::flat_hash_map<const Block*, std::vector<void*>> ptrs_by_block_;
  };

  c10::intrusive_ptr<Block> find_block(void* ptr);
  c10::intrusive_ptr<Block> find_block_covering(void* ptr, size_t& offset);

  // Guards ptr_to_block_ and handles_.
  std::shared_mutex mutex_;
  // Keyed by the address alloc() returned. Ordered, so the block covering an
  // interior pointer is found with upper_bound rather than a scan.
  std::map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;
  HandleCache handles_;
  c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type
      handle_type_ = c10::cuda::CUDACachingAllocator::
          Expandable_Segments_Handle_Type::UNSPECIFIED;
};

} // namespace c10d::symmetric_memory
