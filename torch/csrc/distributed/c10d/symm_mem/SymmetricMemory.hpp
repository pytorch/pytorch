#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d::symmetric_memory {

// SymmetricMemory represents symmetric allocations across a group of devices.
// The allocations represented by a SymmetricMemory object are accessible by
// all devices in the group. The class can be used for op-level custom
// communication patterns (via the get_buffer APIs and the synchronization
// primitives), as well as custom communication kernels (via the buffer and
// signal_pad device pointers).
//
// To acquire a SymmetricMemory object, each rank first allocates
// identical-sized memory via SymmetricMemoryAllocator::alloc(), then invokes
// SymmetricMemoryAllocator::rendezvous() on the memory to establish the
// association across peer buffers. The rendezvous is a one-time process, and
// the mapping between a local memory memory and the associated SymmetricMemory
// object is unique.
//
// NOTE [symmetric memory signal pad]
// Signal pads are P2P-accessible memory regions designated for
// synchronization. SymmetricMemory offers built-in synchronization primitives
// such as barriers, put_signal, and wait_signal, which are all based on signal
// pads. Users may utilize signal pads for their own synchronization logic,
// provided that the signal pads remain zero-filled following successful
// synchronization.
//
// NOTE [symmetric memory synchronization channel]
// Synchronization channels allow users to use a single SymmetricMemory object
// to perform isolated synchronizations on different streams. For example,
// consider the case in which two barriers are issued on two streams for
// different purposes. Without the concept of channels, we cannot guarantee the
// correctness of the barriers since signals issued from barrier on stream A
// can be received by the barrier on stream B. By specifying different channels
// for these two barriers, they can operate correctly in parallel.
class TORCH_API SymmetricMemory : public c10::intrusive_ptr_target {
 public:
  ~SymmetricMemory() override = default;

  virtual std::vector<void*> get_buffer_ptrs() = 0;
  virtual std::vector<void*> get_signal_pad_ptrs() = 0;

  // get_buffer_ptrs_dev() and get_signal_pad_ptrs_dev() each return a pointer
  // to a device array of size world_size, containing buffer pointers and
  // signal pad pointers, respectively.
  virtual void** get_buffer_ptrs_dev() = 0;
  virtual void** get_signal_pad_ptrs_dev() = 0;
  virtual size_t get_buffer_size() = 0;
  virtual size_t get_signal_pad_size() = 0;

  virtual bool has_multicast_support() = 0;
  virtual void* get_multicast_ptr() = 0;

  virtual at::Tensor get_buffer(
      int rank,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) = 0;

  virtual at::Tensor get_signal_pad(
      int rank,
      c10::IntArrayRef sizes,
      std::optional<c10::ScalarType> dtype = std::nullopt,
      int64_t storage_offset = 0) = 0;

  virtual void barrier(int channel, size_t timeout_ms) = 0;
  virtual void put_signal(int dst_rank, int channel, size_t timeout_ms) = 0;
  virtual void wait_signal(int src_rank, int channel, size_t timeout_ms) = 0;

  virtual int get_rank() = 0;
  virtual int get_world_size() = 0;

  virtual const std::vector<int>& get_rank_to_global_rank() {
    TORCH_CHECK(false, "NYI");
  }

  virtual int* get_rank_to_global_rank_dev() {
    TORCH_CHECK(false, "NYI");
  }
};

class SymmetricMemoryAllocator : public c10::intrusive_ptr_target {
 public:
  ~SymmetricMemoryAllocator() override = default;

  virtual void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) = 0;

  virtual void free(void* ptr) = 0;
  virtual size_t get_alloc_size(void* ptr) = 0;
  virtual c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) = 0;
  virtual bool has_multicast_support(int device_idx) = 0;
  virtual c10::DeviceType supported_device_type() = 0;
  virtual std::string name() = 0;
};

C10_EXPORT bool is_finalizing();

C10_EXPORT void register_allocator(
    c10::DeviceType device_type,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator);

C10_EXPORT void register_availability(
    const std::string& name,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator);

C10_EXPORT bool has_allocator(c10::DeviceType device_type);

C10_EXPORT c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
    c10::DeviceType device_type);

// Set a store for rendezvousing symmetric allocations on a group of devices
// identified by `group_name`. The concept of groups is logical; users can
// utilize predefined groups (e.g., a group of device identified by a
// ProcessGroup) or create custom ones. Note that a SymmetricMemoryAllocator
// backends might employ a more efficient communication channel for the actual
// rendezvous process and only use the store for bootstrapping purposes.
TORCH_API void set_group_info(
    const std::string& group_name,
    int rank,
    int world_size,
    c10::intrusive_ptr<Store> store);

struct GroupInfo {
  int rank;
  int world_size;
  c10::intrusive_ptr<c10d::Store> store;
  // Note this field is not automatically populated by set_group_info().  If a
  // SymmetricMemory implementation needs to use it, it must be populated by a
  // call to exchange_global_ranks() first.
  std::vector<int> rank_to_global_rank;
};

C10_EXPORT GroupInfo& get_group_info(const std::string& group_name);

// Identical to empty_strided, but allows symmetric memory access to be
// established for the allocated tensor via SymmetricMemory::rendezvous(). This
// function itself is not a collective operation. It invokes
// SymmetricMemoryAllocator::alloc() for the requested device under the hood.
//
// NOTE [symmetric memory persistent allocation]
// If an `alloc_id` is supplied, empty_strided_p2p will perform persistent
// allocation. This makes the function cache allocated memory and ensure that
// invocations with the same `alloc_id` receive tensors backed by the same
// memory address. For safety, if a previous persistent allocation is still
// active (i.e., the storage of the returned tensor is still alive), persistent
// allocations with the same `alloc_id` will fail. This determinism coupled
// with memory planning of communication buffers (e.g., by Inductor) allows
// communication algorithms to reliably reuse previously established remote
// memory access.
TORCH_API at::Tensor empty_strided_p2p(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::optional<std::string>& group_name,
    std::optional<uint64_t> alloc_id);

// Establishes symmetric memory access on tensors allocated via
// empty_strided_p2p() and empty_strided_p2p_persistent(). rendezvous() is a
// one-time process, and the mapping between a local memory region and the
// associated SymmetricMemory object is unique. Subsequent calls to
// rendezvous() with the same tensor, or tensors allocated with
// empty_strided_p2p_persistent() using the same alloc_id, will receive the
// cached SymmetricMemory object.
//
// The function has a collective semantic and must be invoked simultaneously
// from all rendezvous participants.
TORCH_API c10::intrusive_ptr<SymmetricMemory> rendezvous(
    const at::Tensor& tensor,
    const std::optional<std::string>& group_name = std::nullopt);

TORCH_API bool has_multicast_support(
    c10::DeviceType device_type,
    int device_idx);

TORCH_API void set_backend(const std::string& name);

TORCH_API std::optional<std::string> get_backend(c10::Device device);

} // namespace c10d::symmetric_memory
