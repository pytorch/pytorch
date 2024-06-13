#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {
namespace symmetric_memory {

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
  virtual ~SymmetricMemory() {}

  // For custom kernels
  virtual std::vector<void*> get_buffer_ptrs() = 0;
  virtual std::vector<void*> get_signal_pad_ptrs() = 0;
  virtual void** get_buffer_ptrs_dev() = 0;
  virtual void** get_signal_pad_ptrs_dev() = 0;
  virtual size_t get_buffer_size() = 0;
  virtual size_t get_signal_pad_size() = 0;

  virtual at::Tensor get_buffer(
      int rank,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) = 0;

  virtual void barrier(int channel) = 0;
  virtual void put_signal(int dst_rank, int channel) = 0;
  virtual void wait_signal(int src_rank, int channel) = 0;
};

class SymmetricMemoryAllocator : public c10::intrusive_ptr_target {
 public:
  virtual ~SymmetricMemoryAllocator(){};

  virtual void* alloc(
      size_t size,
      int device_idx,
      const std::string& group_name) = 0;

  virtual void free(void* ptr) = 0;
  virtual size_t get_alloc_size(void* ptr) = 0;
  virtual c10::intrusive_ptr<SymmetricMemory> rendezvous(void* ptr) = 0;
  virtual bool is_rendezvous_completed(void* ptr) = 0;
};

void register_allocator(
    c10::DeviceType device_type,
    c10::intrusive_ptr<SymmetricMemoryAllocator> allocator);

c10::intrusive_ptr<SymmetricMemoryAllocator> get_allocator(
    c10::DeviceType device_type);

// Assign a store to `group_name` for rendezvous
// A SymmetricMemoryAllocator backend might establish a more efficient
// communication channel and only use the store for bootstrapping purpose.
TORCH_API void set_group_info(
    const std::string& group_name,
    int rank,
    int world_size,
    c10::intrusive_ptr<Store> store);

struct GroupInfo {
  int rank;
  int world_size;
  c10::intrusive_ptr<c10d::Store> store;
};

const GroupInfo& get_group_info(const std::string& group_name);

// Allocate a tensor using SymmetricMemoryAllocator::alloc()
TORCH_API at::Tensor empty_strided_p2p(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::string& group_name);

// Same as empty_strided_p2p but handles allocations in a persistent fashion.
// The function caches all allocations and ensures that invocations with the
// same `alloc_id` receive the same allocation, which maps to a unique
// SymmetricMemory object. It is intended for static planning of communication
// buffers (e.g. by Inductor).
TORCH_API at::Tensor empty_strided_p2p_persistent(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::ScalarType dtype,
    c10::Device device,
    const std::string& group_name,
    uint64_t alloc_id);

// Performs a rendezvous on tensors allocated via empty_strided_p2p() and
// empty_strided_p2p_persistent(). The rendezvous is a one-time process, and
// the mapping between a local memory region and the associated SymmetricMemory
// object is unique. Subsequent calls to rendezvous() with the same tensor, or
// tensors allocated with empty_strided_p2p_persistent() using the same
// alloc_id, will return the same cached SymmetricMemory object.
//
// The function has a collective semantic and must be invoked simultaneously
// from all rendezvous participants.
TORCH_API c10::intrusive_ptr<SymmetricMemory> rendezvous(
    const at::Tensor& tensor);

// Returns the SymmetricMemory object associated with the tensor. It can only
// be invoked after rendezvous() but does not need to be invoked collectively.
TORCH_API c10::intrusive_ptr<SymmetricMemory> get_symmetric_memory(
    const at::Tensor& tensor);

} // namespace symmetric_memory
} // namespace c10d
