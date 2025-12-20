// Some stateful GPU libraries, such as cuDNN, cuBLAS, use handles to store
// states. These handles are tied to device, and these libraries
// requires/recommends not to share handles across host threads.
//
// These libraries recommend using one handle per host thread. We may not want
// to do this because threads are relatively light-weight, but creating and
// destroying handles is expensive (destroying the handle causes
// synchronizations). DataParallel, for example, creates new threads for each
// forward pass.
//
// This file implements a handle pool mechanism. The handle pool returns handles
// on demand as threads request them. If all existing handles in the pool are in
// use, it creates a new one. As threads terminate, they release handles back
// into the pool. In this way, the handle pool never creates more handles than
// the high-water mark of active threads, so it's efficient with DataParallel.

#pragma once

#include <mutex>
#include <unordered_map>
#include <utility>

#include <c10/core/Device.h>

namespace at::cuda {
namespace detail {
// RAII type to ensure handles are destroyed when this object is destroyed
template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
struct HandleRaii {
  Handle_t handle{nullptr};

  HandleRaii() {
    Create(&handle);
  }

  ~HandleRaii() {
    if (handle != nullptr)
      Destroy(std::exchange(handle, nullptr));
  }

  // Non-copyable nature of the type ensures that we cannot accidentally
  // create more than one copies of the RAII object, thereby ensuring that
  // the underlying handle is not deallocated when a copy goes out of scope.
  // An alternative is to have a reference counted type but this approach
  // is much simpler and suffices for the handle pool use case.
  HandleRaii(const HandleRaii&) = delete;
  HandleRaii& operator=(const HandleRaii&) = delete;

  // Non-movability of this type is non-necessary but since we don't need
  // to move this type, we disable the move construction and assignment.
  HandleRaii(HandleRaii&&) = delete;
  HandleRaii& operator=(HandleRaii&&) = delete;
};

// GlobalHandleStorage is the pool of handles that are available for use
// This needs to be protected by a mutex because multiple threads might
// try to remove/add handles from/to this pool.
template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
struct GlobalHandleStorage {
private:
  using Handle = HandleRaii<Handle_t, Create, Destroy>;
  using Multimap = std::unordered_multimap<c10::DeviceIndex, Handle>;
  Multimap handles_;
  std::mutex mutex_;

public:
  typename Multimap::node_type extract(c10::DeviceIndex device) {
    std::lock_guard<std::mutex> lock(mutex_);
    return handles_.extract(device);
  }

  void merge(std::unordered_map<c10::DeviceIndex, Handle>&& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    handles_.merge(std::move(source));
  }
};

// ThreadLocalHandleStorage contains the handles that are being used by a given
// thread. When handles are requested, it either takes a handle from the global
// pool or allocates a new one if nothing is available. On destruction, it puts
// the owned handles back into the global pool.
template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
struct ThreadLocalHandleStorage {
private:
  using Handle = HandleRaii<Handle_t, Create, Destroy>;
  std::unordered_map<c10::DeviceIndex, Handle> handles_;
  GlobalHandleStorage<Handle_t, Create, Destroy>& global_;

public:
  explicit ThreadLocalHandleStorage(GlobalHandleStorage<Handle_t, Create, Destroy>& global)
    : global_(global) {}

  Handle_t reserve(c10::DeviceIndex device) {
    { // This thread already has a handle for device
      auto it = handles_.find(device);
      if (it != handles_.end()) {
        return it->second.handle;
      }
    }

    { // Use a handle from the global pool if available
      auto it = handles_.insert(global_.extract(device)).position;
      if (it != handles_.end()) {
        return it->second.handle;
      }
    }

    // Allocate a new handle. operator[] will default construct
    return handles_[device].handle;
  }

  ~ThreadLocalHandleStorage() {
    // Release handles back to the global pool
    if (!handles_.empty())
      global_.merge(std::move(handles_));
  }
};
} // namespace detail

template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
struct DeviceThreadHandlePool {
  static Handle_t reserve(c10::DeviceIndex device) {
    static detail::GlobalHandleStorage<Handle_t, Create, Destroy> available;
    thread_local detail::ThreadLocalHandleStorage<Handle_t, Create, Destroy> local(available);
    return local.reserve(device);
  }
};

} // namespace at::cuda
