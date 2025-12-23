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

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

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
//
// Handles are lazily created as different threads request them, but are never
// destroyed until the end of the process. The maximum number of handles this
// process will create for each device is equal to the high-water mark of the
// number of concurrently active threads that request handles for that device.
// When threads terminate, they release their handles back into the pool for
// reuse. Otherwise, new handles would be created every time new threads were
// spawned, resulting in poor performance for Python modules that repeatedly or
// frequently spawned new sets of threads (like DataParallel, which creates a
// new set of threads for each forward pass).
//
// To prevent potential deadlocks, we explicitly choose not to cap the number
// of handles that are created per device.
//
// Example of danger: If we cap the max handles at 4, and 5 threads are sharing
// a device, only 4 can make forward progress at any time. The other 4 will not
// release their handles until they exit, so the fifth cannot make progress
// until then.  This is not a problem...UNLESS all 5 threads attempt some sort
// of synchronization at an intermediate point (ie, before any of them have
// exited).  We have no way to anticipate or enforce that user threads will not
// attempt such intermediate synchronization. The only way to ensure safety is
// to avoid imposing a cap on the number of handles.
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
  using GlobalStorage = GlobalHandleStorage<Handle_t, Create, Destroy>;

  // In the case where a thread is detached we cannot rely on static storage
  // outliving thread_local storage. This weak_ptr ensures that the global
  // handle pool is not accessed if static storage has been destroyed.
  std::weak_ptr<GlobalStorage> global_;
  std::unordered_map<c10::DeviceIndex, Handle> handles_;

public:
  explicit ThreadLocalHandleStorage(std::weak_ptr<GlobalStorage> global)
    : global_(std::move(global)) {}

  Handle_t reserve(c10::DeviceIndex device) {
    { // This thread already has a handle for device
      auto it = handles_.find(device);
      if (it != handles_.end()) {
        return it->second.handle;
      }
    }

    std::shared_ptr<GlobalStorage> global = global_.lock();
    TORCH_CHECK(global != nullptr, "Cannot create handle during program termination");

    { // Use a handle from the global pool if available
      auto it = handles_.insert(global->extract(device)).position;
      if (it != handles_.end()) {
        return it->second.handle;
      }
    }

    // Allocate a new handle. operator[] will call HandleRaii::HandleRaii
    return handles_[device].handle;
  }

  ~ThreadLocalHandleStorage() {
    // Release handles back to the global pool if it's still alive
    if (!handles_.empty()) {
      if (std::shared_ptr<GlobalStorage> global = global_.lock()) {
        global->merge(std::move(handles_));
      }
    }
  }
};
} // namespace detail

template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
struct DeviceThreadHandlePool {
  static Handle_t reserve(c10::DeviceIndex device) {
    using GlobalStorage = detail::GlobalHandleStorage<Handle_t, Create, Destroy>;
    static auto global = std::make_shared<GlobalStorage>();
    // We need to this funky initialization because thread_local does not behave
    // as expected with MSVC. See https://github.com/pytorch/pytorch/pull/22405
    using ThreadLocalStorage = detail::ThreadLocalHandleStorage<Handle_t, Create, Destroy>;
    thread_local std::unique_ptr<ThreadLocalStorage> local;
    if (local == nullptr) {
      local.reset(new ThreadLocalStorage(global));
    }

    return local->reserve(device);
  }
};

} // namespace at::cuda
