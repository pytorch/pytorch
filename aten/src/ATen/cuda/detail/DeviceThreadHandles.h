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

  HandleRaii(const HandleRaii&) = delete;
  HandleRaii& operator=(const HandleRaii&) = delete;
  HandleRaii(HandleRaii&&) = delete;
  HandleRaii& operator=(HandleRaii&&) = delete;
};

template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
struct GlobalHandleStorage {
  using HandleStorage = HandleRaii<Handle_t, Create, Destroy>;
  std::mutex mutex;
  std::unordered_multimap<c10::DeviceIndex, HandleStorage> handles;
};

template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
class ThreadLocalHandleStorage {
public:
  explicit ThreadLocalHandleStorage(GlobalHandleStorage<Handle_t, Create, Destroy>& global)
    : global_(global) {}

  Handle_t reserve(c10::DeviceIndex device) {
    { // this thread already has a handle for device
      auto it = handles_.find(device);
      if (it != handles_.end()) {
        return it->second.handle;
      }
    }

    { // use a handle from the global pool if available
      std::unique_lock<std::mutex> lock(global_.mutex);
      auto node = global_.handles.extract(device);
      lock.unlock();

      auto it = handles_.insert(std::move(node)).position;
      if (it != handles_.end()) {
        return it->second.handle;
      }
    }

    // allocate a new handle. operator[] with default construct
    return handles_[device].handle;
  }

  ~ThreadLocalHandleStorage() {
    if (handles_.empty())
      return;

    // release handles back to the global pool
    std::lock_guard<std::mutex> lock(global_.mutex);
    global_.handles.merge(std::move(handles_));
  }

private:
  using HandleStorage = HandleRaii<Handle_t, Create, Destroy>;
  GlobalHandleStorage<Handle_t, Create, Destroy>& global_;
  std::unordered_map<c10::DeviceIndex, HandleStorage> handles_;
};

template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
Handle_t reserveHandle(c10::DeviceIndex device) {
  static GlobalHandleStorage<Handle_t, Create, Destroy> global;
  thread_local ThreadLocalHandleStorage<Handle_t, Create, Destroy> local(global);
  return local.reserve(device);
}

} // namespace at::cuda
