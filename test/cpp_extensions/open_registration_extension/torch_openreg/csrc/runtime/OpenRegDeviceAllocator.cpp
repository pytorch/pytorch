#include "OpenRegDeviceAllocator.h"
#include "OpenRegFunctions.h"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

using namespace c10::CachingAllocator;

namespace c10::openreg {

constexpr size_t kAggregate = static_cast<size_t>(StatType::AGGREGATE);


DeviceMemoryAllocator::DeviceMemoryAllocator(c10::DeviceIndex device_index)
    : device_index_(device_index) {}

void* DeviceMemoryAllocator::malloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  void* data = nullptr;
  auto ret = orMalloc(&data, nbytes);

  TORCH_CHECK(
      ret == orSuccess && data != nullptr,
      "Failed to allocate ",
      nbytes,
      " bytes on openreg device ",
      device_index_,
      ". ",
      "Allocated: ",
      stats_.allocated_bytes[0].current,
      " bytes, ",
      "Reserved: ",
      stats_.reserved_bytes[0].current,
      " bytes");

  // Track allocation size for proper deallocation statistics
  allocation_sizes_[data] = nbytes;

  // Update statistics
  stats_.allocated_bytes[kAggregate].increase(nbytes);
  stats_.reserved_bytes[kAggregate].increase(nbytes);
  stats_.num_device_alloc++;

  return data;
}

void DeviceMemoryAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto ret = orFree(ptr);

  if (ret == orSuccess) {
    auto it = allocation_sizes_.find(ptr);
    if (it != allocation_sizes_.end()) {
      size_t nbytes = it->second;

      stats_.allocated_bytes[kAggregate].decrease(nbytes);
      stats_.reserved_bytes[kAggregate].decrease(nbytes);
      stats_.num_device_free++;

      allocation_sizes_.erase(it);
    } else {
      TORCH_WARN(
          "Successfully freed OpenReg memory pointer ",
          ptr,
          " on device ",
          device_index_,
          " that was not tracked by the allocator. "
          "Statistics may be inaccurate.");
    }
  } else {
    // orFree failed
    auto it = allocation_sizes_.find(ptr);
    if (it != allocation_sizes_.end()) {
      TORCH_WARN(
          "orFree failed for tracked pointer ",
          ptr,
          " with size ",
          it->second,
          " bytes on device ",
          device_index_,
          ". Return code: ",
          ret,
          ". Keeping tracking record - this may indicate a double-free or invalid pointer.");
    } else {
      TORCH_WARN(
          "orFree failed for untracked pointer ",
          ptr,
          " on device ",
          device_index_,
          ". Return code: ",
          ret,
          ". This likely indicates a double-free or invalid pointer.");
    }
  }
}

c10::CachingDeviceAllocator::DeviceStats DeviceMemoryAllocator::getStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return stats_;
}

void DeviceMemoryAllocator::resetAccumulatedStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Reset accumulated statistics for all StatTypes
  for (const auto stat_type :
       c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
    stats_.allocated_bytes[stat_type].reset_accumulated();
    stats_.reserved_bytes[stat_type].reset_accumulated();
    stats_.active_bytes[stat_type].reset_accumulated();
    stats_.inactive_split_bytes[stat_type].reset_accumulated();
    stats_.requested_bytes[stat_type].reset_accumulated();
  }

  stats_.num_alloc_retries = 0;
  stats_.num_ooms = 0;
  stats_.num_sync_all_streams = 0;
  stats_.num_device_alloc = 0;
  stats_.num_device_free = 0;
}

void DeviceMemoryAllocator::resetPeakStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Reset peak statistics for all StatTypes
  for (const auto stat_type :
       c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
    stats_.allocated_bytes[stat_type].reset_peak();
    stats_.reserved_bytes[stat_type].reset_peak();
    stats_.active_bytes[stat_type].reset_peak();
    stats_.inactive_split_bytes[stat_type].reset_peak();
    stats_.requested_bytes[stat_type].reset_peak();
  }

  stats_.oversize_allocations.reset_peak();
  stats_.oversize_segments.reset_peak();
}

namespace {

OpenRegDeviceAllocator g_allocator;

void deleteOpenRegMemory(void* ptr) {
  g_allocator.freeMemory(ptr);
}

}

OpenRegDeviceAllocator::OpenRegDeviceAllocator() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto device_count = c10::openreg::device_count();
  device_allocators_.resize(device_count);
  for (const auto i : c10::irange(device_count)) {
    device_allocators_[i] = std::make_unique<DeviceMemoryAllocator>(i);
  }
}


at::DataPtr OpenRegDeviceAllocator::allocate(size_t nbytes) {
  int current_device_index = -1;
  auto ret = orGetDevice(&current_device_index);
  TORCH_CHECK(ret == orSuccess, "Failed to get current OpenReg device");

  auto curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, current_device_index);

  void* data = nullptr;
  if (nbytes > 0) {
    // Allocate memory via device-specific allocator
    data = device_allocators_[current_device_index]->malloc(nbytes);

    // Track which device owns this pointer
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    allocated_blocks_[data] = current_device_index;
  }

  return {data, data, &deleteOpenRegMemory, curr_device};
}

at::DeleterFnPtr OpenRegDeviceAllocator::raw_deleter() const {
  return &deleteOpenRegMemory;
}

void OpenRegDeviceAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  auto ret = orMemcpy(dest, src, count, orMemcpyDeviceToDevice);
  TORCH_CHECK(
      ret == orSuccess, "Failed to copy ", count, " bytes on openreg device");
}

bool OpenRegDeviceAllocator::initialized() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return !device_allocators_.empty();
}

void OpenRegDeviceAllocator::freeMemory(void* ptr) {
  if (!ptr) {
    return;
  }

  // Try to find which device owns this pointer
  c10::DeviceIndex device_index = -1;
  bool found_in_map = false;

  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = allocated_blocks_.find(ptr);
    if (it != allocated_blocks_.end()) {
      device_index = it->second;
      allocated_blocks_.erase(it);
      found_in_map = true;
    }
  }

  if (found_in_map) {
    // Pointer was tracked - free via device-specific allocator with stats
    device_allocators_[device_index]->free(ptr);
  } else {
    // Pointer not tracked - might be already freed by storage or other path
    // Try to free it directly via orFree without updating statistics
    auto ret = orFree(ptr);

    // Only warn if orFree actually failed (not just "not found")
    // In OpenReg's case, orFree returns orErrorUnknown if pointer not in registry
    // which is expected for already-freed memory
    if (ret != orSuccess && ret != orErrorUnknown) {
      TORCH_WARN(
          "orFree failed for untracked OpenReg memory pointer ",
          ptr,
          ". Error code: ", ret);
    }
  }
}

c10::CachingDeviceAllocator::DeviceStats OpenRegDeviceAllocator::
    getDeviceStats(c10::DeviceIndex device) {
  return device_allocators_[device]->getStats();
}

void OpenRegDeviceAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  device_allocators_[device]->resetAccumulatedStats();
}

void OpenRegDeviceAllocator::resetPeakStats(c10::DeviceIndex device) {
  device_allocators_[device]->resetPeakStats();
}

void OpenRegDeviceAllocator::emptyCache(MempoolId_t mempool_id) {
  // OpenReg doesn't implement caching yet
  // TODO: When caching is implemented, release all free blocks here
}

void OpenRegDeviceAllocator::recordStream(
    const DataPtr& ptr,
    c10::Stream stream) {
  // OpenReg doesn't track stream usage yet
  // TODO: When stream support is added, track which streams are using this pointer
}
// ============ Global Registration ============

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

} // namespace c10::openreg
