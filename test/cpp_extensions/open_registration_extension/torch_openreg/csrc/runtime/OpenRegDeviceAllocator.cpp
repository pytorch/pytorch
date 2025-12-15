#include "OpenRegDeviceAllocator.h"
#include "OpenRegFunctions.h"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <unistd.h>

using namespace c10::CachingAllocator;

namespace c10::openreg {

constexpr size_t kAggregate = static_cast<size_t>(StatType::AGGREGATE);

namespace {

constexpr size_t kDefaultPageSizeBytes = 4096;

size_t getPageSizeBytes() {
  static const long page_size_long = sysconf(_SC_PAGESIZE);
  static const size_t page_size_bytes =
      page_size_long > 0 ? static_cast<size_t>(page_size_long)
                         : kDefaultPageSizeBytes;
  return page_size_bytes;
}

size_t roundUpToPageSize(size_t nbytes) {
  const size_t page_size = getPageSizeBytes();
  return ((nbytes + page_size - 1) / page_size) * page_size;
}

} // namespace


DeviceMemoryAllocator::DeviceMemoryAllocator(c10::DeviceIndex device_index)
    : device_index_(device_index) {}

void* DeviceMemoryAllocator::malloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // OpenReg aligns device allocations to page size internally.
  const size_t aligned_nbytes = roundUpToPageSize(nbytes);

  // Single-level cache: pick the smallest cached block
  auto cached_it = cached_blocks_.lower_bound(aligned_nbytes);
  if (cached_it != cached_blocks_.end()) {
    const size_t cached_nbytes = cached_it->first;
    void* data = cached_it->second;
    cached_blocks_.erase(cached_it);
    cached_pointers_.erase(data);

    allocation_sizes_[data] = cached_nbytes;
    stats_.allocated_bytes[kAggregate].increase(cached_nbytes);
    return data;
  }

  void* data = nullptr;
  auto ret = orMalloc(&data, aligned_nbytes);

  TORCH_CHECK(
      ret == orSuccess && data != nullptr,
      "Failed to allocate ",
      aligned_nbytes,
      " bytes on openreg device ",
      device_index_,
      ". ",
      "Allocated: ",
      stats_.allocated_bytes[kAggregate].current,
      " bytes, ",
      "Reserved: ",
      stats_.reserved_bytes[kAggregate].current,
      " bytes");

  // Track allocation size for proper deallocation statistics
  allocation_sizes_[data] = aligned_nbytes;

  // Update statistics
  stats_.allocated_bytes[kAggregate].increase(aligned_nbytes);
  stats_.reserved_bytes[kAggregate].increase(aligned_nbytes);
  stats_.num_device_alloc++;

  return data;
}

void DeviceMemoryAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = allocation_sizes_.find(ptr);
  if (it != allocation_sizes_.end()) {
    const size_t nbytes = it->second;
    allocation_sizes_.erase(it);

    stats_.allocated_bytes[kAggregate].decrease(nbytes);

    // Cache the block for future reuse.
    cached_blocks_.emplace(nbytes, ptr);
    cached_pointers_.insert(ptr);
    return;
  }

  if (cached_pointers_.find(ptr) != cached_pointers_.end()) {
    TORCH_WARN(
        "Attempted to free an OpenReg memory pointer ",
        ptr,
        " on device ",
        device_index_,
        " that is already cached. This likely indicates a double-free.");
    return;
  }

  // Best-effort orFree for untracked pointers; don't update stats.
  const auto ret = orFree(ptr);
  if (ret == orSuccess) {
    TORCH_WARN(
        "Successfully freed OpenReg memory pointer ",
        ptr,
        " on device ",
        device_index_,
        " that was not tracked by the allocator. "
        "Statistics may be inaccurate.");
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

std::vector<void*> DeviceMemoryAllocator::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::vector<void*> removed;
  for (auto it = cached_blocks_.begin(); it != cached_blocks_.end();) {
    const size_t nbytes = it->first;
    void* ptr = it->second;
    const auto ret = orFree(ptr);

    if (ret == orSuccess || ret == orErrorUnknown) {
      stats_.reserved_bytes[kAggregate].decrease(nbytes);
      if (ret == orSuccess) {
        stats_.num_device_free++;
      }

      cached_pointers_.erase(ptr);
      removed.push_back(ptr);
      it = cached_blocks_.erase(it);
    } else {
      TORCH_WARN(
          "orFree failed while emptying OpenReg cache for pointer ",
          ptr,
          " on device ",
          device_index_,
          ". Return code: ",
          ret);
      ++it;
    }
  }
  return removed;
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
  (void)mempool_id;
  for (auto& allocator : device_allocators_) {
    auto removed = allocator->emptyCache();
    if (removed.empty()) {
      continue;
    }

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (void* ptr : removed) {
      allocated_blocks_.erase(ptr);
    }
  }
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
