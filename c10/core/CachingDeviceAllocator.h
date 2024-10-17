#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/irange.h>

#include <array>

namespace c10::CachingDeviceAllocator {

struct Stat {
  void increase(size_t amount) {
    current += static_cast<int64_t>(amount);
    peak = std::max(current, peak);
    allocated += static_cast<int64_t>(amount);
  }

  void decrease(size_t amount) {
    current -= static_cast<int64_t>(amount);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        current >= 0,
        "Negative tracked stat in device allocator (likely logic error).");
    freed += static_cast<int64_t>(amount);
  }

  void reset_accumulated() {
    allocated = 0;
    freed = 0;
  }

  void reset_peak() {
    peak = current;
  }

  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

using StatArray = std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)>;
using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from device memory allocation.
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via device memory deallocation)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory alocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to device malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to device memory allocation
  // after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // COUNT: total number of synchronize_and_free_events() calls
  int64_t num_sync_all_streams = 0;

  // COUNT: total number of device memory allocation calls. This includes both
  // mapped and malloced memory.
  int64_t num_device_alloc = 0;

  // COUNT: total number of device memory deallocation calls. This includes both
  // un-mapped and free memory.
  int64_t num_device_free = 0;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

// Size pretty-printer
inline std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (static_cast<double>(size) / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << static_cast<double>(size) / 1048576.0;
    os << " MiB";
  } else {
    os << static_cast<double>(size) / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

} // namespace c10::CachingDeviceAllocator
