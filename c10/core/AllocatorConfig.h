#pragma once

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <c10/util/llvmMathExtras.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace c10::CachingAllocator {

// "large" allocations may be packed in 20 MiB blocks
const size_t kLargeBuffer = 20971520;

/**
 * Note [AllocatorConfig design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * This class is used to configure the allocator for both device and host
 * allocator. A single AllocatorConfig for all devices, assuming that
 * environment variables apply universally.
 *
 */

class C10_API AllocatorConfig {
 public:
  AllocatorConfig();

  /* Device allocator settings */

  size_t da_max_split_size() {
    return da_max_split_size_;
  }

  size_t da_max_non_split_rounding_size() {
    return da_max_non_split_rounding_size_;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As an example, if we want 4 divisions between 2's power, this can be done
  // using env variable:
  // PYTORCH_ALLOC_CONF=roundup_power2_divisions:4
  size_t da_roundup_power2_divisions(size_t size);

  std::vector<size_t> da_roundup_power2_divisions() {
    return da_roundup_power2_divisions_;
  }

  double da_garbage_collection_threshold() {
    return da_garbage_collection_threshold_;
  }

  bool da_use_async_allocator() {
    return da_use_async_allocator_;
  }

  bool da_use_expandable_segments() {
    return da_use_expandable_segments_;
  }

  bool da_use_release_lock_on_malloc() {
    return da_use_release_lock_on_malloc_;
  }

  /* Host allocator settings */
  bool ha_use_host_register() {
    return ha_use_host_register_;
  }

  size_t ha_num_register_threads() {
    return ha_num_register_threads_;
  }

  bool ha_use_background_threads() {
    return ha_use_background_threads_;
  }

  /* Settings for both device and host allocator */

  std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(last_allocator_settings_mutex_);
    return last_allocator_settings_;
  }

 private:
  /* Internal functions */
  void parseArgs(const char* env);

  void lexArgs(const char* env, std::vector<std::string>& config);

  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);

  /* Internal functions for device allocator */

  size_t daParseMaxSplitSize(const std::vector<std::string>& config, size_t i);

  size_t daParseMaxNonSplitRoundingSize(
      const std::vector<std::string>& config,
      size_t i);

  size_t daParseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);

  size_t daParseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);

  size_t daParseUseAsyncAllocator(
      const std::vector<std::string>& config,
      size_t i);

  /* Internal functions for host allocator */

  size_t haParseUseHostRegister(
      const std::vector<std::string>& config,
      size_t i);

  size_t haParseNumRegisterThreads(
      const std::vector<std::string>& config,
      size_t i);

  size_t haParseUseBackgroundThreads(
      const std::vector<std::string>& config,
      size_t i);

  /* Members prefixed with `da_` are specifically used for the device
   * allocator. */

  // The maximum block size that is allowed to be split.
  std::atomic<size_t> da_max_split_size_{std::numeric_limits<size_t>::max()};
  // The maximum allowable extra size of a memory block without requiring
  // splitting when searching for a free block.
  std::atomic<size_t> da_max_non_split_rounding_size_{kLargeBuffer};
  // Used to store how memory allocations of different sizes should be rounded
  // up to the nearest power of 2 divisions.
  std::vector<size_t> da_roundup_power2_divisions_;
  // The threshold that triggers garbage collection when the ratio of used
  // memory to maximum allowed memory exceeds this value.
  std::atomic<double> da_garbage_collection_threshold_{0};
  // A flag to enable MallocAsync feature.
  std::atomic<bool> da_use_async_allocator_{false};
  // A flag to enable expandable segments feature.
  std::atomic<bool> da_use_expandable_segments_{false};
  // A flag to release the lock on device malloc.
  std::atomic<bool> da_use_release_lock_on_malloc_{false};

  /* Members prefixed with `ha_` are specifically used for the host (pinned)
   * allocator. */

  // A flag that determines whether to register a CPU allocation for use by
  // device.
  std::atomic<bool> ha_use_host_register_{false};
  // The number of threads to parallelize to register a CPU allocation to reduce
  // the overall time.
  std::atomic<size_t> ha_num_register_threads_{1};
  // A flag to enable background thread for processing events.
  std::atomic<bool> ha_use_background_threads_{false};

  /* The following members are used for both device and host allocator. */

  // Record the last allocator config environment setting.
  std::mutex last_allocator_settings_mutex_;
  std::string last_allocator_settings_;
};

C10_API void SetAllocatorConfig(
    at::DeviceType t,
    AllocatorConfig* allocator_config);
C10_API AllocatorConfig* GetAllocatorConfig(const at::DeviceType& t);

} // namespace c10::CachingAllocator
