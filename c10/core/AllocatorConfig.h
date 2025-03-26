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

// Environment config parser for Allocator
class C10_API AllocatorConfig {
 public:
  AllocatorConfig(c10::DeviceType t);

  size_t max_split_size() {
    return max_split_size_;
  }

  size_t max_non_split_rounding_size() {
    return max_non_split_rounding_size_;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As an example, if we want 4 divisions between 2's power, this can be done
  // on CUDA device using env variable:
  // PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  size_t roundup_power2_divisions(size_t size);

  std::vector<size_t> roundup_power2_divisions() {
    return roundup_power2_divisions_;
  }

  double garbage_collection_threshold() {
    return garbage_collection_threshold_;
  }

  bool expandable_segments() {
    return expandable_segments_;
  }

  bool release_lock_on_device_malloc() {
    return release_lock_on_device_malloc_;
  }

  std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(last_allocator_settings_mutex_);
    return last_allocator_settings_;
  }

  /* Pinned memory allocator settings */
  bool pinned_use_host_register() {
    return pinned_use_host_register_;
  }

  size_t pinned_num_register_threads() {
    return pinned_num_register_threads_;
  }

  bool pinned_use_background_threads() {
    return pinned_use_background_threads_;
  }

 private:
  /* Internal functions */
  virtual void parseArgs(const char* env);

  virtual void checkMallocAsyncSupport();

  void lexArgs(const char* env, std::vector<std::string>& config);

  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);

  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);

  size_t parseMaxNonSplitRoundingSize(
      const std::vector<std::string>& config,
      size_t i);

  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);

  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);

  size_t parseAllocatorBackendConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_MallocAsync);

  size_t parsePinnedUseHostRegister(
      const std::vector<std::string>& config,
      size_t i);

  size_t parsePinnedNumRegisterThreads(
      const std::vector<std::string>& config,
      size_t i);

  size_t parsePinnedUseBackgroundThreads(
      const std::vector<std::string>& config,
      size_t i);

  // The maximum block size that is allowed to be split. Default is
  // std::numeric_limits<size_t>::max()
  std::atomic<size_t> max_split_size_;
  // The maximum allowable extra size of a memory block without requiring
  // splitting when searching for a free block. Default is kLargeBuffer
  std::atomic<size_t> max_non_split_rounding_size_;
  // Used to store how memory allocations of different sizes should be rounded
  // up to the nearest power of 2 divisions.
  std::vector<size_t> roundup_power2_divisions_;
  // The threshold that triggers garbage collection when the ratio of used
  // memory to maximum allowed memory exceeds this value. Default is 0.0
  std::atomic<double> garbage_collection_threshold_;
  // A flag to enable expandable segments feature. The default value is false.
  std::atomic<bool> expandable_segments_;
  // A flag to release the lock on device malloc. The default value is false.
  std::atomic<bool> release_lock_on_device_malloc_;
  // Record the last allocator config environment setting.
  std::mutex last_allocator_settings_mutex_;
  std::string last_allocator_settings_;
  // A flag that determines whether to register a CPU allocation for use by
  // device. The default value is false.
  std::atomic<bool> pinned_use_host_register_;
  // The number of threads to parallelize to register a CPU allocation to reduce
  // the overall time. Default is 1.
  std::atomic<size_t> pinned_num_register_threads_;
  // A flag to enable background thread for processing events. The default value
  // is false.
  std::atomic<bool> pinned_use_background_threads_;

  c10::DeviceType device_type_;
};

C10_API void SetAllocatorConfig(
    at::DeviceType t,
    AllocatorConfig* allocator_config);
C10_API AllocatorConfig* GetAllocatorConfig(const at::DeviceType& t);

} // namespace c10::CachingAllocator
