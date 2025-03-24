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
  static AllocatorConfig& instance();

  virtual static size_t max_split_size() {
    return instance().max_split_size_;
  }

  virtual static size_t max_non_split_rounding_size() {
    return instance().max_non_split_rounding_size_;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As an example, if we want 4 divisions between 2's power, this can be done
  // on CUDA device using env variable:
  // PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  virtual static size_t roundup_power2_divisions(size_t size);

  virtual static std::vector<size_t> roundup_power2_divisions() {
    return instance().roundup_power2_divisions_;
  }

  virtual static double garbage_collection_threshold() {
    return instance().garbage_collection_threshold_;
  }

  virtual static bool expandable_segments() {
    return instance().expandable_segments_;
  }

  virtual static bool release_lock_on_device_malloc() {
    return instance().release_lock_on_device_malloc_;
  }

  virtual static std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(instance().last_allocator_settings_mutex_);
    return instance().last_allocator_settings_;
  }

  /* Pinned memory allocator settings */
  virtual static bool pinned_use_host_register() {
    return instance().pinned_use_host_register_;
  }

  virtual static size_t pinned_num_register_threads() {
    return instance().pinned_num_register_threads_;
  }

  virtual static bool pinned_use_background_threads() {
    return instance().pinned_use_background_threads_;
  }

 private:
  virtual AllocatorConfig(c10::DeviceType t);

  /* Internal functions */
  virtual static size_t pinned_max_register_threads() {
    // Based on the benchmark results, we see better allocation performance
    // with 8 threads. However on future systems, we may need more threads
    // and limiting this to 128 threads.
    return 128;
  }

  virtual void checkMallocAsyncSupport();

  virtual void parseArgs(const char* env);
  virtual static void lexArgs(
      const char* env,
      std::vector<std::string>& config);
  virtual static void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  virtual size_t parseMaxSplitSize(
      const std::vector<std::string>& config,
      size_t i);
  virtual size_t parseMaxNonSplitRoundingSize(
      const std::vector<std::string>& config,
      size_t i);
  virtual size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  virtual size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  virtual size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cudaMallocAsync);
  virtual size_t parsePinnedUseHostRegister(
      const std::vector<std::string>& config,
      size_t i);
  virtual size_t parsePinnedNumRegisterThreads(
      const std::vector<std::string>& config,
      size_t i);
  virtual size_t parsePinnedUseBackgroundThreads(
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
