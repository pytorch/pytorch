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
 * This class configures memory allocation for both device and host memory. A
 * single `AllocatorConfig` instance is shared across all accelerator backends,
 * such as CUDA and XPU, under the assumption that relevant environment
 * variables apply uniformly to all accelerators. Each backend can also extend
 * `AllocatorConfig` with its own device-specific configuration—for example,
 * CUDA uses `CUDAAllocatorConfig`—via the `setAllocatorSettings` and
 * `getAllocatorSettings` APIs.
 *
 * The recommended design is to place common configurations in
 * `AllocatorConfig`, and backend-specific configurations in corresponding
 * device-specific classes, such as `CUDAAllocatorConfig`, etc.
 *
 * It is designed to *ONLY* contain configuration options that can be set via
 * environment variables.
 *
 * Naming Convention:
 * - Public API names in `AllocatorConfig` should be device-generic.
 * - Members prefixed with `pinned_` are specific to the host/pinned allocator.
 * - Environment variable names should also be device-generic to ensure
 *     consistency across different hardware backends.
 *
 * Environment Variables:
 * - The default environment variable for configuration is `PYTORCH_ALLOC_CONF`.
 * - For backward compatibility, `PYTORCH_CUDA_ALLOC_CONF` is also supported
 *     with lower priority.
 */

class C10_API AllocatorConfig {
 public:
  static AllocatorConfig& instance();

  C10_DISABLE_COPY_AND_ASSIGN(AllocatorConfig);
  AllocatorConfig(AllocatorConfig&&) = delete;
  AllocatorConfig& operator=(AllocatorConfig&&) = delete;
  ~AllocatorConfig() = default;

  /* Device allocator settings */

  // Returns the maximum block size (in MB) that is allowed to be split. The
  // default is unlimited (all blocks can be split).
  static size_t max_split_size() {
    return instance().max_split_size_;
  }

  // Returns the maximum block size (in MB) that is allowed to be rounded up
  // without requiring splitting when searching for a free block. The default is
  // 20 MiB.
  static size_t max_non_split_rounding_size() {
    return instance().max_non_split_rounding_size_;
  }

  // Return the number of divisions used when rounding up allocation sizes (in
  // MB) to the nearest power-of-2 boundary.
  static size_t roundup_power2_divisions(size_t size);

  // Returns the vector of division factors used for rounding up allocation
  // sizes. These divisions apply to size intervals between 1MB and 64GB.
  static std::vector<size_t> roundup_power2_divisions() {
    return instance().roundup_power2_divisions_;
  }

  // Returns the threshold that triggers garbage collection when the ratio of
  // used memory to maximum allowed memory exceeds this value. The default is 0,
  // meaning no garbage collection is triggered. The value should be in the
  // range (0.0, 1.0).
  static double garbage_collection_threshold() {
    return instance().garbage_collection_threshold_;
  }

  // Returns whether the expandable segment feature is enabled. This allows the
  // allocator to start with one segment that grows as needed, rather than
  // creating a new segment for each allocation. Default is false (expandable
  // segments disabled).
  static bool use_expandable_segments() {
    return instance().use_expandable_segments_;
  }

  /* Host allocator settings */

  // Returns whether the pinned host allocator uses background threads for
  // processing events. This is useful for improving performance in scenarios
  // where many small allocations are made. Default is false (background threads
  // disabled).
  static bool pinned_use_background_threads() {
    return instance().pinned_use_background_threads_;
  }

  /* Settings for both device and host allocator */

  // Returns the current allocator settings as a string. This string is useful
  // to expand device-specific allocator configurations
  static std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(instance().last_allocator_settings_mutex_);
    return instance().last_allocator_settings_;
  }

  // Parses the environment variable `env` to update the allocator settings.
  // If the environment variable is not set, it does nothing.
  // The configuration string should be a comma-separated list of key-value
  // pairs, where each key is a configuration option and the value is the
  // corresponding setting. For example:
  // "max_split_size_mb:100,max_non_split_rounding_mb:20,garbage_collection_threshold:0.5,roundup_power2_divisions:[64:8,256:4,1024:4,>:1],expandable_segments:true,pinned_use_background_threads:true"
  void parseArgs(const std::optional<std::string>& env);

 private:
  AllocatorConfig();

  /* Internal functions */

  void lexArgs(const std::string& env, std::vector<std::string>& config);

  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);

  /* Internal functions for device allocator */

  // Parse `max_split_size_mb` from environment variable.
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  // Parse `max_non_split_rounding_mb` from environment variable.
  size_t parseMaxNonSplitRoundingSize(
      const std::vector<std::string>& config,
      size_t i);
  // Parse `garbage_collection_threshold` from environment variable.
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  // Parse `roundup_power2_divisions` from environment variable.
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  // Parse `expandable_segments` from environment variable.
  size_t parseExpandableSegments(
      const std::vector<std::string>& config,
      size_t i);

  /* Internal functions for host allocator */

  // Parse `pinned_use_background_threads` from environment variable.
  size_t parsePinnedUseBackgroundThreads(
      const std::vector<std::string>& config,
      size_t i);

  /* The following members are specifically used for the device allocator. */

  // The maximum block size that is allowed to be split.
  std::atomic<size_t> max_split_size_{std::numeric_limits<size_t>::max()};
  // The maximum allowable extra size of a memory block without requiring
  // splitting when searching for a free block.
  std::atomic<size_t> max_non_split_rounding_size_{kLargeBuffer};
  // Used to store how memory allocations of different sizes should be rounded
  // up to the nearest power of 2 divisions.
  std::vector<size_t> roundup_power2_divisions_;
  // The threshold that triggers garbage collection when the ratio of used
  // memory to maximum allowed memory exceeds this value.
  std::atomic<double> garbage_collection_threshold_{0};
  // A flag to enable expandable segments feature.
  std::atomic<bool> use_expandable_segments_{false};

  /* The following members are specifically used for the host allocator. */

  // A flag to enable background thread for processing events.
  std::atomic<bool> pinned_use_background_threads_{false};

  /* The following members are used for both device and host allocator. */

  // Record the last allocator config environment setting.
  std::mutex last_allocator_settings_mutex_;
  std::string last_allocator_settings_;
};

C10_API inline void setAllocatorSettings(const std::string& env) {
  AllocatorConfig::instance().parseArgs(env.c_str());
}

C10_API inline std::string getAllocatorSettings(const std::string& env) {
  return AllocatorConfig::instance().last_allocator_settings();
}

} // namespace c10::CachingAllocator
