#pragma once

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <c10/util/llvmMathExtras.h>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10::CachingAllocator {

// "small" allocations are packed in 2 MiB blocks
constexpr size_t kSmallBuffer = 2097152;
// all sizes are rounded to at least 512 bytes
constexpr size_t kMinBlockSize = 512;
// largest "small" allocation is 1 MiB
constexpr size_t kSmallSize = 1048576;
// allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kMinLargeAlloc = 10485760;
// round up large allocations to 2 MiB
constexpr size_t kRoundLarge = 2097152;

// A utility class for tokenizing allocator configuration strings into discrete
// parts. For example, the config string:
//   "key1:val1,key2:[val2,val3]"
// is tokenized into:
//   "key1", ":", "val1", ",", "key2", ":", "[", "val2", ",", "val3", "]",
//
// Tokens include keys, values, and special characters (':', ',', '[', ']').
// Whitespace is ignored.
class ConfigTokenizer {
 public:
  explicit ConfigTokenizer(const std::string& env) {
    std::string buffer;
    for (char ch : env) {
      if (ch == ',' || ch == ':' || ch == '[' || ch == ']') {
        if (!buffer.empty()) {
          config_.emplace_back(std::move(buffer));
          buffer.clear();
        }
        config_.emplace_back(1, ch);
      } else if (!std::isspace(static_cast<unsigned char>(ch))) {
        buffer += ch;
      }
    }
    if (!buffer.empty()) {
      config_.emplace_back(std::move(buffer));
    }
  }

  const std::string& operator[](size_t i) const {
    TORCH_INTERNAL_ASSERT(
        i < config_.size(), "Index out of bounds in ConfigTokenizer");
    return config_[i];
  }

  size_t size() const {
    return config_.size();
  }

  bool checkToken(size_t i, const std::string& token) const {
    checkIndex(i);
    return config_[i] == token;
  }

  size_t toSizeT(size_t i) const {
    checkIndex(i);
    return std::stoull(config_[i]);
  }

  double toDouble(size_t i) const {
    checkIndex(i);
    return std::stod(config_[i]);
  }

  bool toBool(size_t i) const {
    checkIndex(i);
    const auto& token = config_[i];
    if (token == "True") {
      return true;
    } else if (token == "False") {
      return false;
    } else {
      TORCH_CHECK_VALUE(
          false,
          "Expected 'True' or 'False' at index ",
          i,
          " in ConfigTokenizer but got '",
          token,
          "'");
    }
  }

  // Skips the current token group and returns the index of the value token.
  // Assumes the current index `i` points to a key name in a key-value pair.
  size_t skipKey(size_t i) const {
    // Expect a colon after the key
    checkToken(++i, ":");

    ++i; // Move to the value
    checkIndex(i);
    if (config_[i] != "[") {
      // Value is a single token (not a list) -> return its index
      return i;
    }

    // Skip tokens inside the list until matching ']'
    // NOLINTNEXTLINE(bugprone-inc-dec-in-conditions)
    while (++i < config_.size() && config_[i] != "]") {
    }

    TORCH_INTERNAL_ASSERT(
        i < config_.size(),
        "Expected closing bracket ']' in ConfigTokenizer but reached end of config");

    return i; // Return the index of the closing ']'
  }

 private:
  void checkIndex(size_t i) const {
    TORCH_INTERNAL_ASSERT(
        i < config_.size(), "Index out of bounds in ConfigTokenizer");
  }

  std::vector<std::string> config_;
};

/**
 * Note [AcceleratorAllocatorConfig design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * This class configures memory allocation for both device and host memory. A
 * single `AcceleratorAllocatorConfig` instance is shared across all accelerator
 * backends, such as CUDA and XPU, under the assumption that relevant
 * environment variables apply uniformly to all accelerators. Device-specific
 * configuration extensions are supported via hooks (see
 * `registerDeviceConfigParserHook`).
 *
 * Recommended design:
 * - Place common configurations in `AcceleratorAllocatorConfig`.
 * - Extend backend-specific configurations in corresponding device-specific
 *     classes, such as `CUDAAllocatorConfig`, etc.
 *
 * Scope:
 * - Configuration options must be environment-variable driven.
 *
 * Naming Convention:
 * - Public API names in `AcceleratorAllocatorConfig` should be device-generic.
 * - Members prefixed with `pinned_` are specific to the host/pinned allocator.
 * - Environment variable names should be generic across backends.
 * - Comma-separated key-value pairs in the format: `key:value`. Use square
 *     brackets `[]` for list values Example: `key1:123, key2:[val1,val2]`
 *
 * Environment Variables:
 * - The primary environment variable for configuration is `PYTORCH_ALLOC_CONF`.
 * - For backward compatibility, `PYTORCH_CUDA_ALLOC_CONF` is also supported
 *     with lower priority.
 */

class C10_API AcceleratorAllocatorConfig {
 public:
  static AcceleratorAllocatorConfig& instance();

  C10_DISABLE_COPY_AND_ASSIGN(AcceleratorAllocatorConfig);
  AcceleratorAllocatorConfig(AcceleratorAllocatorConfig&&) = delete;
  AcceleratorAllocatorConfig& operator=(AcceleratorAllocatorConfig&&) = delete;
  ~AcceleratorAllocatorConfig() = default;

  /* Device allocator settings */

  static size_t large_segment_size() {
    return instance().large_segment_size_;
  }

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
  static const std::vector<size_t>& roundup_power2_divisions() {
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

  // Use `Construct On First Use Idiom` to avoid `Static Initialization Order`
  // issue.
  static std::unordered_set<std::string>& getMutableKeys() {
    static std::unordered_set<std::string> keys{
        "large_segment_size_mb",
        "max_split_size_mb",
        "max_non_split_rounding_mb",
        "garbage_collection_threshold",
        "roundup_power2_divisions",
        "expandable_segments",
        "pinned_use_background_threads"};
    return keys;
  }

  // Returns the set of valid keys for the allocator configuration.
  // This set is used to validate the presence and correctness of keys in
  // device-specific configuration parsers.
  static const std::unordered_set<std::string>& getKeys() {
    return getMutableKeys();
  }

  // Registers a device-specific configuration parser hook and its key. This
  // allows backends to parse additional device-specific configuration options
  // from the environment variable. The hook should be a function that takes a
  // string (the environment variable value) and parses it to set
  // device-specific configuration options. The hook will be called when the
  // environment variable is parsed. If a hook is already registered, it will be
  // replaced with the new one.
  static void registerDeviceConfigParserHook(
      std::function<void(const std::string&)>&& hook,
      const std::unordered_set<std::string>& keys) {
    device_config_parser_hook_ = std::move(hook);
    auto& mutable_keys = getMutableKeys();
    for (auto& key : keys) {
      TORCH_CHECK_VALUE(
          mutable_keys.insert(key).second,
          "Duplicated key '",
          key,
          "' found in device-specific configuration parser hook registration");
    }
  }

  // Calls the registered device-specific configuration parser hook with the
  // provided environment string. This allows backends to parse additional
  // device-specific configuration options from the environment variable.
  // If no hook is registered, this function does nothing.
  static void callDeviceConfigParserHook(const std::string& env) {
    if (device_config_parser_hook_) {
      device_config_parser_hook_(env);
    }
  }

  // Parses the environment variable `env` to update the allocator settings.
  // If the environment variable is not set, it does nothing.
  // The configuration string should be a comma-separated list of key-value
  // pairs, where each key is a configuration option and the value is the
  // corresponding setting. For example:
  // "max_split_size_mb:100,max_non_split_rounding_mb:20,garbage_collection_threshold:0.5,roundup_power2_divisions:[64:8,256:4,1024:4,>:1],expandable_segments:true,pinned_use_background_threads:true"
  void parseArgs(const std::string& env);

 private:
  AcceleratorAllocatorConfig();

  /* Internal functions for device allocator */

  // Parse `large_segment_size_mb` from environment variable.
  size_t parseLargeSegmentSize(const ConfigTokenizer& tokenizer, size_t i);
  // Parse `max_split_size_mb` from environment variable.
  size_t parseMaxSplitSize(const ConfigTokenizer& tokenizer, size_t i);
  // Parse `max_non_split_rounding_mb` from environment variable.
  size_t parseMaxNonSplitRoundingSize(
      const ConfigTokenizer& tokenizer,
      size_t i);
  // Parse `garbage_collection_threshold` from environment variable.
  size_t parseGarbageCollectionThreshold(
      const ConfigTokenizer& tokenizer,
      size_t i);
  // Parse `roundup_power2_divisions` from environment variable.
  size_t parseRoundUpPower2Divisions(
      const ConfigTokenizer& tokenizer,
      size_t i);
  // Parse `expandable_segments` from environment variable.
  size_t parseExpandableSegments(const ConfigTokenizer& tokenizer, size_t i);

  /* Internal functions for host allocator */

  // Parse `pinned_use_background_threads` from environment variable.
  size_t parsePinnedUseBackgroundThreads(
      const ConfigTokenizer& tokenizer,
      size_t i);

  /* The following members are specifically used for the device allocator. */

  // "large" allocations may be packed in blocks of this size
  std::atomic<size_t> large_segment_size_{20971520}; // 20 MB by default
  // The maximum block size that is allowed to be split.
  std::atomic<size_t> max_split_size_{std::numeric_limits<size_t>::max()};
  // The maximum allowable extra size of a memory block without requiring
  // splitting when searching for a free block.
  std::atomic<size_t> max_non_split_rounding_size_;
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

  // Optional hook for parsing additional device-specific allocator settings.
  // This allows backends (e.g., CUDA, XPU) to register a custom parser for
  // their own environment configuration extensions.
  inline static std::function<void(const std::string&)>
      device_config_parser_hook_{nullptr};
};

C10_API inline void setAllocatorSettings(const std::string& env) {
  AcceleratorAllocatorConfig::instance().parseArgs(env);
  AcceleratorAllocatorConfig::callDeviceConfigParserHook(env);
}

C10_API inline std::string getAllocatorSettings() {
  return AcceleratorAllocatorConfig::instance().last_allocator_settings();
}

struct DeviceConfigParserHookRegistry {
  explicit DeviceConfigParserHookRegistry(
      std::function<void(const std::string&)>&& hook,
      const std::unordered_set<std::string>& keys) {
    // Use static method to avoid static initialization order fiasco issues
    AcceleratorAllocatorConfig::registerDeviceConfigParserHook(
        std::move(hook), keys);
  }
};

// Assume each config parser has `parseArgs` and `getKeys` methods
#define REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(parser_cls)      \
  namespace {                                                 \
  static at::CachingAllocator::DeviceConfigParserHookRegistry \
      g_device_config_parse_hook_registry_instance(           \
          [](const std::string& env) {                        \
            parser_cls::instance().parseArgs(env);            \
          },                                                  \
          parser_cls::getKeys());                             \
  }

} // namespace c10::CachingAllocator
