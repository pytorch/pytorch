#include <c10/core/AllocatorConfig.h>
#include <c10/core/DeviceType.h>

#include <array>

namespace c10::CachingAllocator {

namespace {
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;
constexpr size_t kMB = 1024 * 1024ul;
constexpr size_t kRoundUpPowerOfTwoStart = 1 * kMB; // 1MB
constexpr size_t kRoundUpPowerOfTwoEnd = 64 * 1024 * kMB; // 64GB
constexpr size_t kPinnedMaxRegisterThreads = 128;
} // anonymous namespace

AllocatorConfig::AllocatorConfig(c10::DeviceType t)
    : max_split_size_(std::numeric_limits<size_t>::max()),
      max_non_split_rounding_size_(kLargeBuffer),
      garbage_collection_threshold_(0),
      expandable_segments_(false),
      release_lock_on_device_malloc_(false),
      pinned_use_host_register_(false),
      pinned_num_register_threads_(1),
      pinned_use_background_threads_(false),
      device_type_(t) {
  roundup_power2_divisions_.assign(kRoundUpPowerOfTwoIntervals, 0);
}

size_t AllocatorConfig::roundup_power2_divisions(size_t size) {
  size_t log_size = (63 - llvm::countLeadingZeros(size));

  // Our intervals start at 1MB and end at 64GB
  const size_t interval_start =
      63 - llvm::countLeadingZeros(kRoundUpPowerOfTwoStart);
  const size_t interval_end =
      63 - llvm::countLeadingZeros(kRoundUpPowerOfTwoEnd);
  TORCH_CHECK(
      interval_end - interval_start == kRoundUpPowerOfTwoIntervals,
      "kRoundUpPowerOfTwoIntervals mismatch");

  auto index = (log_size > interval_start) ? (log_size - interval_start) : 0ul;
  index = std::min(index, kRoundUpPowerOfTwoIntervals - 1);
  return roundup_power2_divisions_[index];
}

void AllocatorConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void AllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i] == std::string(1, c),
      "Error parsing CachingAllocator::AllocatorConfig settings, expected ",
      c);
}

size_t AllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  constexpr size_t min_allowed_split_size_mb = kLargeBuffer / kMB;
  constexpr size_t max_allowed_split_size_mb =
      std::numeric_limits<size_t>::max() / kMB;

  if (++i < config.size()) {
    size_t val_env = stoi(config[i]);
    TORCH_CHECK(
        val_env >= min_allowed_split_size_mb,
        "CachingAllocator option max_split_size_mb too small, must be >= ",
        min_allowed_split_size_mb);
    val_env = std::min(val_env, max_allowed_split_size_mb);
    max_split_size_ = val_env * kMB;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value");
  }
  return i;
}

size_t AllocatorConfig::parseMaxNonSplitRoundingSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  constexpr size_t min_allowed_split_size_mb = kLargeBuffer / kMB;
  constexpr size_t max_allowed_split_size_mb =
      std::numeric_limits<size_t>::max() / kMB;

  if (++i < config.size()) {
    size_t val_env = stoi(config[i]);
    TORCH_CHECK(
        val_env >= min_allowed_split_size_mb,
        "CachingAllocator option max_non_split_rounding_mb too small, must be >= ",
        min_allowed_split_size_mb);
    val_env = std::min(val_env, max_allowed_split_size_mb);
    max_non_split_rounding_size_ = val_env * kMB;
  } else {
    TORCH_CHECK(false, "Error, expecting max_non_split_rounding_mb value");
  }
  return i;
}

size_t AllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');

  if (++i < config.size()) {
    double val_env = stod(config[i]);
    TORCH_CHECK(
        val_env > 0 && val_env < 1.0,
        "garbage_collect_threshold is invalid, set it in (0.0, 1.0)");
    garbage_collection_threshold_ = val_env;
  } else {
    TORCH_CHECK(false, "Error, expecting garbage_collection_threshold value");
  }
  return i;
}

size_t AllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  bool first_value = true;

  if (++i < config.size()) {
    if (std::string_view(config[i]) == "[") {
      size_t last_index = 0;
      // NOLINTNEXTLINE(bugprone-inc-dec-in-conditions)
      while (++i < config.size() && std::string_view(config[i]) != "]") {
        const std::string& val1 = config[i];
        size_t val2 = 0;

        consumeToken(config, ++i, ':');
        if (++i < config.size()) {
          val2 = stoi(config[i]);
        } else {
          TORCH_CHECK(false, "Error parsing roundup_power2_divisions value");
        }
        TORCH_CHECK(
            val2 == 0 || llvm::isPowerOf2_64(val2),
            "For roundups, the divisons has to be power of 2 or 0 to disable roundup ");

        if (std::string_view(val1) == ">") {
          std::fill(
              std::next(
                  roundup_power2_divisions_.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              roundup_power2_divisions_.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ");

          size_t index = 63 - llvm::countLeadingZeros(val1_long);
          index = std::clamp(index, 0ul, roundup_power2_divisions_.size() - 1);

          if (first_value) {
            std::fill(
                roundup_power2_divisions_.begin(),
                std::next(
                    roundup_power2_divisions_.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          if (index < roundup_power2_divisions_.size()) {
            roundup_power2_divisions_[index] = val2;
          }
          last_index = index;
        }

        if (std::string_view(config[i + 1]) != "]") {
          consumeToken(config, ++i, ',');
        }
      }
    } else { // Keep this for backwards compatibility
      size_t val1 = stoi(config[i]);
      TORCH_CHECK(
          llvm::isPowerOf2_64(val1),
          "For roundups, the divisons has to be power of 2 ");
      std::fill(
          roundup_power2_divisions_.begin(),
          roundup_power2_divisions_.end(),
          val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value");
  }
  return i;
}

size_t AllocatorConfig::parseAllocatorBackendConfig(
    const std::vector<std::string>& config,
    size_t i,
    bool& used_MallocAsync) {
  std::string backend_name =
      c10::DeviceTypeName(device_type_, true) + "MallocAsync";
  consumeToken(config, ++i, ':');

  if (++i < config.size()) {
    TORCH_CHECK(
        ((config[i] == "native") || (config[i] == backend_name)),
        "Unknown allocator backend, "
        "options are native and ",
        backend_name);
    used_MallocAsync = (config[i] == backend_name);
    if (used_MallocAsync) {
      checkMallocAsyncSupport();
    }
    // TORCH_INTERNAL_ASSERT(
    //     config[i] == get()->name(),
    //     "Allocator backend parsed at runtime != "
    //     "allocator backend parsed at load time");
  } else {
    TORCH_CHECK(false, "Error parsing allocator backend value");
  }
  return i;
}

size_t AllocatorConfig::parsePinnedUseHostRegister(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  std::string env_name = "pinned_use_" +
      c10::DeviceTypeName(device_type_, true) + "_host_register";
  if (++i < config.size()) {
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for ",
        env_name);
    pinned_use_host_register_ = (config[i] == "True");
  } else {
    TORCH_CHECK(false, "Error, expecting ", env_name, " value");
  }
  return i;
}

size_t AllocatorConfig::parsePinnedNumRegisterThreads(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val_env = stoi(config[i]);
    TORCH_CHECK(
        llvm::isPowerOf2_64(val_env),
        "Number of register threads has to be power of 2");
    // Based on the benchmark results, we see better allocation performance
    // with 8 threads. However on future systems, we may need more threads
    // and limiting this to 128 threads.
    TORCH_CHECK(
        val_env <= kPinnedMaxRegisterThreads,
        "Number of register threads should be less than or equal to ",
        kPinnedMaxRegisterThreads);
    pinned_num_register_threads_ = val_env;
  } else {
    TORCH_CHECK(false, "Error, expecting pinned_num_register_threads value");
  }
  return i;
}

size_t AllocatorConfig::parsePinnedUseBackgroundThreads(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for pinned_use_background_threads");
    pinned_use_background_threads_ = (config[i] == "True");
  } else {
    TORCH_CHECK(false, "Error, expecting pinned_use_background_threads value");
  }
  return i;
}

void AllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  max_split_size_ = std::numeric_limits<size_t>::max();
  roundup_power2_divisions_.assign(kRoundUpPowerOfTwoIntervals, 0);
  garbage_collection_threshold_ = 0;
  bool used_MallocAsync = false;
  bool used_native_specific_option = false;

  if (env == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(last_allocator_settings_mutex_);
    last_allocator_settings_ = env;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    if (config_item_view == "max_split_size_mb") {
      i = parseMaxSplitSize(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "max_non_split_rounding_mb") {
      i = parseMaxNonSplitRoundingSize(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "garbage_collection_threshold") {
      i = parseGarbageCollectionThreshold(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "roundup_power2_divisions") {
      i = parseRoundUpPower2Divisions(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "backend") {
      i = parseAllocatorBackendConfig(config, i, used_MallocAsync);
    } else if (config_item_view == "expandable_segments") {
      used_native_specific_option = true;
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for expandable_segments");
      config_item_view = config[i];
      expandable_segments_ = (config_item_view == "True");
    } else if (
        config_item_view ==
        ("release_lock_on_" + c10::DeviceTypeName(device_type_, true) +
         "malloc")) {
      used_native_specific_option = true;
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for release_lock_on_",
          device_type_,
          "malloc");
      config_item_view = config[i];
      release_lock_on_device_malloc_ = (config_item_view == "True");
    } else if (
        config_item_view ==
        ("pinned_use_" + c10::DeviceTypeName(device_type_, true) +
         "_host_register")) {
      i = parsePinnedUseHostRegister(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "pinned_num_register_threads") {
      i = parsePinnedNumRegisterThreads(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "pinned_use_background_threads") {
      i = parsePinnedUseBackgroundThreads(config, i);
      used_native_specific_option = true;
    } else {
      TORCH_CHECK(
          false, "Unrecognized CachingAllocator option: ", config_item_view);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (used_MallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:",
        device_type_,
        "MallocAsync ignores max_split_size_mb, roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

static std::array<AllocatorConfig*, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_configs{};

void SetAllocatorConfig(at::DeviceType t, AllocatorConfig* allocator_config) {
  TORCH_INTERNAL_ASSERT(!allocator_configs[static_cast<int>(t)]);
  allocator_configs[static_cast<int>(t)] = allocator_config;
}

AllocatorConfig* GetAllocatorConfig(const at::DeviceType& t) {
  auto* allocator_config = allocator_configs[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocator_config, "AllocatorConfig for ", t, " is not set.");
  return allocator_config;
}

} // namespace c10::CachingAllocator
