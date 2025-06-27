#include <c10/core/AllocatorConfig.h>
#include <c10/core/DeviceType.h>
#include <c10/util/env.h>

#include <array>
#include <cstdlib>

namespace c10::CachingAllocator {

namespace {
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;
constexpr size_t kMB = 1024 * 1024ul;
constexpr size_t kRoundUpPowerOfTwoStart = 1 * kMB; // 1MB
constexpr size_t kRoundUpPowerOfTwoEnd = 64 * 1024ul * kMB; // 64GB
} // anonymous namespace

AllocatorConfig& AllocatorConfig::instance() {
  static AllocatorConfig instance;
#define C10_ALLOCATOR_CONFIG_PARSE_ENV(env, deprecated)                       \
  auto env##_name = c10::utils::get_env(#env);                                \
  if (env##_name.has_value()) {                                               \
    if (deprecated) {                                                         \
      TORCH_WARN_ONCE(#env " is deprecated, use PYTORCH_ALLOC_CONF instead"); \
    }                                                                         \
    instance.parseArgs(env##_name);                                           \
    return true;                                                              \
  }
  static bool env_flag [[maybe_unused]] = []() {
    C10_ALLOCATOR_CONFIG_PARSE_ENV(PYTORCH_ALLOC_CONF, false)
    // Keep this for backwards compatibility
    C10_ALLOCATOR_CONFIG_PARSE_ENV(PYTORCH_CUDA_ALLOC_CONF, /*deprecated=*/true)
    C10_ALLOCATOR_CONFIG_PARSE_ENV(PYTORCH_HIP_ALLOC_CONF, /*deprecated=*/true)
    return false;
  }();
#undef C10_ALLOCATOR_CONFIG_PARSE_ENV
  return instance;
}

AllocatorConfig::AllocatorConfig() {
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
  return instance().roundup_power2_divisions_[index];
}

void AllocatorConfig::lexArgs(
    const std::string& env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  for (char ch : env) {
    if (ch == ',' || ch == ':' || ch == '[' || ch == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, ch);
    } else if (ch != ' ') {
      buf.emplace_back(ch);
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
            "For roundups, the divisions has to be power of 2 or 0 to disable roundup ");

        if (std::string_view(val1) == ">") {
          std::fill(
              std::next(
                  roundup_power2_divisions_.begin(),
                  static_cast<std::vector<size_t>::difference_type>(
                      last_index + 1)),
              roundup_power2_divisions_.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ");

          size_t index = 63 - llvm::countLeadingZeros(val1_long);
          index = std::clamp(
              index, size_t{0}, roundup_power2_divisions_.size() - 1);

          if (first_value) {
            std::fill(
                roundup_power2_divisions_.begin(),
                std::next(
                    roundup_power2_divisions_.begin(),
                    static_cast<std::vector<size_t>::difference_type>(index)),
                val2);
            first_value = false;
          }
          roundup_power2_divisions_[index] = val2;
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
          "For roundups, the divisions has to be power of 2 ");
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

size_t AllocatorConfig::parseExpandableSegments(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for expandable_segments");
    use_expandable_segments_ = (config[i] == "True");
  } else {
    TORCH_CHECK(false, "Error, expecting expandable_segments value");
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

void AllocatorConfig::parseArgs(const std::optional<std::string>& env) {
  // The following option will be reset to its default value if not explicitly
  // set each time.
  max_split_size_ = std::numeric_limits<size_t>::max();
  roundup_power2_divisions_.assign(kRoundUpPowerOfTwoIntervals, 0);
  garbage_collection_threshold_ = 0;

  if (!env.has_value()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(last_allocator_settings_mutex_);
    last_allocator_settings_ = env.value();
  }

  std::vector<std::string> config;
  lexArgs(env.value(), config);

  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    if (config_item_view == "max_split_size_mb") {
      i = parseMaxSplitSize(config, i);
    } else if (config_item_view == "max_non_split_rounding_mb") {
      i = parseMaxNonSplitRoundingSize(config, i);
    } else if (config_item_view == "garbage_collection_threshold") {
      i = parseGarbageCollectionThreshold(config, i);
    } else if (config_item_view == "roundup_power2_divisions") {
      i = parseRoundUpPower2Divisions(config, i);
    } else if (config_item_view == "expandable_segments") {
      i = parseExpandableSegments(config, i);
    } else if (config_item_view == "pinned_use_background_threads") {
      i = parsePinnedUseBackgroundThreads(config, i);
    } else {
      TORCH_CHECK(
          false, "Unrecognized CachingAllocator option: ", config_item_view);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }
}

} // namespace c10::CachingAllocator
