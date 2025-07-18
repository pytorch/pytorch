#include <c10/core/AllocatorConfig.h>
#include <c10/core/DeviceType.h>
#include <c10/util/env.h>

namespace c10::CachingAllocator {

namespace {
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;
constexpr size_t kMB = 1024 * 1024ul;
constexpr size_t kRoundUpPowerOfTwoStart = 1 * kMB; // 1MB
constexpr size_t kRoundUpPowerOfTwoEnd = 64 * 1024ul * kMB; // 64GB
} // anonymous namespace

AcceleratorAllocatorConfig& AcceleratorAllocatorConfig::instance() {
  static AcceleratorAllocatorConfig instance;
#define C10_ALLOCATOR_CONFIG_PARSE_ENV(env, deprecated)                       \
  auto env##_name = c10::utils::get_env(#env);                                \
  if (env##_name.has_value()) {                                               \
    if (deprecated) {                                                         \
      TORCH_WARN_ONCE(#env " is deprecated, use PYTORCH_ALLOC_CONF instead"); \
    }                                                                         \
    instance.parseArgs(env##_name.value());                                   \
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

AcceleratorAllocatorConfig::AcceleratorAllocatorConfig() {
  roundup_power2_divisions_.assign(kRoundUpPowerOfTwoIntervals, 0);
}

size_t AcceleratorAllocatorConfig::roundup_power2_divisions(size_t size) {
  size_t log_size = (63 - llvm::countLeadingZeros(size));

  // Our intervals start at 1MB and end at 64GB
  const size_t interval_start =
      63 - llvm::countLeadingZeros(kRoundUpPowerOfTwoStart);
  const size_t interval_end =
      63 - llvm::countLeadingZeros(kRoundUpPowerOfTwoEnd);
  TORCH_CHECK(
      interval_end - interval_start == kRoundUpPowerOfTwoIntervals,
      "kRoundUpPowerOfTwoIntervals mismatch");

  size_t index =
      (log_size > interval_start) ? (log_size - interval_start) : 0ul;
  index = std::min(index, kRoundUpPowerOfTwoIntervals - 1);
  return instance().roundup_power2_divisions_[index];
}

size_t AcceleratorAllocatorConfig::parseMaxSplitSize(
    const ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  constexpr size_t min_allowed_split_size_mb = kLargeBuffer / kMB;
  constexpr size_t max_allowed_split_size_mb =
      std::numeric_limits<size_t>::max() / kMB;

  size_t val_env = tokenizer.toSizeT(++i);
  TORCH_CHECK(
      val_env >= min_allowed_split_size_mb,
      "CachingAllocator option max_split_size_mb too small, must be >= ",
      min_allowed_split_size_mb);
  val_env = std::min(val_env, max_allowed_split_size_mb);
  max_split_size_ = val_env * kMB;

  return i;
}

size_t AcceleratorAllocatorConfig::parseMaxNonSplitRoundingSize(
    const ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  constexpr size_t min_allowed_split_size_mb = kLargeBuffer / kMB;
  constexpr size_t max_allowed_split_size_mb =
      std::numeric_limits<size_t>::max() / kMB;

  size_t val_env = tokenizer.toSizeT(++i);
  TORCH_CHECK(
      val_env >= min_allowed_split_size_mb,
      "CachingAllocator option max_non_split_rounding_mb too small, must be >= ",
      min_allowed_split_size_mb);
  val_env = std::min(val_env, max_allowed_split_size_mb);
  max_non_split_rounding_size_ = val_env * kMB;

  return i;
}

size_t AcceleratorAllocatorConfig::parseGarbageCollectionThreshold(
    const ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  double val_env = tokenizer.toDouble(++i);
  TORCH_CHECK(
      val_env > 0 && val_env < 1.0,
      "garbage_collect_threshold is invalid, set it in (0.0, 1.0)");
  garbage_collection_threshold_ = val_env;

  return i;
}

size_t AcceleratorAllocatorConfig::parseRoundUpPower2Divisions(
    const ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  bool first_value = true;

  if (tokenizer[++i] == "[") {
    size_t last_index = 0;
    // NOLINTNEXTLINE(bugprone-inc-dec-in-conditions)
    while (++i < tokenizer.size() && tokenizer[i] != "]") {
      size_t value_index = i;
      tokenizer.checkToken(++i, ":");
      size_t value = tokenizer.toSizeT(++i);
      TORCH_CHECK(
          value == 0 || llvm::isPowerOf2_64(value),
          "For roundups, the divisions has to be power of 2 or 0 to disable roundup ");

      if (tokenizer[value_index] == ">") {
        std::fill(
            std::next(
                roundup_power2_divisions_.begin(),
                static_cast<std::vector<size_t>::difference_type>(
                    last_index + 1)),
            roundup_power2_divisions_.end(),
            value);
      } else {
        size_t boundary = tokenizer.toSizeT(value_index);
        TORCH_CHECK(
            llvm::isPowerOf2_64(boundary),
            "For roundups, the intervals have to be power of 2 ");

        size_t index = 63 - llvm::countLeadingZeros(boundary);
        index =
            std::clamp(index, size_t{0}, roundup_power2_divisions_.size() - 1);

        if (first_value) {
          std::fill(
              roundup_power2_divisions_.begin(),
              std::next(
                  roundup_power2_divisions_.begin(),
                  static_cast<std::vector<size_t>::difference_type>(index)),
              value);
          first_value = false;
        }
        roundup_power2_divisions_[index] = value;
        last_index = index;
      }

      if (tokenizer[i + 1] != "]") {
        tokenizer.checkToken(++i, ",");
      }
    }
    TORCH_INTERNAL_ASSERT(
        i < tokenizer.size(),
        "Expected closing bracket ']' in ConfigTokenizer but reached end of config");
  } else { // Keep this for backwards compatibility
    size_t value = tokenizer.toSizeT(i);
    TORCH_CHECK(
        llvm::isPowerOf2_64(value),
        "For roundups, the divisions has to be power of 2 ");
    std::fill(
        roundup_power2_divisions_.begin(),
        roundup_power2_divisions_.end(),
        value);
  }
  return i;
}

size_t AcceleratorAllocatorConfig::parseExpandableSegments(
    const ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  use_expandable_segments_ = tokenizer.toBool(++i);

  return i;
}

size_t AcceleratorAllocatorConfig::parsePinnedUseBackgroundThreads(
    const ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  pinned_use_background_threads_ = tokenizer.toBool(++i);

  return i;
}

void AcceleratorAllocatorConfig::parseArgs(const std::string& env) {
  // The following option will be reset to its default value if not explicitly
  // set each time.
  max_split_size_ = std::numeric_limits<size_t>::max();
  roundup_power2_divisions_.assign(kRoundUpPowerOfTwoIntervals, 0);
  garbage_collection_threshold_ = 0;

  {
    std::lock_guard<std::mutex> lock(last_allocator_settings_mutex_);
    last_allocator_settings_ = env;
  }

  ConfigTokenizer tokenizer(env);
  for (size_t i = 0; i < tokenizer.size(); i++) {
    const auto& key = tokenizer[i];
    if (key == "max_split_size_mb") {
      i = parseMaxSplitSize(tokenizer, i);
    } else if (key == "max_non_split_rounding_mb") {
      i = parseMaxNonSplitRoundingSize(tokenizer, i);
    } else if (key == "garbage_collection_threshold") {
      i = parseGarbageCollectionThreshold(tokenizer, i);
    } else if (key == "roundup_power2_divisions") {
      i = parseRoundUpPower2Divisions(tokenizer, i);
    } else if (key == "expandable_segments") {
      i = parseExpandableSegments(tokenizer, i);
    } else if (key == "pinned_use_background_threads") {
      i = parsePinnedUseBackgroundThreads(tokenizer, i);
    } else {
      i = tokenizer.skipKey(i);
    }

    if (i + 1 < tokenizer.size()) {
      tokenizer.checkToken(++i, ",");
    }
  }
}

} // namespace c10::CachingAllocator
