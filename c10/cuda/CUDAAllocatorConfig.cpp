#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/llvmMathExtras.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

namespace c10::cuda::CUDACachingAllocator {

constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

CUDAAllocatorConfig::CUDAAllocatorConfig()
    : m_max_split_size(std::numeric_limits<size_t>::max()),
      m_max_non_split_rounding_size(kLargeBuffer),
      m_garbage_collection_threshold(0),
      m_pinned_num_register_threads(1),
      m_pinned_reserve_segment_size_mb(0),
      m_expandable_segments(false),
#if CUDA_VERSION >= 12030
      m_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::UNSPECIFIED),
#else
      m_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::POSIX_FD),
#endif
      m_release_lock_on_cudamalloc(false),
      m_pinned_use_cuda_host_register(false),
      m_graph_capture_record_stream_reuse(false),
      m_pinned_use_background_threads(false) {
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
}

size_t CUDAAllocatorConfig::roundup_power2_divisions(size_t size) {
  size_t log_size = (63 - llvm::countLeadingZeros(size));

  // Our intervals start at 1MB and end at 64GB
  const size_t interval_start =
      63 - llvm::countLeadingZeros(static_cast<size_t>(1048576));
  const size_t interval_end =
      63 - llvm::countLeadingZeros(static_cast<size_t>(68719476736));
  TORCH_CHECK(
      (interval_end - interval_start == kRoundUpPowerOfTwoIntervals),
      "kRoundUpPowerOfTwoIntervals mismatch");

  int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

  index = std::max(0, index);
  index = std::min(index, static_cast<int>(kRoundUpPowerOfTwoIntervals) - 1);
  return instance().m_roundup_power2_divisions[index];
}

void CUDAAllocatorConfig::lexArgs(
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

void CUDAAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i] == std::string(1, c),
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

size_t CUDAAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  constexpr int mb = 1024 * 1024;
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / mb,
        "CachingAllocator option max_split_size_mb too small, must be > ",
        kLargeBuffer / mb,
        "");
    val1 = std::max(val1, kLargeBuffer / mb);
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / mb));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t CUDAAllocatorConfig::parseMaxNonSplitRoundingSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  constexpr int mb = 1024 * 1024;
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / mb,
        "CachingAllocator option max_non_split_rounding_mb too small, must be > ",
        kLargeBuffer / mb,
        "");
    val1 = std::max(val1, kLargeBuffer / mb);
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / mb));
    m_max_non_split_rounding_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_non_split_rounding_mb value", "");
  }
  return i;
}

size_t CUDAAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(
        val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
    TORCH_CHECK(
        val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(
        false, "Error, expecting garbage_collection_threshold value", "");
  }
  return i;
}

size_t CUDAAllocatorConfig::parseRoundUpPower2Divisions(
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
          TORCH_CHECK(
              false, "Error parsing roundup_power2_divisions value", "");
        }
        TORCH_CHECK(
            val2 == 0 || llvm::isPowerOf2_64(val2),
            "For roundups, the divisions has to be power of 2 or 0 to disable roundup ",
            "");

        if (std::string_view(val1) == ">") {
          std::fill(
              std::next(
                  m_roundup_power2_divisions.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              m_roundup_power2_divisions.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ",
              "");

          size_t index = 63 - llvm::countLeadingZeros(val1_long);
          index = std::max((size_t)0, index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          if (first_value) {
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          if (index < m_roundup_power2_divisions.size()) {
            m_roundup_power2_divisions[index] = val2;
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
          "For roundups, the divisions has to be power of 2 ",
          "");
      std::fill(
          m_roundup_power2_divisions.begin(),
          m_roundup_power2_divisions.end(),
          val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  return i;
}

size_t CUDAAllocatorConfig::parseAllocatorConfig(
    const std::vector<std::string>& config,
    size_t i,
    bool& used_cudaMallocAsync) {
  // For ease of maintenance and understanding, the CUDA and ROCm
  // implementations of this function are separated. This avoids having many
  // #ifdef's throughout.
#ifdef USE_ROCM
  // Ease burden on ROCm users by allowing either cuda or hip tokens.
  // cuda token is broken up to prevent hipify matching it.
#define PYTORCH_TOKEN1 \
  "cud"                \
  "aMallocAsync"
#define PYTORCH_TOKEN2 "hipMallocAsync"
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        ((config[i] == "native") || (config[i] == PYTORCH_TOKEN1) ||
         (config[i] == PYTORCH_TOKEN2)),
        "Unknown allocator backend, "
        "options are native, " PYTORCH_TOKEN1 ", and " PYTORCH_TOKEN2);
    used_cudaMallocAsync =
        (config[i] == PYTORCH_TOKEN1 || config[i] == PYTORCH_TOKEN2);
    TORCH_INTERNAL_ASSERT(
        config[i] == get()->name() ||
            (config[i] == PYTORCH_TOKEN1 && get()->name() == PYTORCH_TOKEN2),
        "Allocator backend parsed at runtime != "
        "allocator backend parsed at load time, ",
        config[i],
        " != ",
        get()->name());
  } else {
    TORCH_CHECK(false, "Error parsing backend value", "");
  }
  return i;
#undef PYTORCH_TOKEN1
#undef PYTORCH_TOKEN2
#else // USE_ROCM
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        ((config[i] == "native") || (config[i] == "cudaMallocAsync")),
        "Unknown allocator backend, "
        "options are native and cudaMallocAsync");
    used_cudaMallocAsync = (config[i] == "cudaMallocAsync");
    if (used_cudaMallocAsync) {
#if CUDA_VERSION >= 11040
      int version = 0;
      C10_CUDA_CHECK(cudaDriverGetVersion(&version));
      TORCH_CHECK(
          version >= 11040,
          "backend:cudaMallocAsync requires CUDA runtime "
          "11.4 or newer, but cudaDriverGetVersion returned ",
          version);
#else
      TORCH_CHECK(
          false,
          "backend:cudaMallocAsync requires PyTorch to be built with "
          "CUDA 11.4 or newer, but CUDA_VERSION is ",
          CUDA_VERSION);
#endif
    }
    TORCH_INTERNAL_ASSERT(
        config[i] == get()->name(),
        "Allocator backend parsed at runtime != "
        "allocator backend parsed at load time");
  } else {
    TORCH_CHECK(false, "Error parsing backend value", "");
  }
  return i;
#endif // USE_ROCM
}

void CUDAAllocatorConfig::parseArgs(const std::optional<std::string>& env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;
  bool used_cudaMallocAsync = false;
  bool used_native_specific_option = false;

  if (!env.has_value()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(m_last_allocator_settings_mutex);
    m_last_allocator_settings = env.value();
  }

  std::vector<std::string> config;
  lexArgs(env.value(), config);

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
      i = parseAllocatorConfig(config, i, used_cudaMallocAsync);
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
      m_expandable_segments = (config_item_view == "True");
    } else if (
        // ROCm build's hipify step will change "cuda" to "hip", but for ease of
        // use, accept both. We must break up the string to prevent hipify here.
        config_item_view == "release_lock_on_hipmalloc" ||
        config_item_view ==
            "release_lock_on_c"
            "udamalloc") {
      used_native_specific_option = true;
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for release_lock_on_cudamalloc");
      config_item_view = config[i];
      m_release_lock_on_cudamalloc = (config_item_view == "True");
    } else if (
        // ROCm build's hipify step will change "cuda" to "hip", but for ease of
        // use, accept both. We must break up the string to prevent hipify here.
        config_item_view == "pinned_use_hip_host_register" ||
        config_item_view ==
            "pinned_use_c"
            "uda_host_register") {
      i = parsePinnedUseCudaHostRegister(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "pinned_num_register_threads") {
      i = parsePinnedNumRegisterThreads(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "pinned_reserve_segment_size_mb") {
      i = parsePinnedReserveSegmentSize(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "pinned_use_background_threads") {
      i = parsePinnedUseBackgroundThreads(config, i);
      used_native_specific_option = true;
    } else if (config_item_view == "graph_capture_record_stream_reuse") {
      i = parseGraphCaptureRecordStreamReuse(config, i);
      used_native_specific_option = true;
    } else {
      TORCH_CHECK(
          false, "Unrecognized CachingAllocator option: ", config_item_view);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (used_cudaMallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:cudaMallocAsync ignores max_split_size_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

size_t CUDAAllocatorConfig::parsePinnedUseCudaHostRegister(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for pinned_use_cuda_host_register");
    m_pinned_use_cuda_host_register = (config[i] == "True");
  } else {
    TORCH_CHECK(
        false, "Error, expecting pinned_use_cuda_host_register value", "");
  }
  return i;
}

size_t CUDAAllocatorConfig::parseGraphCaptureRecordStreamReuse(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for graph_capture_record_stream_reuse");
    m_graph_capture_record_stream_reuse = (config[i] == "True");
  } else {
    TORCH_CHECK(
        false, "Error, expecting graph_capture_record_stream_reuse value", "");
  }

  return i;
}

size_t CUDAAllocatorConfig::parsePinnedNumRegisterThreads(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val2 = stoi(config[i]);
    TORCH_CHECK(
        llvm::isPowerOf2_64(val2),
        "Number of register threads has to be power of 2 ",
        "");
    auto maxThreads = CUDAAllocatorConfig::pinned_max_register_threads();
    TORCH_CHECK(
        val2 <= maxThreads,
        "Number of register threads should be less than or equal to " +
            std::to_string(maxThreads),
        "");
    m_pinned_num_register_threads = val2;
  } else {
    TORCH_CHECK(
        false, "Error, expecting pinned_num_register_threads value", "");
  }
  return i;
}

size_t CUDAAllocatorConfig::parsePinnedReserveSegmentSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val2 = stoi(config[i]);
    TORCH_CHECK(
        val2 > 0, "Pinned reserve segment size has to be greater than 0 ", "");
    m_pinned_reserve_segment_size_mb = val2;
  } else {
    TORCH_CHECK(
        false, "Error, expecting pinned_reserve_segment_size_mb value", "");
  }
  return i;
}

size_t CUDAAllocatorConfig::parsePinnedUseBackgroundThreads(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        (config[i] == "True" || config[i] == "False"),
        "Expected a single True/False argument for pinned_use_background_threads");
    m_pinned_use_background_threads = (config[i] == "True");
  } else {
    TORCH_CHECK(
        false, "Error, expecting pinned_use_background_threads value", "");
  }
  return i;
}

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  CUDACachingAllocator::CUDAAllocatorConfig::instance().parseArgs(env.c_str());
}

} // namespace c10::cuda::CUDACachingAllocator
