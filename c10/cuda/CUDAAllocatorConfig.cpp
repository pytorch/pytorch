#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/llvmMathExtras.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

namespace c10::cuda::CUDACachingAllocator {

CUDAAllocatorConfig::CUDAAllocatorConfig()
    : m_pinned_num_register_threads(1),
      m_pinned_reserve_segment_size_mb(0),
#if CUDA_VERSION >= 12030
      m_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::UNSPECIFIED),
#else
      m_expandable_segments_handle_type(
          Expandable_Segments_Handle_Type::POSIX_FD),
#endif
      m_release_lock_on_cudamalloc(false),
      m_pinned_use_cuda_host_register(false),
      m_graph_capture_record_stream_reuse(false) {
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

void CUDAAllocatorConfig::parseArgs(const std::string& env) {
  // If empty, set the default values
  bool used_cudaMallocAsync = false;
  bool used_native_specific_option = false;

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    if (config_item_view == "backend") {
      i = parseAllocatorConfig(config, i, used_cudaMallocAsync);
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
    } else if (config_item_view == "graph_capture_record_stream_reuse") {
      i = parseGraphCaptureRecordStreamReuse(config, i);
      used_native_specific_option = true;
    } else {
      const auto& keys =
          c10::CachingAllocator::AcceleratorAllocatorConfig::getKeys();
      TORCH_CHECK(
          keys.find(config[i]) != keys.end(),
          "Unrecognized key '",
          config_item_view,
          "' in CUDA allocator config.");
      // Skip the key and its value
      consumeToken(config, ++i, ':');
      i++;
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

REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(CUDAAllocatorConfig)

} // namespace c10::cuda::CUDACachingAllocator
