#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

namespace c10::cuda::CUDACachingAllocator {

size_t CUDAAllocatorConfig::parseAllocatorConfig(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i,
    bool& used_cudaMallocAsync) {
  // Ease burden on ROCm users by allowing either cuda or hip tokens.
  // cuda token is broken up to prevent hipify matching it.
#define PYTORCH_TOKEN1 \
  "cud"                \
  "aMallocAsync"
#define PYTORCH_TOKEN2 "hipMallocAsync"
  tokenizer.checkToken(++i, ":");
  i++; // Move to the value after the colon
#ifdef USE_ROCM
  TORCH_CHECK_VALUE(
      ((tokenizer[i] == "native") || (tokenizer[i] == PYTORCH_TOKEN1) ||
       (tokenizer[i] == PYTORCH_TOKEN2)),
      "Unknown allocator backend, "
      "options are native, " PYTORCH_TOKEN1 ", and " PYTORCH_TOKEN2);
  used_cudaMallocAsync =
      (tokenizer[i] == PYTORCH_TOKEN1 || tokenizer[i] == PYTORCH_TOKEN2);
  TORCH_INTERNAL_ASSERT(
      tokenizer[i] == get()->name() ||
          (tokenizer[i] == PYTORCH_TOKEN1 && get()->name() == PYTORCH_TOKEN2),
      "Allocator backend parsed at runtime != "
      "allocator backend parsed at load time, ",
      tokenizer[i],
      " != ",
      get()->name());
#else // USE_ROCM
  TORCH_CHECK_VALUE(
      ((tokenizer[i] == "native") || (tokenizer[i] == PYTORCH_TOKEN1)),
      "Unknown allocator backend, "
      "options are native and " PYTORCH_TOKEN1);
  used_cudaMallocAsync = (tokenizer[i] == PYTORCH_TOKEN1);
  TORCH_INTERNAL_ASSERT(
      tokenizer[i] == get()->name(),
      "Allocator backend parsed at runtime != "
      "allocator backend parsed at load time, ",
      tokenizer[i],
      " != ",
      get()->name());
  if (used_cudaMallocAsync) {
#if CUDA_VERSION >= 11040
    int version = 0;
    C10_CUDA_CHECK(cudaDriverGetVersion(&version));
    TORCH_CHECK(
        version >= 11040,
        "backend:cudaMallocAsync requires CUDA runtime "
        "11.4 or newer, but cudaDriverGetVersion returned ",
        version);
#else // CUDA_VERSION >= 11040
    TORCH_CHECK(
        false,
        "backend:cudaMallocAsync requires PyTorch to be built with "
        "CUDA 11.4 or newer, but CUDA_VERSION is ",
        CUDA_VERSION);
#endif // CUDA_VERSION >= 11040
  }
#endif // USE_ROCM
  return i;
}

void CUDAAllocatorConfig::parseArgs(const std::string& env) {
  bool used_cudaMallocAsync = false;
  bool used_native_specific_option = false;

  c10::CachingAllocator::ConfigTokenizer tokenizer(env);
  for (size_t i = 0; i < tokenizer.size(); i++) {
    const auto& key = tokenizer[i];
    if (key == "backend") {
      i = parseAllocatorConfig(tokenizer, i, used_cudaMallocAsync);
    } else if (
        // ROCm build's hipify step will change "cuda" to "hip", but for ease of
        // use, accept both. We must break up the string to prevent hipify here.
        key == "release_lock_on_hipmalloc" ||
        key ==
            "release_lock_on_c"
            "udamalloc") {
      used_native_specific_option = true;
      tokenizer.checkToken(++i, ":");
      m_release_lock_on_cudamalloc = tokenizer.toBool(++i);
    } else if (
        // ROCm build's hipify step will change "cuda" to "hip", but for ease of
        // use, accept both. We must break up the string to prevent hipify here.
        key == "pinned_use_hip_host_register" ||
        key ==
            "pinned_use_c"
            "uda_host_register") {
      i = parsePinnedUseCudaHostRegister(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "pinned_num_register_threads") {
      i = parsePinnedNumRegisterThreads(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "pinned_reserve_segment_size_mb") {
      i = parsePinnedReserveSegmentSize(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "graph_capture_record_stream_reuse") {
      i = parseGraphCaptureRecordStreamReuse(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "per_process_memory_fraction") {
      i = parsePerProcessMemoryFraction(tokenizer, i);
      used_native_specific_option = true;
    } else {
      const auto& keys =
          c10::CachingAllocator::AcceleratorAllocatorConfig::getKeys();
      TORCH_CHECK_VALUE(
          keys.find(key) != keys.end(),
          "Unrecognized key '",
          key,
          "' in CUDA allocator config.");
      // Skip the key and its value
      i = tokenizer.skipKey(i);
    }

    if (i + 1 < tokenizer.size()) {
      tokenizer.checkToken(++i, ",");
    }
  }

  if (used_cudaMallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:cudaMallocAsync ignores max_split_size_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

size_t CUDAAllocatorConfig::parsePinnedUseCudaHostRegister(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  m_pinned_use_cuda_host_register = tokenizer.toBool(++i);
  return i;
}

size_t CUDAAllocatorConfig::parseGraphCaptureRecordStreamReuse(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  m_graph_capture_record_stream_reuse = tokenizer.toBool(++i);
  return i;
}

double CUDAAllocatorConfig::parsePerProcessMemoryFraction(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  double val_env = tokenizer.toDouble(++i);
  TORCH_CHECK_VALUE(
      val_env >= 0.0 && val_env <= 1.0,
      "per_process_memory_fraction is invalid, set it in [0.0, 1.0]");
  m_per_process_memory_fraction = val_env;
  return i;
}

size_t CUDAAllocatorConfig::parsePinnedNumRegisterThreads(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  size_t val2 = tokenizer.toSizeT(++i);
  TORCH_CHECK_VALUE(
      llvm::isPowerOf2_64(val2),
      "Number of register threads has to be power of 2, got ",
      val2);
  auto maxThreads = CUDAAllocatorConfig::pinned_max_register_threads();
  TORCH_CHECK_VALUE(
      val2 <= maxThreads,
      "Number of register threads should be less than or equal to ",
      maxThreads,
      ", got ",
      val2);
  m_pinned_num_register_threads = val2;
  return i;
}

size_t CUDAAllocatorConfig::parsePinnedReserveSegmentSize(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  size_t val2 = tokenizer.toSizeT(++i);
  TORCH_CHECK_VALUE(
      val2 > 0, "Pinned reserve segment size has to be greater than 0");
  m_pinned_reserve_segment_size_mb = val2;
  return i;
}

REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(CUDAAllocatorConfig)

} // namespace c10::cuda::CUDACachingAllocator
