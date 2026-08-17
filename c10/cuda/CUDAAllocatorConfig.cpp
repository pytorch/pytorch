#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cmath>
#include <locale>
#include <sstream>

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
    } else if (key == "pinned_free_catch_all") {
      i = parsePinnedFreeCatchAll(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "throw_on_cudamalloc_oom") {
      i = parseThrowOnCudaMallocOom(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "expandable_segments_reserve") {
      i = parseExpandableSegmentsReserve(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "expandable_segments_reserve_by_class") {
      i = parseExpandableSegmentsReserveByClass(tokenizer, i);
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

size_t CUDAAllocatorConfig::parsePerProcessMemoryFraction(
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

size_t CUDAAllocatorConfig::parsePinnedFreeCatchAll(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  m_pinned_free_catch_all = tokenizer.toBool(++i);
  return i;
}

size_t CUDAAllocatorConfig::parseThrowOnCudaMallocOom(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  // Format: throw_on_cudamalloc_oom:true or throw_on_cudamalloc_oom:false
  // When enabled, throws OOM error before calling cudaMalloc if the allocation
  // would likely fail due to insufficient memory.
  tokenizer.checkToken(++i, ":");
  m_throw_on_cudamalloc_oom = tokenizer.toBool(++i);
  return i;
}

std::optional<ExpandableSegmentReserveSpec> CUDAAllocatorConfig::
    parseReserveSpec(const std::string& token) {
  // A trailing 'G' means the value is absolute GiB; otherwise it is a fraction
  // of total GPU memory. Malformed or non-positive values are intentionally
  // non-fatal: we log and return nullopt so the caller keeps its default rather
  // than aborting the process over a bad PYTORCH_CUDA_ALLOC_CONF entry.
  bool is_gib = !token.empty() && (token.back() == 'G' || token.back() == 'g');
  const std::string num = is_gib ? token.substr(0, token.size() - 1) : token;
  // Parse in the classic (C) locale so a non-C process locale cannot change how
  // the decimal separator is interpreted, and reject non-finite values (inf/nan
  // would otherwise slip past the range check and poison resolveBytes).
  double value = 0.0;
  std::istringstream iss(num);
  iss.imbue(std::locale::classic());
  iss >> value;
  const bool ok = !num.empty() && !iss.fail() && iss.eof() &&
      std::isfinite(value) && value > 0.0;
  if (!ok) {
    TORCH_WARN(
        "Ignoring invalid expandable-segment reserve value '",
        token,
        "' in PYTORCH_CUDA_ALLOC_CONF (expected a positive, finite fraction "
        "like 0.5 or an absolute size like 40G); using default.");
    return std::nullopt;
  }
  return ExpandableSegmentReserveSpec{/*is_fraction=*/!is_gib, value};
}

size_t CUDAAllocatorConfig::parseExpandableSegmentsReserve(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  // Bad value -> keep the default (leave m_expandable_segments_reserve_set
  // false so untagged streams stay on the historical reserve).
  auto spec = parseReserveSpec(tokenizer[++i]);
  if (spec) {
    std::lock_guard<std::mutex> lock(m_reserve_mutex);
    m_expandable_segments_reserve = *spec;
    m_expandable_segments_reserve_set = true;
  }
  return i;
}

size_t CUDAAllocatorConfig::parseExpandableSegmentsReserveByClass(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  // Format: expandable_segments_reserve_by_class:[name:val,name:val,...]
  tokenizer.checkToken(++i, ":");
  tokenizer.checkToken(++i, "[");
  // Build into a local map, then publish under the lock so readers on the
  // segment-creation path never observe a partially-updated map.
  ska::flat_hash_map<std::string, ExpandableSegmentReserveSpec> parsed;
  while (!tokenizer.checkToken(i + 1, "]")) {
    const std::string& name = tokenizer[++i];
    tokenizer.checkToken(++i, ":");
    // A malformed entry is skipped (logged); other entries still apply.
    if (auto spec = parseReserveSpec(tokenizer[++i])) {
      parsed[name] = *spec;
    }
    if (tokenizer.checkToken(i + 1, ",")) {
      ++i;
    }
  }
  ++i; // consume ']'
  {
    std::lock_guard<std::mutex> lock(m_reserve_mutex);
    m_expandable_segments_reserve_by_class = std::move(parsed);
  }
  return i;
}

REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(CUDAAllocatorConfig)

} // namespace c10::cuda::CUDACachingAllocator
