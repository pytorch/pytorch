#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/llvmMathExtras.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <vector>

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

// Environment config parser
class CUDAAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
#ifndef PYTORCH_C10_DRIVER_API_SUPPORTED
    if (instance().m_expandable_segments) {
      TORCH_WARN_ONCE("expandable_segments not supported on this platform")
    }
    return false;
#else
    return instance().m_expandable_segments;
#endif
  }

  static bool release_lock_on_cudamalloc() {
    return instance().m_release_lock_on_cudamalloc;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  static CUDAAllocatorConfig& instance() {
    static CUDAAllocatorConfig* s_instance = ([]() {
      auto inst = new CUDAAllocatorConfig();
      const char* env = getenv("PYTORCH_CUDA_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  CUDAAllocatorConfig();

  void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cudaMallocAsync);

  std::atomic<size_t> m_max_split_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<bool> m_expandable_segments;
  std::atomic<bool> m_release_lock_on_cudamalloc;
};

// General caching allocator utilities
C10_CUDA_API void setAllocatorSettings(const std::string& env);

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
