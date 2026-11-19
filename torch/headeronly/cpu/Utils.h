#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <cstdint>

#if defined(__s390x__)
#include <unistd.h>
#endif

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, cpu)

// Adds cache sizes available through sysconf without cpuinfo. The map's mapped
// type must accept int64_t values.
template <typename CapabilitiesMap>
inline void populate_cpu_cache_sizes_from_sysconf(
    CapabilitiesMap& capabilities) {
  (void)capabilities;
#if defined(__s390x__)
#if defined(_SC_LEVEL1_DCACHE_SIZE)
  {
    auto value = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    capabilities["l1d_cache_size"] =
        static_cast<std::int64_t>((value > 0) ? value : 0);
  }
#endif // defined(_SC_LEVEL1_DCACHE_SIZE)

#if defined(_SC_LEVEL2_CACHE_SIZE)
  {
    auto value = sysconf(_SC_LEVEL2_CACHE_SIZE);
    capabilities["l2_cache_size"] =
        static_cast<std::int64_t>((value > 0) ? value : 0);
  }
#endif // defined(_SC_LEVEL2_CACHE_SIZE)
#endif // defined(__s390x__)
}

HIDDEN_NAMESPACE_END(torch, headeronly, cpu)
