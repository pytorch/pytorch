#include <gtest/gtest.h>

#include <torch/headeronly/cpu/Utils.h>

#include <cstdint>
#include <string>
#include <unordered_map>

TEST(TestCPUCacheSizes, HeaderOnly) {
  std::unordered_map<std::string, std::int64_t> cache_sizes;
  torch::headeronly::cpu::populate_cpu_cache_sizes_from_sysconf(cache_sizes);

#if defined(__s390x__)
#if defined(_SC_LEVEL1_DCACHE_SIZE)
  const auto l1d_cache_size = cache_sizes.find("l1d_cache_size");
  ASSERT_NE(l1d_cache_size, cache_sizes.end());
  EXPECT_GE(l1d_cache_size->second, 0);
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
  const auto l2_cache_size = cache_sizes.find("l2_cache_size");
  ASSERT_NE(l2_cache_size, cache_sizes.end());
  EXPECT_GE(l2_cache_size->second, 0);
#endif
#else
  EXPECT_TRUE(cache_sizes.empty());
#endif
}
