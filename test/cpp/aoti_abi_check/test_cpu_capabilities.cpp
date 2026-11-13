#include <gtest/gtest.h>

#include <torch/headeronly/cpu/Utils.h>

#include <string>
#include <variant>

TEST(TestCPUCapabilities, HeaderOnly) {
  const auto capabilities =
      torch::headeronly::cpu::headeronly_get_cpu_capabilities();

  const auto architecture = capabilities.find("architecture");
  ASSERT_NE(architecture, capabilities.end());
  ASSERT_TRUE(std::holds_alternative<std::string>(architecture->second));
  EXPECT_FALSE(std::get<std::string>(architecture->second).empty());

#if !defined(__s390x__) && !defined(__powerpc__)
  if (cpuinfo_initialize()) {
    const auto logical_cores = capabilities.find("num_logical_cores");
    ASSERT_NE(logical_cores, capabilities.end());
    ASSERT_TRUE(std::holds_alternative<int64_t>(logical_cores->second));
    EXPECT_GT(std::get<int64_t>(logical_cores->second), 0);
#if defined(__x86_64__) || defined(_M_X64)
    const auto avx = capabilities.find("avx");
    ASSERT_NE(avx, capabilities.end());
    EXPECT_TRUE(std::holds_alternative<bool>(avx->second));
#endif
  }
#endif
}
