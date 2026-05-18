#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/util/Exception.h>

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <string>

TEST(AllocCpuTest, MimallocStatsApi) {
  if (c10::is_mimalloc_enabled()) {
    c10::reset_mimalloc_stats();

    auto* allocator = c10::GetCPUAllocator();
    auto data = allocator->allocate(1024);

    const auto stats = c10::get_mimalloc_stats_json();

    EXPECT_NE(stats.find("\"mimalloc_version\""), std::string::npos);
    EXPECT_NE(stats.find("\"process\""), std::string::npos);

    const auto show_stats = c10::get_mimalloc_option("show_stats");
    c10::set_mimalloc_option("show_stats", 1);
    EXPECT_EQ(c10::get_mimalloc_option("show_stats"), 1);
    c10::set_mimalloc_option("show_stats", show_stats);
    EXPECT_EQ(c10::get_mimalloc_option("show_stats"), show_stats);

    EXPECT_THROW(c10::get_mimalloc_option("not_an_option"), c10::Error);
    EXPECT_THROW(c10::set_mimalloc_option("not_an_option", 1), c10::Error);
#if INT64_MAX > LONG_MAX
    EXPECT_THROW(
        c10::set_mimalloc_option(
            "show_stats",
            static_cast<int64_t>(std::numeric_limits<long>::max()) + 1),
        c10::Error);
#endif
  } else {
    EXPECT_THROW(c10::get_mimalloc_stats_json(), c10::Error);
    EXPECT_THROW(c10::reset_mimalloc_stats(), c10::Error);
    EXPECT_THROW(c10::get_mimalloc_option("show_stats"), c10::Error);
    EXPECT_THROW(c10::set_mimalloc_option("show_stats", 1), c10::Error);
  }
}
