#ifdef USE_KINETO

#include <gtest/gtest.h>

#include "profiler/OpenRegActivityProfiler.h"

namespace openreg::profiler {

TEST(OpenRegActivityProfilerTest, Name) {
  OpenRegActivityProfiler profiler;
  EXPECT_EQ(profiler.name(), "openreg");
}

TEST(OpenRegActivityProfilerTest, AvailableActivities) {
  OpenRegActivityProfiler profiler;
  const auto& acts = profiler.availableActivities();
  EXPECT_EQ(acts.size(), 1u);
  EXPECT_TRUE(acts.count(libkineto::ActivityType::CONCURRENT_KERNEL));
}

TEST(OpenRegActivityProfilerTest, ConfigureReturnsNonNullSession) {
  OpenRegActivityProfiler profiler;
  libkineto::Config config;
  auto session = profiler.configure({}, config);
  EXPECT_NE(session, nullptr);
}

TEST(OpenRegActivityProfilerTest, AsyncConfigureReturnsNonNullSession) {
  OpenRegActivityProfiler profiler;
  libkineto::Config config;
  auto session = profiler.configure(
      /*ts_ms=*/0, /*duration_ms=*/1000, {}, config);
  EXPECT_NE(session, nullptr);
}

} // namespace openreg::profiler

#endif // USE_KINETO