#ifdef USE_KINETO

#include <gtest/gtest.h>

#include <csrc/tracer.h>
#include "profiler/OpenRegActivityProfiler.h"
#include "profiler/OpenRegActivityProfilerSession.h"

namespace openreg::profiler {

TEST(OpenRegActivityProfilerTest, Name) {
  OpenRegActivityProfiler profiler;
  EXPECT_EQ(profiler.name(), "openreg");
}

TEST(OpenRegActivityProfilerTest, AvailableActivities) {
  OpenRegActivityProfiler profiler;
  const auto& acts = profiler.availableActivities();
  EXPECT_EQ(acts.size(), 5u);
  EXPECT_TRUE(acts.count(libkineto::ActivityType::CONCURRENT_KERNEL));
  EXPECT_TRUE(acts.count(libkineto::ActivityType::GPU_MEMCPY));
  EXPECT_TRUE(acts.count(libkineto::ActivityType::GPU_MEMSET));
  EXPECT_TRUE(acts.count(libkineto::ActivityType::PRIVATEUSE1_RUNTIME));
  EXPECT_TRUE(acts.count(libkineto::ActivityType::PRIVATEUSE1_DRIVER));
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

class OpenRegSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    OpenRegTracer::instance().disableActivityTracing();
  }
  void TearDown() override {
    OpenRegTracer::instance().disableActivityTracing();
  }
};

TEST_F(OpenRegSessionTest, StartStopTogglesTracerEnabled) {
  OpenRegActivityProfilerSession session;
  EXPECT_FALSE(OpenRegTracer::instance().isActivityTracingEnabled());
  session.start();
  EXPECT_TRUE(OpenRegTracer::instance().isActivityTracingEnabled());
  session.stop();
  EXPECT_FALSE(OpenRegTracer::instance().isActivityTracingEnabled());
}

} // namespace openreg::profiler

#endif // USE_KINETO