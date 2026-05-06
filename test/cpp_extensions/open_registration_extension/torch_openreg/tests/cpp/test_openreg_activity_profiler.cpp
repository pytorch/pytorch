#ifdef USE_KINETO

#include <gtest/gtest.h>

#include <output_base.h>
#include "profiler/OpenRegActivityProfiler.h"
#include "profiler/OpenRegActivityProfilerSession.h"
#include "profiler/OpenRegTracer.h"

// ActivityBuffers is only forward-declared in public Kineto headers; its full
// definition lives in the private libkineto/src/ directory which is not on
// the include path. This stub makes unique_ptr<ActivityBuffers> in
// MockLogger::finalizeTrace destructible. finalizeTrace is never called in
// these tests.
namespace libkineto {
struct ActivityBuffers {};
} // namespace libkineto

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

struct MockLogger : public libkineto::ActivityLogger {
  void handleDeviceInfo(
      [[maybe_unused]] const libkineto::DeviceInfo& info,
      [[maybe_unused]] uint64_t time) override {}
  void handleResourceInfo(
      [[maybe_unused]] const libkineto::ResourceInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleOverheadInfo(
      [[maybe_unused]] const libkineto::ActivityLogger::OverheadInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleTraceSpan(
      [[maybe_unused]] const libkineto::TraceSpan& span) override {}
  void handleActivity(const libkineto::ITraceActivity& activity) override {
    activities.push_back(&activity);
  }
  void handleGenericActivity(
      [[maybe_unused]] const libkineto::GenericTraceActivity& activity) override {}
  void handleTraceStart(
      [[maybe_unused]] const std::unordered_map<std::string, std::string>& metadata,
      [[maybe_unused]] const std::string& device_properties) override {}
  void finalizeMemoryTrace(
      [[maybe_unused]] const std::string& traceDir,
      [[maybe_unused]] const libkineto::Config& config) override {}
  void finalizeTrace(
      [[maybe_unused]] const libkineto::Config& config,
      [[maybe_unused]] std::unique_ptr<libkineto::ActivityBuffers> buffers,
      [[maybe_unused]] int64_t endTime,
      [[maybe_unused]] std::unordered_map<
          std::string,
          std::vector<std::string>>& metadata) override {}
  std::vector<const libkineto::ITraceActivity*> activities;
};

class OpenRegSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    OpenRegTracer::instance().disable();
    OpenRegTracer::instance().flush();
  }
  void TearDown() override {
    OpenRegTracer::instance().disable();
    OpenRegTracer::instance().flush();
  }
};

TEST_F(OpenRegSessionTest, StartStopTogglesTracerEnabled) {
  OpenRegActivityProfilerSession session;
  EXPECT_FALSE(OpenRegTracer::instance().isEnabled());
  session.start();
  EXPECT_TRUE(OpenRegTracer::instance().isEnabled());
  session.stop();
  EXPECT_FALSE(OpenRegTracer::instance().isEnabled());
}

TEST_F(OpenRegSessionTest, ProcessTraceConvertsTimestampsNsToUs) {
  OpenRegActivityProfilerSession session;
  session.start();
  // start=1000ns, end=3000ns → timestamp()=1µs, duration()=2µs
  OpenRegTracer::instance().record(
      {ActivityKind::KERNEL, "test_op", 1000, 3000, 0, 0, 1});
  session.stop();
  MockLogger logger;
  session.processTrace(logger);
  ASSERT_EQ(logger.activities.size(), 1u);
  EXPECT_EQ(logger.activities[0]->timestamp(), int64_t{1});
  EXPECT_EQ(logger.activities[0]->duration(), int64_t{2});
}

TEST_F(OpenRegSessionTest, ProcessTraceConvertsCorrelationFlow) {
  OpenRegActivityProfilerSession session;
  session.start();
  OpenRegTracer::instance().record(
      {ActivityKind::KERNEL, "test_op", 1000, 2000, 0, 0, /*correlation_id=*/42});
  session.stop();
  MockLogger logger;
  session.processTrace(logger);
  ASSERT_EQ(logger.activities.size(), 1u);
  EXPECT_EQ(logger.activities[0]->flowId(), int64_t{42});
  EXPECT_EQ(
      logger.activities[0]->flowType(),
      static_cast<int>(libkineto::kLinkAsyncCpuGpu));
  EXPECT_FALSE(logger.activities[0]->flowStart());
}

TEST_F(OpenRegSessionTest, GetTraceBufferNonEmptyAfterProcessTrace) {
  OpenRegActivityProfilerSession session;
  session.start();
  OpenRegTracer::instance().record(
      {ActivityKind::KERNEL, "test_op", 1000, 2000, 0, 0, 1});
  session.stop();
  MockLogger logger;
  session.processTrace(logger);
  auto buf = session.getTraceBuffer();
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(buf->activities.size(), 1u);
}

TEST_F(OpenRegSessionTest, GetDeviceInfoCorrect) {
  OpenRegActivityProfilerSession session;
  auto info = session.getDeviceInfo();
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->id, int64_t{0});
  EXPECT_EQ(info->sortIndex, int64_t{0});
  EXPECT_EQ(info->name, "OpenReg");
  EXPECT_EQ(info->label, "OpenReg 0");
}

TEST_F(OpenRegSessionTest, GetResourceInfosCorrect) {
  OpenRegActivityProfilerSession session;
  session.start();
  OpenRegTracer::instance().record(
      {ActivityKind::KERNEL, "test_op", 1000, 2000,
       /*device_id=*/1, /*stream_id=*/2, /*correlation_id=*/1});
  session.stop();
  MockLogger logger;
  session.processTrace(logger);
  auto resources = session.getResourceInfos();
  ASSERT_EQ(resources.size(), 1u);
  EXPECT_EQ(resources[0].deviceId, int64_t{1});
  EXPECT_EQ(resources[0].id, int64_t{2});
  EXPECT_EQ(resources[0].name, "stream 2");
}

} // namespace openreg::profiler

#endif // USE_KINETO