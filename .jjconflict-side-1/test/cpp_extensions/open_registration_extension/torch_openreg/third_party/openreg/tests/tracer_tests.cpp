#include <gtest/gtest.h>

#include <csrc/tracer.h>

using namespace openreg::profiler;

class OpenRegTracerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    OpenRegTracer::instance().disableActivityTracing();
  }

  void TearDown() override {
    OpenRegTracer::instance().disableActivityTracing();
  }
};

TEST_F(OpenRegTracerTest, Singleton) {
  auto& a = OpenRegTracer::instance();
  auto& b = OpenRegTracer::instance();
  EXPECT_EQ(&a, &b);
}

TEST_F(OpenRegTracerTest, EnableDisable) {
  auto& t = OpenRegTracer::instance();
  EXPECT_FALSE(t.isActivityTracingEnabled());
  t.enableActivityTracing();
  EXPECT_TRUE(t.isActivityTracingEnabled());
  t.disableActivityTracing();
  EXPECT_FALSE(t.isActivityTracingEnabled());
}

TEST_F(OpenRegTracerTest, CorrelationPushPop) {
  auto& t = OpenRegTracer::instance();
  EXPECT_EQ(t.getExternalCorrelationId(), 0);

  t.pushExternalCorrelationId(42);
  EXPECT_EQ(t.getExternalCorrelationId(), 42);

  t.pushExternalCorrelationId(99);
  EXPECT_EQ(t.getExternalCorrelationId(), 99);

  t.popExternalCorrelationId();
  EXPECT_EQ(t.getExternalCorrelationId(), 42);

  t.popExternalCorrelationId();
  EXPECT_EQ(t.getExternalCorrelationId(), 0);
}
