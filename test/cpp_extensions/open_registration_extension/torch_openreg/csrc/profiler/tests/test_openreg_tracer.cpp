#include <gtest/gtest.h>

#include "profiler/OpenRegTracer.h"

using namespace torch_openreg::profiler;

class OpenRegTracerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& t = OpenRegTracer::instance();
    t.disable();
    t.flush();
  }

  void TearDown() override {
    auto& t = OpenRegTracer::instance();
    t.disable();
    t.flush();
  }
};

TEST_F(OpenRegTracerTest, Singleton) {
  auto& a = OpenRegTracer::instance();
  auto& b = OpenRegTracer::instance();
  EXPECT_EQ(&a, &b);
}

TEST_F(OpenRegTracerTest, EnableDisable) {
  auto& t = OpenRegTracer::instance();
  EXPECT_FALSE(t.isEnabled());
  t.enable();
  EXPECT_TRUE(t.isEnabled());
  t.disable();
  EXPECT_FALSE(t.isEnabled());
}

TEST_F(OpenRegTracerTest, CorrelationPushPop) {
  auto& t = OpenRegTracer::instance();
  EXPECT_EQ(t.currentCorrelation(), 0);

  t.pushCorrelation(42);
  EXPECT_EQ(t.currentCorrelation(), 42);

  t.pushCorrelation(99);
  EXPECT_EQ(t.currentCorrelation(), 99);

  t.popCorrelation();
  EXPECT_EQ(t.currentCorrelation(), 42);

  t.popCorrelation();
  EXPECT_EQ(t.currentCorrelation(), 0);
}

TEST_F(OpenRegTracerTest, RecordAndFlush) {
  auto& t = OpenRegTracer::instance();
  t.record({ActivityKind::KERNEL, "matmul", 100, 200, 0, 1, 10});
  t.record({ActivityKind::MEMCPY, "h2d", 300, 400, 0, 0, 11});

  auto records = t.flush();
  ASSERT_EQ(records.size(), 2u);

  EXPECT_EQ(records[0].kind, ActivityKind::KERNEL);
  EXPECT_EQ(records[0].name, "matmul");
  EXPECT_EQ(records[0].start, 100);
  EXPECT_EQ(records[0].end, 200);
  EXPECT_EQ(records[0].device_id, 0);
  EXPECT_EQ(records[0].stream_id, 1);
  EXPECT_EQ(records[0].correlation_id, 10u);

  EXPECT_EQ(records[1].kind, ActivityKind::MEMCPY);
  EXPECT_EQ(records[1].name, "h2d");
  EXPECT_EQ(records[1].correlation_id, 11u);

  EXPECT_TRUE(t.flush().empty());
}
