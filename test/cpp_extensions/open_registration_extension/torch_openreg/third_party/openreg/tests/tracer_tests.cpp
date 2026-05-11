#include <gtest/gtest.h>

#include <csrc/OpenRegTracer.h>

using namespace openreg::profiler;

class OpenRegTracerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    OpenRegTracer::instance().disable();
  }

  void TearDown() override {
    OpenRegTracer::instance().disable();
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
