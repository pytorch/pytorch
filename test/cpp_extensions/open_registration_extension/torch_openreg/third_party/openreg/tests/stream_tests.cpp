#include <gtest/gtest.h>
#include <include/openreg.h>

#include <atomic>
#include <thread>

namespace {

class StreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orSetDevice(0);
  }
};

TEST_F(StreamTest, StreamCreateAndDestroy) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);
  EXPECT_NE(stream, nullptr);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, StreamCreateWithInvalidPriority) {
  orStream_t stream = nullptr;
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, min_p - 1), orErrorUnknown);
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, max_p + 1), orErrorUnknown);
}

TEST_F(StreamTest, StreamTaskExecution) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  std::atomic<int> counter{0};
  EXPECT_EQ(openreg::addTaskToStream(stream, [&] { counter++; }), orSuccess);

  EXPECT_EQ(orStreamSynchronize(stream), orSuccess);
  EXPECT_EQ(counter.load(), 1);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, StreamQuery) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  EXPECT_EQ(orStreamQuery(stream), orSuccess);

  std::atomic<int> counter{0};
  openreg::addTaskToStream(stream, [&] { counter++; });

  EXPECT_EQ(orStreamSynchronize(stream), orSuccess);
  EXPECT_EQ(orStreamQuery(stream), orSuccess);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, DeviceSynchronize) {
  orStream_t stream1 = nullptr;
  orStream_t stream2 = nullptr;

  EXPECT_EQ(orStreamCreate(&stream1), orSuccess);
  EXPECT_EQ(orStreamCreate(&stream2), orSuccess);

  std::atomic<int> counter{0};
  openreg::addTaskToStream(stream1, [&] { counter++; });
  openreg::addTaskToStream(stream2, [&] { counter++; });

  EXPECT_EQ(orDeviceSynchronize(), orSuccess);
  EXPECT_EQ(counter.load(), 2);

  EXPECT_EQ(orStreamDestroy(stream1), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream2), orSuccess);
}

} // namespace
