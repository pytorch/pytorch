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

TEST_F(StreamTest, StreamCreateNullptr) {
  // Creation API should reject null double-pointer inputs.
  EXPECT_EQ(orStreamCreate(nullptr), orErrorUnknown);
}

TEST_F(StreamTest, StreamCreateWithInvalidPriority) {
  orStream_t stream = nullptr;
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, min_p - 1), orErrorUnknown);
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, max_p + 1), orErrorUnknown);
}

TEST_F(StreamTest, StreamCreateWithPriorityValidBounds) {
  orStream_t stream = nullptr;
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  // Lowest priority should be accepted.
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, min_p), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);

  // Highest priority should also be accepted.
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, max_p), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, StreamDestroyNullptr) {
  // Destroying nullptr should follow CUDA error behavior.
  EXPECT_EQ(orStreamDestroy(nullptr), orErrorUnknown);
}

TEST_F(StreamTest, StreamGetPriority) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  int priority = -1;
  EXPECT_EQ(orStreamGetPriority(stream, &priority), orSuccess);
  EXPECT_EQ(priority, 0);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
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

TEST_F(StreamTest, AddTaskToStreamNullptr) {
  // Queueing work should fail fast if the stream handle is invalid.
  EXPECT_EQ(openreg::addTaskToStream(nullptr, [] {}), orErrorUnknown);
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

TEST_F(StreamTest, DeviceSynchronizeWithNoStreams) {
  // Even without registered streams, device sync should succeed.
  EXPECT_EQ(orDeviceSynchronize(), orSuccess);
}

TEST_F(StreamTest, StreamPriorityRange) {
  int min_p = -1;
  int max_p = -1;
  // OpenReg currently exposes only one priority level; verify the fixed range.
  EXPECT_EQ(orDeviceGetStreamPriorityRange(&min_p, &max_p), orSuccess);
  EXPECT_EQ(min_p, 0);
  EXPECT_EQ(max_p, 1);
}

} // namespace
