#include <gtest/gtest.h>
#include <include/openreg.h>

#include <atomic>
#include <thread>

namespace {

class EventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orSetDevice(0);
  }
};

TEST_F(EventTest, EventCreateAndDestroy) {
  orEvent_t event = nullptr;
  EXPECT_EQ(orEventCreate(&event), orSuccess);
  EXPECT_NE(event, nullptr);

  EXPECT_EQ(orEventDestroy(event), orSuccess);
}

TEST_F(EventTest, EventCreateWithFlagsTiming) {
  orEvent_t event = nullptr;
  EXPECT_EQ(orEventCreateWithFlags(&event, orEventEnableTiming), orSuccess);
  EXPECT_NE(event, nullptr);

  EXPECT_EQ(orEventDestroy(event), orSuccess);
}

TEST_F(EventTest, EventRecordAndSynchronize) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  orEvent_t event = nullptr;
  EXPECT_EQ(orEventCreate(&event), orSuccess);

  EXPECT_EQ(orEventRecord(event, stream), orSuccess);
  EXPECT_EQ(orEventSynchronize(event), orSuccess);
  EXPECT_EQ(orEventQuery(event), orSuccess);

  EXPECT_EQ(orEventDestroy(event), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(EventTest, EventElapsedTime) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  orEvent_t start = nullptr;
  orEvent_t end = nullptr;
  EXPECT_EQ(orEventCreateWithFlags(&start, orEventEnableTiming), orSuccess);
  EXPECT_EQ(orEventCreateWithFlags(&end, orEventEnableTiming), orSuccess);

  EXPECT_EQ(orEventRecord(start, stream), orSuccess);

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_EQ(orEventRecord(end, stream), orSuccess);

  EXPECT_EQ(orEventSynchronize(start), orSuccess);
  EXPECT_EQ(orEventSynchronize(end), orSuccess);

  float elapsed_ms = 0.0f;
  EXPECT_EQ(orEventElapsedTime(&elapsed_ms, start, end), orSuccess);
  EXPECT_GE(elapsed_ms, 0.0f);

  EXPECT_EQ(orEventDestroy(start), orSuccess);
  EXPECT_EQ(orEventDestroy(end), orSuccess);
}

TEST_F(EventTest, StreamWaitEvent) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  orEvent_t event = nullptr;
  EXPECT_EQ(orEventCreate(&event), orSuccess);

  EXPECT_EQ(orEventRecord(event, stream), orSuccess);
  EXPECT_EQ(orStreamWaitEvent(stream, event, 0), orSuccess);

  EXPECT_EQ(orEventSynchronize(event), orSuccess);
  EXPECT_EQ(orEventDestroy(event), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

} // namespace
