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

TEST_F(EventTest, EventCreationNullptr) {
  // Creation APIs must fail fast on null handles to mirror CUDA semantics.
  EXPECT_EQ(orEventCreate(nullptr), orErrorUnknown);
  EXPECT_EQ(
      orEventCreateWithFlags(nullptr, orEventEnableTiming), orErrorUnknown);
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

TEST_F(EventTest, EventRecordInvalidArgs) {
  orEvent_t event = nullptr;
  EXPECT_EQ(orEventCreate(&event), orSuccess);

  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  // Record/sync/destroy should validate both stream and event pointers.
  EXPECT_EQ(orEventRecord(nullptr, stream), orErrorUnknown);
  EXPECT_EQ(orEventRecord(event, nullptr), orErrorUnknown);
  EXPECT_EQ(orEventSynchronize(nullptr), orErrorUnknown);
  EXPECT_EQ(orEventDestroy(nullptr), orErrorUnknown);

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

TEST_F(EventTest, EventElapsedTimeDifferentDevicesFails) {
  orStream_t stream1 = nullptr;
  orStream_t stream2 = nullptr;
  orEvent_t event1= nullptr;
  orEvent_t event2 = nullptr;

  EXPECT_EQ(orSetDevice(0), orSuccess);
  EXPECT_EQ(orEventCreateWithFlags(&event1, orEventEnableTiming), orSuccess);
  EXPECT_EQ(orStreamCreate(&stream1), orSuccess);

  // Switch device before creating the end event to force a mismatch.
  EXPECT_EQ(orSetDevice(1), orSuccess);
  EXPECT_EQ(orEventCreateWithFlags(&event2, orEventEnableTiming), orSuccess);
  EXPECT_EQ(orStreamCreate(&stream2), orSuccess);

  // recording events to a stream is not allowed if the stream and the event are not on the same device
  EXPECT_EQ(orEventRecord(event1, stream2), orErrorUnknown);
  EXPECT_EQ(orEventRecord(event2, stream1), orErrorUnknown);

  EXPECT_EQ(orEventRecord(event1, stream1), orSuccess);
  EXPECT_EQ(orEventRecord(event2, stream2), orSuccess);
  EXPECT_EQ(orEventSynchronize(event1), orSuccess);
  EXPECT_EQ(orEventSynchronize(event2), orSuccess);

  float elapsed_ms = 0.0f;
  EXPECT_EQ(orEventElapsedTime(&elapsed_ms, event1, event2), orErrorUnknown);

  EXPECT_EQ(orEventDestroy(event1), orSuccess);
  EXPECT_EQ(orEventDestroy(event2), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream1), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream2), orSuccess);
}

TEST_F(EventTest, EventElapsedTimeRequiresTimingFlag) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  orEvent_t start = nullptr;
  orEvent_t end = nullptr;
  EXPECT_EQ(orEventCreate(&start), orSuccess);
  EXPECT_EQ(orEventCreate(&end), orSuccess);

  EXPECT_EQ(orEventRecord(start, stream), orSuccess);
  EXPECT_EQ(orEventRecord(end, stream), orSuccess);
  EXPECT_EQ(orEventSynchronize(start), orSuccess);
  EXPECT_EQ(orEventSynchronize(end), orSuccess);

  // Without timing-enabled events, querying elapsed time must fail.
  float elapsed_ms = 0.0f;
  EXPECT_EQ(orEventElapsedTime(&elapsed_ms, start, end), orErrorUnknown);

  EXPECT_EQ(orEventDestroy(start), orSuccess);
  EXPECT_EQ(orEventDestroy(end), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
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

TEST_F(EventTest, StreamWaitEventInvalidArgs) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  orEvent_t event = nullptr;
  EXPECT_EQ(orEventCreate(&event), orSuccess);

  // Validate both stream and event inputs for wait calls.
  EXPECT_EQ(orStreamWaitEvent(nullptr, event, 0), orErrorUnknown);
  EXPECT_EQ(orStreamWaitEvent(stream, nullptr, 0), orErrorUnknown);

  EXPECT_EQ(orEventDestroy(event), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

} // namespace
