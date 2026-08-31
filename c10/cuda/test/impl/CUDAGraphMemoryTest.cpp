#include <c10/cuda/impl/CUDAGraphMemory.h>

#include <gtest/gtest.h>

namespace c10::cuda::CUDAGraphMemory {
namespace {

TEST(CUDAGraphMemoryCaptureTrackerTest, TracksConditionalCaptureTree) {
  CaptureTracker capture_tracker;

  capture_tracker.captureBegin({1, std::nullopt});
  capture_tracker.captureBegin({2, 1});
  EXPECT_TRUE(capture_tracker.hasActiveCaptures());

  capture_tracker.captureEnd(2);
  EXPECT_TRUE(capture_tracker.hasActiveCaptures());
  capture_tracker.captureEnd(1);
  EXPECT_FALSE(capture_tracker.hasActiveCaptures());
}

TEST(CUDAGraphMemoryCaptureTrackerTest, RootEndClearsActiveDescendants) {
  CaptureTracker capture_tracker;

  capture_tracker.captureBegin({1, std::nullopt});
  capture_tracker.captureBegin({2, 1});
  capture_tracker.captureEnd(1);

  EXPECT_FALSE(capture_tracker.hasActiveCaptures());
  EXPECT_THROW(capture_tracker.captureEnd(2), c10::Error);
}

} // namespace
} // namespace c10::cuda::CUDAGraphMemory
