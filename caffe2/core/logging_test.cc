#include <algorithm>

#include "caffe2/core/logging.h"
#include "gtest/gtest.h"

namespace caffe2 {

TEST(LoggingTest, TestEnforceTrue) {
  // This should just work.
  CAFFE_ENFORCE(true, "Isn't it?");
}

TEST(LoggingTest, TestEnforceFalse) {
  bool kFalse = false;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
  try {
    CAFFE_ENFORCE(false, "This throws.");
    // This should never be triggered.
    EXPECT_FALSE(true);
  } catch (const EnforceNotMet& err) {}
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
}

TEST(LoggingDeathTest, TestEnforceUsingFatal) {
  bool kTrue = true;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
  EXPECT_DEATH(
      CAFFE_ENFORCE(false, "This goes fatal."), "");
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
}

}  // namespace caffe2
