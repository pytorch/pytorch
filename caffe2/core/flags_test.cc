#include <gtest/gtest.h>
#include "caffe2/core/macros.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_bool(caffe2_flags_test_only_flag, true, "Only used in test.");

namespace caffe2 {

TEST(FlagsTest, TestGflagsCorrectness) {
#ifdef CAFFE2_USE_GFLAGS
  EXPECT_EQ(FLAGS_caffe2_flags_test_only_flag, true);
  EXPECT_EQ(::FLAGS_caffe2_flags_test_only_flag, true);
  // Change the caffe2 namespace and check global
  FLAGS_caffe2_flags_test_only_flag = false;  
  EXPECT_EQ(FLAGS_caffe2_flags_test_only_flag, false);
  EXPECT_EQ(::FLAGS_caffe2_flags_test_only_flag, false);
  // Change global and check caffe2 namespace
  ::FLAGS_caffe2_flags_test_only_flag = true;  
  EXPECT_EQ(FLAGS_caffe2_flags_test_only_flag, true);
  EXPECT_EQ(::FLAGS_caffe2_flags_test_only_flag, true);
#else  // CAFFE2_USE_GFLAGS
  LOG(INFO) << "Caffe2 is not built with gflags. Nothing to test here.";
#endif
}

} // namespace caffe2
