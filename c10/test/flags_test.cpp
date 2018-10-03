#include <gtest/gtest.h>

#include <iostream>

#include "c10/util/Flags.h"

C10_DEFINE_bool(c10_flags_test_only_flag, true, "Only used in test.");

namespace c10 {

TEST(FlagsTest, TestGflagsCorrectness) {
#ifdef C10_USE_GFLAGS
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);
  EXPECT_EQ(::FLAGS_c10_flags_test_only_flag, true);
  // Change the c10 namespace and check global
  FLAGS_c10_flags_test_only_flag = false;
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, false);
  EXPECT_EQ(::FLAGS_c10_flags_test_only_flag, false);
  // Change global and check c10 namespace
  ::FLAGS_c10_flags_test_only_flag = true;
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);
  EXPECT_EQ(::FLAGS_c10_flags_test_only_flag, true);
#else // C10_USE_GFLAGS
  std::cout << "Caffe2 is not built with gflags. Nothing to test here."
            << std::endl;
#endif
}

} // namespace c10
