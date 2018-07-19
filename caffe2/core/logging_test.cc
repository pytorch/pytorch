#include <algorithm>

#include "caffe2/core/logging.h"
#include <gtest/gtest.h>

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
    ADD_FAILURE();
  } catch (const EnforceNotMet&) {
  }
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
}

TEST(LoggingTest, TestEnforceEquals) {
  int x = 4;
  int y = 5;
  try {
    CAFFE_ENFORCE_THAT(Equals(++x, ++y));
    // This should never be triggered.
    ADD_FAILURE();
  } catch (const EnforceNotMet& err) {
    EXPECT_NE(err.msg().find("5 vs 6"), string::npos);
  }

  // arguments are expanded only once
  CAFFE_ENFORCE_THAT(Equals(++x, y));
  EXPECT_EQ(x, 6);
  EXPECT_EQ(y, 6);
}

TEST(LoggingTest, EnforceShowcase) {
  // It's not really a test but rather a convenient thing that you can run and
  // see all messages
  int one = 1;
  int two = 2;
  int three = 3;
#define WRAP_AND_PRINT(exp)                     \
  try {                                         \
    exp;                                        \
  } catch (const EnforceNotMet&) {              \
    /* EnforceNotMet already does LOG(ERROR) */ \
  }
  WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_NE(one * 2, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_GT(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_GE(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_LT(three, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_LE(three, two));

  WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(
      one * two + three, three * two, "It's a pretty complicated expression"));

  WRAP_AND_PRINT(CAFFE_ENFORCE_THAT(Equals(one * two + three, three * two)));
}

TEST(LoggingTest, Join) {
  auto s = Join(", ", vector<int>({1, 2, 3}));
  EXPECT_EQ(s, "1, 2, 3");
  s = Join(":", vector<string>());
  EXPECT_EQ(s, "");
  s = Join(", ", set<int>({3, 1, 2}));
  EXPECT_EQ(s, "1, 2, 3");
}

#if GTEST_HAS_DEATH_TEST
TEST(LoggingDeathTest, TestEnforceUsingFatal) {
  bool kTrue = true;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
  EXPECT_DEATH(CAFFE_ENFORCE(false, "This goes fatal."), "");
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
}
#endif

} // namespace caffe2
