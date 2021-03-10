#include <iostream>
#include <memory>

#define CAFFE2_TESTONLY_FORCE_STD_STRING_TEST

#include "caffe2/core/common.h"
#include <gtest/gtest.h>

namespace caffe2 {

#ifndef __ANDROID__

// Simple tests to make sure that our stoi and stod implementations are
// matching the std implementations, but not testing it very extensively
// as one should be using the std version most of the time.
TEST(CommonTest, TestStoi) {
  EXPECT_TRUE(CAFFE2_TESTONLY_WE_ARE_USING_CUSTOM_STRING_FUNCTIONS);
  string s = "1234";
  int i_std = std::stoi(s);
  int i_caffe2 = ::c10::stoi(s);
  EXPECT_EQ(i_std, i_caffe2);
}

TEST(CommonTest, TestStod) {
  // Full string is parsed.
  string s = "1.234";
  std::size_t p_std = 0, p_caffe2 = 0;
  double d_std = std::stod(s, &p_std);
  double d_caffe2 = ::c10::stod(s, &p_caffe2);
  EXPECT_EQ(d_std, d_caffe2);
  EXPECT_EQ(p_std, p_caffe2);

  // Only part of the string is parsed.
  s = "1.234 5.678";
  d_std = std::stod(s, &p_std);
  d_caffe2 = ::c10::stod(s, &p_caffe2);
  EXPECT_EQ(d_std, d_caffe2);
  EXPECT_EQ(p_std, p_caffe2);
}

#endif // __ANDROID__

}  // namespace caffe2


