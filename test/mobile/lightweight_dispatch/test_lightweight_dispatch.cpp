#include <gtest/gtest.h>

std::string add_negative_flag(const std::string& flag) {
  std::string filter = ::testing::GTEST_FLAG(filter);
  if (filter.find('-') == std::string::npos) {
    filter.push_back('-');
  } else {
    filter.push_back(':');
  }
  filter += flag;
  return filter;
}
int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = add_negative_flag("*_CUDA:*_MultiCUDA");

    return RUN_ALL_TESTS();
}
