#include <gtest/gtest.h>

#include <torch/cuda.h>

#include <iostream>
#include <string>

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
  if (!torch::cuda::is_available()) {
    std::cout << "CUDA not available. Disabling CUDA and MultiCUDA tests"
              << std::endl;
    ::testing::GTEST_FLAG(filter) = add_negative_flag("*_CUDA:*_MultiCUDA");
  } else if (torch::cuda::device_count() < 2) {
    std::cout << "Only one CUDA device detected. Disabling MultiCUDA tests"
              << std::endl;
    ::testing::GTEST_FLAG(filter) = add_negative_flag("*_MultiCUDA");
  }

  return RUN_ALL_TESTS();
}
