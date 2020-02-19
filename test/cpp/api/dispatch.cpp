#include <gtest/gtest.h>

#include <torch/torch.h>
#include <ATen/native/Pow.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <test/cpp/api/support.h>
#include <iostream>
#include <vector>
#include <type_traits>
#include <cstdlib>

using namespace at;
using namespace torch::test;

struct DispatchTest : torch::test::SeedingFixture {};

TEST_F(DispatchTest, TestAVX2) {
  const std::vector<int> ints {1, 2, 3, 4};
  const std::vector<int> result {1, 4, 27, 256};
  const auto vals_tensor = torch::tensor(ints);
  const auto pows_tensor = torch::tensor(ints);
  setenv("ATEN_CPU_CAPABILITY", "avx2", 1);
  const auto actual_pow_avx2 = vals_tensor.pow(pows_tensor);
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(result[i], actual_pow_avx2[i].item<int>());
  }
}

TEST_F(DispatchTest, TestAVX) {
  const std::vector<int> ints {1, 2, 3, 4};
  const std::vector<int> result {1, 4, 27, 256};
  const auto vals_tensor = torch::tensor(ints);
  const auto pows_tensor = torch::tensor(ints);
  setenv("ATEN_CPU_CAPABILITY", "avx", 1);
  const auto actual_pow_avx = vals_tensor.pow(pows_tensor);
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(result[i], actual_pow_avx[i].item<int>());
  }
}

TEST_F(DispatchTest, TestDefault) {
  const std::vector<int> ints {1, 2, 3, 4};
  const std::vector<int> result {1, 4, 27, 256};
  const auto vals_tensor = torch::tensor(ints);
  const auto pows_tensor = torch::tensor(ints);
  setenv("ATEN_CPU_CAPABILITY", "default", 1);
  const auto actual_pow_default = vals_tensor.pow(pows_tensor);
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(result[i], actual_pow_default[i].item<int>());
  }
}
