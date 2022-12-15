#include <gtest/gtest.h>

#include <ATen/native/Pow.h>
#include <c10/util/irange.h>
#include <test/cpp/api/support.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

struct DispatchTest : torch::test::SeedingFixture {};

TEST_F(DispatchTest, TestAVX2) {
  const std::vector<int> ints{1, 2, 3, 4};
  const std::vector<int> result{1, 4, 27, 256};
  const auto vals_tensor = torch::tensor(ints);
  const auto pows_tensor = torch::tensor(ints);
#ifdef _WIN32
  _putenv("ATEN_CPU_CAPABILITY=avx2");
#else
  setenv("ATEN_CPU_CAPABILITY", "avx2", 1);
#endif
  const auto actual_pow_avx2 = vals_tensor.pow(pows_tensor);
  for (const auto i : c10::irange(4)) {
    ASSERT_EQ(result[i], actual_pow_avx2[i].item<int>());
  }
}

TEST_F(DispatchTest, TestAVX512) {
  const std::vector<int> ints{1, 2, 3, 4};
  const std::vector<int> result{1, 4, 27, 256};
  const auto vals_tensor = torch::tensor(ints);
  const auto pows_tensor = torch::tensor(ints);
#ifdef _WIN32
  _putenv("ATEN_CPU_CAPABILITY=avx512");
#else
  setenv("ATEN_CPU_CAPABILITY", "avx512", 1);
#endif
  const auto actual_pow_avx512 = vals_tensor.pow(pows_tensor);
  for (const auto i : c10::irange(4)) {
    ASSERT_EQ(result[i], actual_pow_avx512[i].item<int>());
  }
}

TEST_F(DispatchTest, TestDefault) {
  const std::vector<int> ints{1, 2, 3, 4};
  const std::vector<int> result{1, 4, 27, 256};
  const auto vals_tensor = torch::tensor(ints);
  const auto pows_tensor = torch::tensor(ints);
#ifdef _WIN32
  _putenv("ATEN_CPU_CAPABILITY=default");
#else
  setenv("ATEN_CPU_CAPABILITY", "default", 1);
#endif
  const auto actual_pow_default = vals_tensor.pow(pows_tensor);
  for (const auto i : c10::irange(4)) {
    ASSERT_EQ(result[i], actual_pow_default[i].item<int>());
  }
}
