#include <gtest/gtest.h>

#include <ATen/native/Pow.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <test/cpp/api/support.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <cstdlib>
#include <vector>

struct DispatchTest : torch::test::SeedingFixture {};

TEST_F(DispatchTest, TestAVX2) {
  const std::vector<int> ints{1, 2, 3, 4};
  const std::vector<int> result{1, 4, 27, 256};
  const auto vals_tensor = torch::tensor(ints);
  const auto pows_tensor = torch::tensor(ints);
  c10::utils::set_env("ATEN_CPU_CAPABILITY", "avx2");
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
  c10::utils::set_env("ATEN_CPU_CAPABILITY", "avx512");
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
  c10::utils::set_env("ATEN_CPU_CAPABILITY", "default");
  const auto actual_pow_default = vals_tensor.pow(pows_tensor);
  for (const auto i : c10::irange(4)) {
    ASSERT_EQ(result[i], actual_pow_default[i].item<int>());
  }
}
