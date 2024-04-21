#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>

#include <assert.h>

using namespace at;

// optional in cuda files
TEST(OptionalTest, OptionalTestCUDA) {
  if (!at::cuda::is_available()) return;
  c10::optional<int64_t> trivially_destructible;
  c10::optional<std::vector<int64_t>> non_trivially_destructible;
  ASSERT_FALSE(trivially_destructible.has_value());
  ASSERT_FALSE(non_trivially_destructible.has_value());

  trivially_destructible = {5};
  non_trivially_destructible = std::vector<int64_t>{5, 10};
  ASSERT_TRUE(trivially_destructible.has_value());
  ASSERT_TRUE(non_trivially_destructible.has_value());
}
