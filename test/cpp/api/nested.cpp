#include <gtest/gtest.h>

#include <torch/nested.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

// Simple test that verifies the nested namespace is registered properly
//   properly in C++
TEST(NestedTest, Nested) {
  auto a = torch::randn({2, 3});
  auto b = torch::randn({4, 5});
  auto nt = torch::nested::nested_tensor({a, b});
  torch::nested::to_padded_tensor(nt, 0);
}
