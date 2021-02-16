#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/special.h>

#include <test/cpp/api/support.h>

// NOTE: Visual Studio and ROCm builds don't understand complex literals
//   as of August 2020

// Simple test that verifies the special namespace is registered properly
//   properly in C++
TEST(SpecialTest, special) {
    auto t = torch::randn(128, torch::dtype(torch::kDouble));
    torch::special::lgamma(t);
}