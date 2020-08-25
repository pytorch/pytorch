#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/fft.h>

#include <test/cpp/api/support.h>

// NOTE: Visual Studio and ROCm builds don't understand complex literals
//   as of August 2020

// Simple test that verifies the fft namespace is registered properly
//   properly in C++
TEST(FFTTest, fft) {
    auto t = torch::randn(128, torch::dtype(torch::kComplexDouble));
    torch::fft::fft(t);
}
