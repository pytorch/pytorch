#include <gtest/gtest.h>

#include <torch/torch.h>
#include <test/cpp/api/support.h>


// Tests that the fft function can be called as usual
TEST(FFTTest, unclobbered_fft) {
    auto t = torch::randn({64, 2}, torch::dtype(torch::kDouble));
    torch::fft(t, 1);
}

// Clobbers torch::fft the function with torch::fft the namespace
#include <torch/fft.h>


// NOTE: Visual Studio and ROCm builds don't understand complex literals
//   as of August 2020

// Simple test that verifies the fft namespace is registered properly
//   properly in C++
TEST(FFTTest, fft) {
    auto t = torch::randn(128, torch::dtype(torch::kComplexDouble));
    torch::fft::fft(t);
}
