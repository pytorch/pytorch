#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/fft.h>

#include <test/cpp/api/support.h>

// NOTE: Visual Studio doesn't understand complex literals
TEST(FFTTest, fft) {
  #ifdef _WIN32
    auto t = torch::randn(128, torch::dtype(torch::kComplexDouble));
    torch::fft:fft(t);
  #else
    auto t = torch::tensor({
      1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
      6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
      -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
      -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j}, torch::kComplexDouble);
    auto expected = torch::tensor({
        -2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
        2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
        -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
        1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j}, torch::kComplexDouble);

    auto result = torch::fft::fft(t);
    ASSERT_TRUE(torch::allclose(t, expected));
  #endif
}
