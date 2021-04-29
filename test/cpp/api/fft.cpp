#include <gtest/gtest.h>

#include <torch/torch.h>
#include <test/cpp/api/support.h>


// Naive DFT of a 1 dimensional tensor
torch::Tensor naive_dft(torch::Tensor x, bool forward=true) {
  TORCH_INTERNAL_ASSERT(x.dim() == 1);
  x = x.contiguous();
  auto out_tensor = torch::zeros_like(x);
  const int64_t len = x.size(0);

  // Roots of unity, exp(-2*pi*j*n/N) for n in [0, N), reversed for inverse transform
  std::vector<c10::complex<double>> roots(len);
  const auto angle_base = (forward ? -2.0 : 2.0) * M_PI / len;
  for (int64_t i = 0; i < len; ++i) {
    auto angle = i * angle_base;
    roots[i] = c10::complex<double>(std::cos(angle), std::sin(angle));
  }

  const auto in = x.data_ptr<c10::complex<double>>();
  const auto out = out_tensor.data_ptr<c10::complex<double>>();
  for (int64_t i = 0; i < len; ++i) {
    for (int64_t j = 0; j < len; ++j) {
      out[i] += roots[(j * i) % len] * in[j];
    }
  }
  return out_tensor;
}

// NOTE: Visual Studio and ROCm builds don't understand complex literals
//   as of August 2020

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, fft) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(128, torch::kComplexDouble);
  auto actual = torch::fft::fft(t);
  auto expect = naive_dft(t);
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, fft_real) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(128, torch::kDouble);
  auto actual = torch::fft::fft(t);
  auto expect = torch::fft::fft(t.to(torch::kComplexDouble));
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, fft_pad) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(128, torch::kComplexDouble);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto actual = torch::fft::fft(t, 200);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto expect = torch::fft::fft(torch::constant_pad_nd(t, {0, 72}));
  ASSERT_TRUE(torch::allclose(actual, expect));

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  actual = torch::fft::fft(t, 64);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  expect = torch::fft::fft(torch::constant_pad_nd(t, {0, -64}));
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, fft_norm) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(128, torch::kComplexDouble);
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto unnorm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/{});
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto norm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"forward");
  ASSERT_TRUE(torch::allclose(unnorm / 128, norm));

  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto ortho_norm = torch::fft::fft(t, /*n=*/{}, /*axis=*/-1, /*norm=*/"ortho");
  ASSERT_TRUE(torch::allclose(unnorm / std::sqrt(128), ortho_norm));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, ifft) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto T = torch::randn(128, torch::kComplexDouble);
  auto actual = torch::fft::ifft(T);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto expect = naive_dft(T, /*forward=*/false) / 128;
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, fft_ifft) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(77, torch::kComplexDouble);
  auto T = torch::fft::fft(t);
  ASSERT_EQ(T.size(0), 77);
  ASSERT_EQ(T.scalar_type(), torch::kComplexDouble);

  auto t_round_trip = torch::fft::ifft(T);
  ASSERT_EQ(t_round_trip.size(0), 77);
  ASSERT_EQ(t_round_trip.scalar_type(), torch::kComplexDouble);
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, rfft) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(129, torch::kDouble);
  auto actual = torch::fft::rfft(t);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto expect = torch::fft::fft(t.to(torch::kComplexDouble)).slice(0, 0, 65);
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, rfft_irfft) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(128, torch::kDouble);
  auto T = torch::fft::rfft(t);
  ASSERT_EQ(T.size(0), 65);
  ASSERT_EQ(T.scalar_type(), torch::kComplexDouble);

  auto t_round_trip = torch::fft::irfft(T);
  ASSERT_EQ(t_round_trip.size(0), 128);
  ASSERT_EQ(t_round_trip.scalar_type(), torch::kDouble);
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, ihfft) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto T = torch::randn(129, torch::kDouble);
  auto actual = torch::fft::ihfft(T);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto expect = torch::fft::ifft(T.to(torch::kComplexDouble)).slice(0, 0, 65);
  ASSERT_TRUE(torch::allclose(actual, expect));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(FFTTest, hfft_ihfft) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto t = torch::randn(64, torch::kComplexDouble);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  t[0] = .5; // Must be purely real to satisfy hermitian symmetry
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto T = torch::fft::hfft(t, 127);
  ASSERT_EQ(T.size(0), 127);
  ASSERT_EQ(T.scalar_type(), torch::kDouble);

  auto t_round_trip = torch::fft::ihfft(T);
  ASSERT_EQ(t_round_trip.size(0), 64);
  ASSERT_EQ(t_round_trip.scalar_type(), torch::kComplexDouble);
  ASSERT_TRUE(torch::allclose(t, t_round_trip));
}
