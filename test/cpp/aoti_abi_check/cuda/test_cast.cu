#include <gtest/gtest.h>

#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/FloatRounding.h>
#include <torch/headeronly/util/TypeCast.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

#define SKIP_IF_NO_CUDA()                                  \
  do {                                                     \
    int n = 0;                                             \
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { \
      GTEST_SKIP() << "No CUDA device available";          \
    }                                                      \
  } while (0)

#define CUDA_ASSERT_OK(expr)                                        \
  do {                                                              \
    cudaError_t err_ = (expr);                                      \
    STD_TORCH_CHECK(                                                \
        err_ == cudaSuccess,                                        \
        #expr,                                                      \
        " failed: ",                                                \
        cudaGetErrorString(err_));                                  \
  } while (0)

using torch::headeronly::BFloat16;
using torch::headeronly::convert;
using torch::headeronly::convert_with_rounding;
using torch::headeronly::Float8_e4m3fn;
using torch::headeronly::Float8_e8m0fnu;
using torch::headeronly::Half;
using torch::headeronly::RoundingMode;

constexpr int kModes = 4;
// Per input: round_to_odd_float, convert<Half>, then convert_with_rounding to
// Half, BFloat16 (whose subnormals cover float subnormals), Float8_e4m3fn
// (which saturates) and Float8_e8m0fnu (which has no zero), from the double
// and from the float, in each mode.
constexpr int kResults = 2 + 8 * kModes;

template <typename T>
__host__ __device__ uint32_t bits(T value) {
  if constexpr (sizeof(T) == 1) {
    uint8_t b;
    memcpy(&b, &value, sizeof(b));
    return b;
  } else if constexpr (sizeof(T) == 2) {
    uint16_t b;
    memcpy(&b, &value, sizeof(b));
    return b;
  } else {
    uint32_t b;
    memcpy(&b, &value, sizeof(b));
    return b;
  }
}

__host__ __device__ void round_all(double x, uint32_t* out) {
  out[0] = bits(torch::headeronly::detail::round_to_odd_float(x));
  out[1] = bits(convert<Half>(x));
  const float f = static_cast<float>(x);
  uint32_t* o = out + 2;
  for (int m = 0; m < kModes; ++m) {
    const auto mode = static_cast<RoundingMode>(m);
    *o++ = bits(convert_with_rounding<Half>(x, mode));
    *o++ = bits(convert_with_rounding<Half>(f, mode));
    *o++ = bits(convert_with_rounding<BFloat16>(x, mode));
    *o++ = bits(convert_with_rounding<BFloat16>(f, mode));
    *o++ = bits(convert_with_rounding<Float8_e4m3fn>(x, mode));
    *o++ = bits(convert_with_rounding<Float8_e4m3fn>(f, mode));
    *o++ = bits(convert_with_rounding<Float8_e8m0fnu>(x, mode));
    *o++ = bits(convert_with_rounding<Float8_e8m0fnu>(f, mode));
  }
}

__global__ void round_kernel(const double* x, uint32_t* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    round_all(x[i], out + i * kResults);
  }
}

// Device code (round_to_odd_float's __double2float_rz branch, and
// leading_zeros' __clzll, which normalizes subnormal sources) must give the
// host's bits.
TEST(TestCastCuda, ConvertFromDoubleDeviceMatchesHost) {
  SKIP_IF_NO_CUDA();
  std::vector<double> xs = {
      0.0,
      -0.0,
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      1e39,
      -1e39,
      65520.0,
      0x1p-149,
      0x1.8p-140,
      -0x1p-130,
      // Subnormal doubles.
      0x1p-1074,
      -0x1.8p-1060,
      0x1.234p-1030};
  // Half's values, the midpoints between them and their neighbours, and
  // float subnormals.
  for (int i = 0; i < 0x7C00; i += 7) {
    const double v = static_cast<float>(Half(i, Half::from_bits()));
    const double next = static_cast<float>(Half(i + 1, Half::from_bits()));
    const double mid = (v + next) / 2;
    for (const double x :
         {v, mid, std::nextafter(mid, 0.0), std::nextafter(mid, 1e9)}) {
      xs.push_back(x);
      xs.push_back(-x);
    }
  }
  for (int e = 1; e < 23; ++e) {
    xs.push_back(std::ldexp(1.0 + std::ldexp(1.0, -e), -140));
  }
  const int n = static_cast<int>(xs.size());

  double* d_x = nullptr;
  uint32_t* d_out = nullptr;
  CUDA_ASSERT_OK(cudaMalloc(&d_x, n * sizeof(double)));
  CUDA_ASSERT_OK(cudaMalloc(&d_out, n * kResults * sizeof(uint32_t)));
  CUDA_ASSERT_OK(
      cudaMemcpy(d_x, xs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  round_kernel<<<64, 128>>>(d_x, d_out, n);
  CUDA_ASSERT_OK(cudaGetLastError());
  std::vector<uint32_t> device(n * kResults);
  CUDA_ASSERT_OK(cudaMemcpy(
      device.data(),
      d_out,
      device.size() * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDA_ASSERT_OK(cudaFree(d_x));
  CUDA_ASSERT_OK(cudaFree(d_out));

  for (int i = 0; i < n; ++i) {
    uint32_t host[kResults];
    round_all(xs[i], host);
    for (int r = 0; r < kResults; ++r) {
      if (std::isnan(xs[i]) && r == 1) {
        // Half's own float conversion encodes NaN differently on device.
        EXPECT_EQ(device[i * kResults + r] & 0x7E00, 0x7E00u);
        continue;
      }
      ASSERT_EQ(device[i * kResults + r], host[r])
          << "x=" << xs[i] << " result " << r;
    }
  }
}

} // namespace
