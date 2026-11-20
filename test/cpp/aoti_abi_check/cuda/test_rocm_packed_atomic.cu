// Official builds set ROCM_VERSION >= 6.4 and drop the packed fallback.
// tp2 leaves it at 0, so force that here and compile the branch.
#ifdef USE_ROCM
#undef ROCM_VERSION
#define ROCM_VERSION 0
#endif

#include <gtest/gtest.h>

#include <torch/headeronly/cuda/KernelUtils.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>

namespace {

#define HIP_ASSERT_OK(expr)                                               \
  do {                                                                    \
    hipError_t err_ = (expr);                                             \
    STD_TORCH_CHECK(                                                      \
        err_ == hipSuccess, #expr, " failed: ", hipGetErrorString(err_)); \
  } while (0)

__global__ void packed_atomic_kernel(
    torch::headeronly::Half* half,
    torch::headeronly::BFloat16* bfloat,
    float* fp32) {
  torch::headeronly::fastAtomicAdd(
      half,
      int64_t{0},
      int64_t{2},
      static_cast<torch::headeronly::Half>(1),
      true);
  torch::headeronly::fastAtomicAdd(
      bfloat,
      int64_t{0},
      int64_t{2},
      static_cast<torch::headeronly::BFloat16>(1),
      true);
  torch::headeronly::fastAtomicAdd(fp32, int64_t{0}, int64_t{1}, 1.f, true);
}

} // namespace

TEST(TestRocmPackedAtomic, LegacyPackedBranch) {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
    GTEST_SKIP() << "No HIP device available";
  }
  torch::headeronly::Half* half = nullptr;
  torch::headeronly::BFloat16* bfloat = nullptr;
  float* fp32 = nullptr;
  HIP_ASSERT_OK(hipMalloc(&half, 2 * sizeof(torch::headeronly::Half)));
  HIP_ASSERT_OK(hipMalloc(&bfloat, 2 * sizeof(torch::headeronly::BFloat16)));
  HIP_ASSERT_OK(hipMalloc(&fp32, sizeof(float)));
  HIP_ASSERT_OK(hipMemset(half, 0, 2 * sizeof(torch::headeronly::Half)));
  HIP_ASSERT_OK(hipMemset(bfloat, 0, 2 * sizeof(torch::headeronly::BFloat16)));
  HIP_ASSERT_OK(hipMemset(fp32, 0, sizeof(float)));
  packed_atomic_kernel<<<1, 1>>>(half, bfloat, fp32);
  HIP_ASSERT_OK(hipGetLastError());
  HIP_ASSERT_OK(hipDeviceSynchronize());
  float host = -1.f;
  HIP_ASSERT_OK(hipMemcpy(&host, fp32, sizeof(float), hipMemcpyDeviceToHost));
  EXPECT_EQ(host, 1.f);
  hipFree(half);
  hipFree(bfloat);
  hipFree(fp32);
}
#endif
