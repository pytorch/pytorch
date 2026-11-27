// Official builds set ROCM_VERSION >= 60400, which drops the packed fallback.
// Force it to 0 here so this TU compiles that branch.
#ifdef USE_ROCM
#undef ROCM_VERSION
#define ROCM_VERSION 0
// Run the scoped CAS even on GPUs that have the packed hardware atomic.
#define TORCH_HEADERONLY_INTERNAL_TEST_FORCE_PACKED_CAS 1
#endif

#include <gtest/gtest.h>

#include <torch/headeronly/cuda/KernelUtils.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include <cstdint>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>

namespace {

#define HIP_ASSERT_OK(expr)                                               \
  do {                                                                    \
    hipError_t err_ = (expr);                                             \
    STD_TORCH_CHECK(                                                      \
        err_ == hipSuccess, #expr, " failed: ", hipGetErrorString(err_)); \
  } while (0)

// Every thread adds 1 to the same two packed words, so CASes contend and
// retry.
constexpr int kBlocks = 2;
constexpr int kThreadsPerBlock = 64;
constexpr int kThreads = kBlocks * kThreadsPerBlock;
constexpr float kTargetSeed = 3.f;
constexpr float kExpected = kTargetSeed + kThreads;
static_assert(kExpected <= 256.f, "bfloat16 integers are exact only to 256");

// Partners hold 5.0 and -0.0; -0.0 padding keeps both bit-for-bit, while
// +0.0 padding would flip the -0.0.
constexpr float kPartnerSeed = 5.f;
constexpr uint16_t kNegZeroBits = 0x8000; // -0.0 in both fp16 and bf16

// The 4-element buffer is two 32-bit words. Index 0 pairs with partner 1
// (low lane) and index 3 with partner 2 (high lane).
constexpr int64_t kNumel = 4;
constexpr int64_t kLowTarget = 0;
constexpr int64_t kLowPartner = 1;
constexpr int64_t kHighPartner = 2;
constexpr int64_t kHighTarget = 3;

__global__ void packed_atomic_kernel(
    torch::headeronly::Half* half,
    torch::headeronly::BFloat16* bfloat,
    float* fp32) {
  const auto one_h = static_cast<torch::headeronly::Half>(1);
  const auto one_bf = static_cast<torch::headeronly::BFloat16>(1);
  torch::headeronly::fastAtomicAdd(half, kLowTarget, kNumel, one_h, true);
  torch::headeronly::fastAtomicAdd(half, kHighTarget, kNumel, one_h, true);
  torch::headeronly::fastAtomicAdd(bfloat, kLowTarget, kNumel, one_bf, true);
  torch::headeronly::fastAtomicAdd(bfloat, kHighTarget, kNumel, one_bf, true);
  torch::headeronly::fastAtomicAdd(fp32, int64_t{0}, int64_t{1}, 1.f, true);
}

template <typename scalar_t>
void seed(scalar_t* host) {
  host[kLowTarget] = static_cast<scalar_t>(kTargetSeed);
  host[kLowPartner] = static_cast<scalar_t>(kPartnerSeed);
  host[kHighPartner].x = kNegZeroBits;
  host[kHighTarget] = static_cast<scalar_t>(kTargetSeed);
}

// A CAS that stored the addend instead of old + addend would leave each
// target at 1 and overwrite the 5.0 partner with -0.0.
template <typename scalar_t>
void expectPackedSums(const scalar_t* out, const char* type) {
  EXPECT_EQ(static_cast<float>(out[kLowTarget]), kExpected) << type;
  EXPECT_EQ(static_cast<float>(out[kHighTarget]), kExpected) << type;
  EXPECT_EQ(out[kLowPartner].x, static_cast<scalar_t>(kPartnerSeed).x)
      << type << " low partner";
  EXPECT_EQ(out[kHighPartner].x, kNegZeroBits) << type << " high partner";
}

} // namespace

TEST(TestRocmPackedAtomic, LegacyPackedBranch) {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
    GTEST_SKIP() << "No HIP device available";
  }
  torch::headeronly::Half half_host[kNumel];
  torch::headeronly::BFloat16 bfloat_host[kNumel];
  float fp32_host = kTargetSeed;
  seed(half_host);
  seed(bfloat_host);

  torch::headeronly::Half* half = nullptr;
  torch::headeronly::BFloat16* bfloat = nullptr;
  float* fp32 = nullptr;
  HIP_ASSERT_OK(hipMalloc(&half, sizeof(half_host)));
  HIP_ASSERT_OK(hipMalloc(&bfloat, sizeof(bfloat_host)));
  HIP_ASSERT_OK(hipMalloc(&fp32, sizeof(fp32_host)));
  HIP_ASSERT_OK(
      hipMemcpy(half, half_host, sizeof(half_host), hipMemcpyHostToDevice));
  HIP_ASSERT_OK(hipMemcpy(
      bfloat, bfloat_host, sizeof(bfloat_host), hipMemcpyHostToDevice));
  HIP_ASSERT_OK(
      hipMemcpy(fp32, &fp32_host, sizeof(fp32_host), hipMemcpyHostToDevice));

  packed_atomic_kernel<<<kBlocks, kThreadsPerBlock>>>(half, bfloat, fp32);
  HIP_ASSERT_OK(hipGetLastError());
  HIP_ASSERT_OK(hipDeviceSynchronize());

  HIP_ASSERT_OK(
      hipMemcpy(half_host, half, sizeof(half_host), hipMemcpyDeviceToHost));
  HIP_ASSERT_OK(hipMemcpy(
      bfloat_host, bfloat, sizeof(bfloat_host), hipMemcpyDeviceToHost));
  HIP_ASSERT_OK(
      hipMemcpy(&fp32_host, fp32, sizeof(fp32_host), hipMemcpyDeviceToHost));
  HIP_ASSERT_OK(hipFree(half));
  HIP_ASSERT_OK(hipFree(bfloat));
  HIP_ASSERT_OK(hipFree(fp32));

  expectPackedSums(half_host, "Half");
  expectPackedSums(bfloat_host, "BFloat16");
  EXPECT_EQ(fp32_host, kExpected);
}
#endif
