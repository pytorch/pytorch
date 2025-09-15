#include <gtest/gtest.h>

#include <torch/nativert/kernels/TritonKernel.h>

using namespace ::testing;
using namespace torch::nativert;

TEST(TritonKernelManagerRegistrationTests, TestRegister) {
  EXPECT_TRUE(TritonKernelManagerRegistry()->Has(at::kCPU));

#ifdef USE_CUDA
#ifdef USE_ROCM
  EXPECT_TRUE(TritonKernelManagerRegistry()->Has(at::kHIP));
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kCUDA));

#else
  EXPECT_TRUE(TritonKernelManagerRegistry()->Has(at::kCUDA));
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kHIP));

#endif // USE_ROCM
#else
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kCUDA));
  EXPECT_FALSE(TritonKernelManagerRegistry()->Has(at::kHIP));
#endif // USE_CUDA
}
