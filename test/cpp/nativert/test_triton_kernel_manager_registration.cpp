#include <gtest/gtest.h>

#include <torch/nativert/kernels/TritonKernel.h>

using namespace ::testing;
using namespace torch::nativert;

TEST(TritonKernelManagerRegistrationTests, TestRegister) {
#ifndef USE_CUDA
  EXPECT_TRUE(create_cuda_triton_kernel_manager == nullptr);
#else
  EXPECT_FALSE(create_cuda_triton_kernel_manager == nullptr);
#endif // USE_CUDA
}
