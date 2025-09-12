#include <gtest/gtest.h>

#include <torch/nativert/executor/AOTInductorDelegateExecutor.h>

using namespace ::testing;
using namespace torch::nativert;

TEST(AOTIModelContainerRegistrationTests, TestRegister) {
  EXPECT_TRUE(AOTIModelContainerRunnerRegistry()->Has(at::kCPU));

#ifdef USE_CUDA
  EXPECT_TRUE(AOTIModelContainerRunnerRegistry()->Has(at::kCUDA));
#else
  EXPECT_FALSE(AOTIModelContainerRunnerRegistry()->Has(at::kCUDA));
#endif // USE_CUDA
}
