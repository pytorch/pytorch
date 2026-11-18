#include <gtest/gtest.h>

#include <ATen/TunableOpExecution.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/Tunable.h>

TEST(TunableOpLegacyTest, ProcessSettingsRemainAvailableWithoutManagedOptIn) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "requires a CUDA or HIP device";
  }
  EXPECT_FALSE(at::IsManagedTunableOp());
  at::cuda::tunable::TuningContext context;
  context.EnableTunableOp(false);
  EXPECT_FALSE(context.IsTunableOpEnabled());
  context.EnableTunableOp(true);
  EXPECT_TRUE(context.IsTunableOpEnabled());
  context.EnableTuning(true);
  EXPECT_TRUE(context.IsTuningEnabled());
  context.EnableWildcardFallback(true);
  EXPECT_TRUE(context.IsWildcardFallbackEnabled());
}
