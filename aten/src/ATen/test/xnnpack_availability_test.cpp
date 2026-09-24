#include <gtest/gtest.h>

#include <ATen/Context.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>

#ifndef EXPECT_XNNPACK_AVAILABLE
#error EXPECT_XNNPACK_AVAILABLE must be defined
#endif

TEST(XNNPACKAvailabilityTest, MatchesLinkedProvider) {
  EXPECT_EQ(
      at::globalContext().isXNNPACKAvailable(), EXPECT_XNNPACK_AVAILABLE);
}

TEST(XNNPACKAvailabilityTest, OptimizerRequiresLinkedProvider) {
  auto graph = std::make_shared<torch::jit::Graph>();
#if EXPECT_XNNPACK_AVAILABLE
  EXPECT_NO_THROW(torch::jit::insertPrePackedOps(graph));
#else
  EXPECT_ANY_THROW(torch::jit::insertPrePackedOps(graph));
#endif
}
