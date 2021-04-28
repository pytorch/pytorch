#include <gtest/gtest.h>

#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/torch.h>

using namespace torch::jit;

#ifdef USE_XNNPACK

TEST(MemoryCleanUp, NoErrorWithoutRelease) {
  Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");
  m.eval();
  auto m_optimized = optimizeForMobile(m);
  std::stringstream ss;
  EXPECT_NO_THROW(m_optimized.save(ss));
}

TEST(MemoryCleanUp, UnpackError) {
  at::globalContext().setReleaseWeightsWhenPrepacking(true);
  Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
  )");
  m.eval();
  auto m_optimized = optimizeForMobile(m);
  std::stringstream ss;
  EXPECT_ANY_THROW(m_optimized.save(ss));
}

#endif
