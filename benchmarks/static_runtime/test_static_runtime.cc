#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include "deep_wide_pt.h"

TEST(StaticRuntime, TrivialModel) {
  torch::jit::Module mod = getTrivialScriptModel();
  auto a = torch::randn({2, 2});
  auto b = torch::randn({2, 2});
  auto c = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({a, b, c});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({a, b, c});
  torch::jit::StaticRuntime runtime(mod);
  at::Tensor output_2 = runtime.run(input_tensors)[0];
  EXPECT_TRUE(output_1.equal(output_2));
}
