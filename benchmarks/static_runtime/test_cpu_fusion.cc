#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;

TEST(CpuFusion, Simple) {
  const auto simple_script = R"JIT(
    def forward(self, a, b):
        return (a + b).relu().tanh()
  )JIT";

  Module m("module");
  m.define(simple_script);

  StaticModuleOptions opts; // start with the defaults.
  opts.enable_tensorexpr_fusion = true;

  auto input1 = at::randn({2, 3});
  auto input2 = at::ones({2, 3});

  auto smodule = StaticModule(m, /* is_frozen */ false, opts, {input1, input2});
  StaticRuntime runtime(smodule);

  // Test with sample inputs
  {
    auto actual = runtime({input1, input2}, {});
    auto expect = at::tanh(at::relu(input1 + input2));
    EXPECT_TRUE(at::allclose(expect, actual.toTensor()));
  }

  // Test with different inputs
  {
    auto new_input1 = at::randn({5, 14});
    auto new_input2 = at::randn({5, 14});
    auto actual = runtime({new_input1, new_input2}, {});
    auto expect = at::tanh(at::relu(new_input1 + new_input2));
    EXPECT_TRUE(at::allclose(expect, actual.toTensor()));
  }
}

TEST(CpuFusion, FallbackGraph) {
  const auto simple_script = R"JIT(
    def forward(self, a, b):
        return (a + b).relu().tanh()
  )JIT";

  Module m("module");
  m.define(simple_script);

  StaticModuleOptions opts; // start with the defaults.
  opts.enable_tensorexpr_fusion = true;

  auto sample_input1 = at::randn({2, 3});
  auto sample_input2 = at::ones({2, 3});
  auto smodule = StaticModule(
      m, /* is_frozen */ false, opts, {sample_input1, sample_input2});

  StaticRuntime runtime(smodule);

  // The sample inputs above were contiguous. Now, use a strided input
  // to trigger running the fallback graph.
  {
    auto input1 = at::narrow(at::randn({2, 6}), 1, 0, 3);
    auto input2 = at::ones({2, 3});
    auto expect = at::tanh(at::relu(input1 + input2));
    auto actual = runtime({input1, input2}, {});
    EXPECT_TRUE(at::allclose(expect, actual.toTensor()));
  }

  // Test with strided inputs of different size.
  {
    auto input1 = at::narrow(at::randn({10, 30}), 1, 0, 25);
    auto input2 = at::randn({10, 25});
    auto expect = at::tanh(at::relu(input1 + input2));
    auto actual = runtime({input1, input2}, {});
    EXPECT_TRUE(at::allclose(expect, actual.toTensor()));
  }
}
