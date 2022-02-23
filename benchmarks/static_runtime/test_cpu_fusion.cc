#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>
#include <thread>

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

TEST(CpuFusion, ParallelRuntimes) {
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

  auto exec_runtime = [&](const std::vector<std::pair<int, int>> inputs) {
    StaticRuntime runtime(smodule);
    for (auto inp : inputs) {
      auto a = at::randn({inp.first, inp.second});
      auto b = at::randn({inp.first, inp.second});
      auto expect = at::tanh(at::relu(a + b));
      auto actual = runtime({a, b}, {});
      EXPECT_TRUE(at::allclose(expect, actual.toTensor()));
    }
  };

  constexpr size_t kNumThreads = 2;
  std::vector<std::thread> threads;
  for (size_t id = 0; id < kNumThreads; ++id) {
    std::vector<std::pair<int, int>> inputs = {
        {id, id + 1},
        {id + 10, id + 11},
        {id + 20, id + 21},
        {id + 30, id + 31},
        {id + 40, id + 41},
        {id + 50, id + 51},
        {id + 60, id + 61},
        {id + 70, id + 71}};
    threads.emplace_back(exec_runtime, inputs);
  }

  for (auto& t : threads) {
    t.join();
  }
}
