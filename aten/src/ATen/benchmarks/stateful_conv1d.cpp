#include <benchmark/benchmark.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>

#include <vector>

static void stateful_conv1d(benchmark::State& state) {
  const size_t input_channels = static_cast<size_t>(state.range(0));
  const size_t output_channels = static_cast<size_t>(state.range(1));
  const size_t kernel = static_cast<size_t>(state.range(2));
  const size_t batch_size = static_cast<size_t>(state.range(3));
  const size_t width = static_cast<size_t>(state.range(4));
  const bool optimized = static_cast<bool>(state.range(5));

  torch::jit::Module m("m");
  m.register_parameter("weight_1", torch::rand({output_channels, input_channels, kernel}), false);
  m.register_parameter("bias_1", torch::rand({output_channels}), false);
  m.register_parameter("weight_2", torch::rand({output_channels, output_channels, kernel}), false);
  m.register_parameter("bias_2", torch::rand({output_channels}), false);
  m.register_parameter("weight_3", torch::rand({output_channels, output_channels, kernel}), false);
  m.register_parameter("bias_3", torch::rand({output_channels}), false);
  m.register_parameter("weight_4", torch::rand({output_channels, output_channels, kernel}), false);
  m.register_parameter("bias_4", torch::rand({output_channels}), false);

  m.define(R"(
    def forward(self, x):
      x = torch.conv1d(x, self.weight_1, self.bias_1, 1, 0, 1, 1)
      x = torch.conv1d(x, self.weight_2, self.bias_2, 1, 0, 1, 1)
      x = torch.conv1d(x, self.weight_3, self.bias_3, 1, 0, 1, 1)
      x = torch.conv1d(x, self.weight_4, self.bias_4, 1, 0, 1, 1)
      return x
  )");

  std::vector<std::vector<torch::jit::IValue>> inputs;
  for (const auto i : c10::irange(10)) {
    inputs.emplace_back(
        {torch::jit::IValue(torch::rand({batch_size, input_channels, width}))});
  }

  auto m_cloned = m.clone();
  torch::jit::transformConv1dToConv2d(m_cloned);
  auto m_optimized = torch::jit::optimizeForMobile(m_cloned);
  torch::jit::IValue output;

  if (!optimized) {
    for (auto _ : state) {
      for (const auto& input : inputs) {
        output = m.forward(input);
      }
    }
  } else {
    for (auto _ : state) {
      for (const auto& input : inputs) {
        output = m_optimized.forward(input);
      }
    }
  }
}

static void GenerateSizes(benchmark::internal::Benchmark* b) {
  b->ArgNames({"Input Channels",
               "Output Channels",
               "Kernel",
               "Batch Size",
               "Width",
               "Optimized"});

  for (size_t input_channels = 32; input_channels < 256; input_channels *= 2) {
    for (size_t output_channels = 32; output_channels < 256; output_channels *= 2) {
      for (const auto kernel : c10::irange(3, 8)) {
        for (const auto batch_size : c10::irange(1, 5)) {
          for (size_t width = 32; width < 256; width *= 2) {
            b->Args({input_channels, output_channels, kernel, batch_size, width, true});
            b->Args({input_channels, output_channels, kernel, batch_size, width, false});
          }
        }
      }
    }
  }
}

BENCHMARK(stateful_conv1d)->Apply(GenerateSizes);
BENCHMARK_MAIN();
