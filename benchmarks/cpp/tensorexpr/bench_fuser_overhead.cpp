#include <benchmark/benchmark.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/torch.h>
#include <c10/core/InferenceMode.h>

using namespace torch::jit;

static const std::string two_adds = R"JIT(
def two_adds(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return x + y + z
)JIT";

static void FusedOverhead(benchmark::State& state) {
  c10::InferenceMode mode;
  overrideCanFuseOnCPU(true);

  Module m("m");
  m.define(two_adds);

  auto x = torch::ones({1});
  auto y = torch::ones({1});
  auto z = torch::ones({1});

  // Warmup.
  for (int i = 0; i < 8; i++) {
    m.run_method("two_adds", x, y, z);
  }

  for (auto _ : state) {
    m.run_method("two_adds", x, y, z);
  }
}

static void UnfusedOverhead(benchmark::State& state) {
  torch::NoGradGuard ng;
  torch::AutoNonVariableTypeMode nv;
  overrideCanFuseOnCPU(false);

  Module m("m");
  m.define(two_adds);

  auto x = torch::ones({1});
  auto y = torch::ones({1});
  auto z = torch::ones({1});

  // Warmup.
  for (int i = 0; i < 8; i++) {
    m.run_method("two_adds", x, y, z);
  }

  for (auto _ : state) {
    m.run_method("two_adds", x, y, z);
  }
}

BENCHMARK(FusedOverhead);
BENCHMARK(UnfusedOverhead);
