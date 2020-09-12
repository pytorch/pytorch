#include <benchmark/benchmark.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include "deep_wide_pt.h"

const int embedding_size = 32;
const int num_features = 50;

using namespace torch;

static void BM_deep_wide_base(benchmark::State& state) {
  std::shared_ptr<DeepAndWide> net =
      std::make_shared<DeepAndWide>(num_features);

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});
  // warmup
  net->forward(ad_emb_packed, user_emb, wide);
  for (auto _ : state) {
    net->forward(ad_emb_packed, user_emb, wide);
  }
}

static void BM_deep_wide_jit_graph_executor(benchmark::State& state) {
  auto mod = getDeepAndWideSciptModel();

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<IValue> inputs({ad_emb_packed, user_emb, wide});

  CHECK_EQ(setenv("TORCH_JIT_DISABLE_NEW_EXECUTOR", "1", 1), 0);

  mod.forward(inputs);
  for (auto _ : state) {
    mod.forward(inputs);
  }
}

static void BM_deep_wide_jit_profiling_executor(benchmark::State& state) {
  auto mod = getDeepAndWideSciptModel();

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<IValue> inputs({ad_emb_packed, user_emb, wide});

  CHECK_EQ(unsetenv("TORCH_JIT_DISABLE_NEW_EXECUTOR"), 0);

  mod.forward(inputs);
  for (auto _ : state) {
    mod.forward(inputs);
  }
}

static void BM_deep_wide_static(benchmark::State& state) {
  auto mod = getDeepAndWideSciptModel();
  torch::jit::StaticRuntime runtime(mod);

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<at::Tensor> inputs({ad_emb_packed, user_emb, wide});

  runtime.run(inputs);
  for (auto _ : state) {
    runtime.run(inputs);
  }
}

BENCHMARK(BM_deep_wide_base)->RangeMultiplier(8)->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_jit_graph_executor)
    ->RangeMultiplier(8)
    ->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_jit_profiling_executor)
    ->RangeMultiplier(8)
    ->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_static)->RangeMultiplier(8)->Ranges({{1, 20}});

BENCHMARK_MAIN();
