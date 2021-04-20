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

static void BM_deep_wide_fast(benchmark::State& state) {
  std::shared_ptr<DeepAndWideFast> net =
      std::make_shared<DeepAndWideFast>(num_features);

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
  torch::jit::StaticModule smod(mod);

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<at::Tensor> inputs({ad_emb_packed, user_emb, wide});

  smod(inputs);
  for (auto _ : state) {
    smod(inputs);
  }
}

torch::jit::StaticRuntime getStaticRuntime() {
  static auto smod = std::make_shared<torch::jit::StaticModule>(getDeepAndWideSciptModel());
  return torch::jit::StaticRuntime(*smod);
}

static void BM_deep_wide_static_threaded(benchmark::State& state) {
   auto sr = getStaticRuntime();

  const int batch_size = 1; // state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<at::Tensor> inputs({ad_emb_packed, user_emb, wide});

  sr(inputs);
  for (auto _ : state) {
    sr(inputs);
  }
}

static void BM_leaky_relu_const(benchmark::State& state) {
  auto mod = getLeakyReLUConstScriptModel();
  torch::jit::StaticModule smod(mod);

  const int batch_size = state.range(0);
  auto data = torch::randn({batch_size, num_features});
  std::vector<at::Tensor> inputs({data});

  smod(inputs);
  for (auto _ : state) {
    smod(inputs);
  }
}

static void BM_leaky_relu(benchmark::State& state) {
  auto mod = getLeakyReLUScriptModel();
  torch::jit::StaticModule smod(mod);

  const int batch_size = state.range(0);
  auto neg_slope = torch::randn(1);
  auto data = torch::randn({batch_size, num_features});
  std::vector<at::Tensor> inputs({data, neg_slope[0]});

  smod(inputs);
  for (auto _ : state) {
    smod(inputs);
  }
}

BENCHMARK(BM_leaky_relu)->RangeMultiplier(8)->Ranges({{1, 20}});
BENCHMARK(BM_leaky_relu_const)->RangeMultiplier(8)->Ranges({{1, 20}});

static void BM_long_static_memory_optimization(benchmark::State& state) {
  auto mod = getLongScriptModel();
  torch::jit::StaticModuleOptions opts;
  opts.optimize_memory = state.range(1);
  torch::jit::StaticModule smod(mod, opts);

  const auto N = state.range(0);
  auto a = torch::randn({N, N});
  auto b = torch::randn({N, N});
  auto c = torch::randn({N, N});
  std::vector<at::Tensor> inputs({a, b, c});

  smod(inputs);
  for (auto _ : state) {
    smod(inputs);
  }
}

BENCHMARK(BM_deep_wide_base)->RangeMultiplier(8)->Ranges({{1, 20}});
BENCHMARK(BM_deep_wide_fast)->RangeMultiplier(8)->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_jit_graph_executor)
    ->RangeMultiplier(8)
    ->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_jit_profiling_executor)
    ->RangeMultiplier(8)
    ->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_static)->RangeMultiplier(8)->Ranges({{1, 20}});
BENCHMARK(BM_deep_wide_static_threaded)->Threads(8);

BENCHMARK(BM_long_static_memory_optimization)
  ->Args({2<<0, 0})
  ->Args({2<<2, 0})
  ->Args({2<<4, 0})
  ->Args({2<<8, 0})
  ->Args({2<<0, 1})
  ->Args({2<<2, 1})
  ->Args({2<<4, 1})
  ->Args({2<<8, 1});

int main(int argc, char** argv)
{
  c10::ParseCommandLineFlags(&argc, &argv);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
