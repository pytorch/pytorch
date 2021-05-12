#include <ATen/ATen.h>

#include <benchmark/benchmark.h>

static void tensor_add(benchmark::State& state) {
  const size_t batchSize = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  at::Tensor a = at::rand({batchSize, channels});
  at::Tensor b = at::rand({batchSize, channels});
  at::Tensor c;
  for (auto _ : state) {
    c = a + b;
  }
}

static void GenerateSizes(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "C"});

  for (size_t n = 8; n < 1024;) {
    for (size_t c = 8; c < 1024;) {
      b->Args({n, c});
      c *= 2;
    }
    n *= 2;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
BENCHMARK(tensor_add)->Apply(GenerateSizes);
BENCHMARK_MAIN();
