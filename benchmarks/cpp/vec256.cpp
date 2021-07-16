#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>
#include <benchmark/benchmark.h>

using namespace at::vec;

void __attribute__((noinline)) loadu(float* __restrict__ ap, int64_t an) {
  benchmark::DoNotOptimize(Vectorized<float>::loadu(ap, an));
}

void __attribute__((noinline))
store(Vectorized<float> vec, float* __restrict__ ap, int64_t an) {
  vec.store(ap, an);
}

static void Loadu(benchmark::State& state) {
  auto a = at::randn({7});
  auto ap = a.data_ptr<float>();
  auto an = a.numel();
  for (auto _ : state) {
    loadu(ap, an);
  }
  auto b = at::empty({7});
  auto vec = Vectorized<float>::loadu(ap, an);
  vec.store(b.data_ptr<float>(), b.numel());
  TORCH_INTERNAL_ASSERT(at::equal(a, b));
}
BENCHMARK(Loadu);

static void Store(benchmark::State& state) {
  auto a = at::randn({7});
  auto b = at::empty({7});
  auto bp = b.data_ptr<float>();
  auto bn = b.numel();
  auto vec = Vectorized<float>::loadu(a.data_ptr<float>(), a.numel());
  for (auto _ : state) {
    store(vec, bp, bn);
  }
  TORCH_INTERNAL_ASSERT(at::equal(a, b));
}
BENCHMARK(Store);

BENCHMARK_MAIN();
