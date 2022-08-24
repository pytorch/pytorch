#include <array>
#include <deque>
#include <memory>
#include <tuple>
#include <vector>

#include <benchmark/benchmark.h>

#include <c10/macros/Macros.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/util.h>

using namespace torch::profiler::impl;

// Make stored values opaque to the compiler.
template <typename T>
static C10_NOINLINE T opaque_get_x() {
  return T();
}

static C10_NOINLINE int opaque_round_robin(int i, int k) {
  return i % k;
}

template <typename T>
static void BM_store(benchmark::State& state) {
  T container;
  for (auto _ : state) {
    container.emplace_back(opaque_get_x<typename T::value_type>());
  }
}

template <typename T>
static void BM_store_round_robin(benchmark::State& state) {
  auto k = state.range(0);

  // AppendOnlyList is not movable, so we cannot use a vector.
  auto containers = std::make_unique<T[]>(k);

  int i = 0;
  for (auto _ : state) {
    containers[opaque_round_robin(i++, k)].emplace_back(
        opaque_get_x<typename T::value_type>());
  }
}

BENCHMARK_TEMPLATE(BM_store, std::vector<int>);
BENCHMARK_TEMPLATE(BM_store, std::deque<int>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<int, 64>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<int, 128>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<int, 512>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<int, 1024>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<int, 4 * 1024>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<int, 16 * 1024>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<int, 64 * 1024>);

BENCHMARK_TEMPLATE(BM_store, std::vector<std::array<int, 4>>);
BENCHMARK_TEMPLATE(BM_store, std::deque<std::array<int, 4>>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<std::array<int, 4>, 64>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<std::array<int, 4>, 512>);
BENCHMARK_TEMPLATE(BM_store, AppendOnlyList<std::array<int, 4>, 2048>);

#define BENCHMARK_ROUND_ROBIN(...)                      \
  BENCHMARK_TEMPLATE(BM_store_round_robin, __VA_ARGS__) \
      ->Args({4})                                       \
      ->Args({8})                                       \
      ->Args({12})
BENCHMARK_ROUND_ROBIN(std::vector<std::array<int, 4>>);
BENCHMARK_ROUND_ROBIN(std::deque<std::array<int, 4>>);
BENCHMARK_ROUND_ROBIN(AppendOnlyList<std::array<int, 4>, 64>);
BENCHMARK_ROUND_ROBIN(AppendOnlyList<std::array<int, 4>, 128>);
BENCHMARK_ROUND_ROBIN(AppendOnlyList<std::array<int, 4>, 512>);
BENCHMARK_ROUND_ROBIN(AppendOnlyList<std::array<int, 4>, 2048>);
#undef BENCHMARK_ROUND_ROBIN

static void BM_getTimeSinceEpoch(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(getTimeSinceEpoch());
  }
}

static void BM_getTime(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(getTime());
  }
}

static void BM_getApproximateTime(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(getApproximateTime());
  }
}

BENCHMARK(BM_getTimeSinceEpoch);
BENCHMARK(BM_getTime);
BENCHMARK(BM_getApproximateTime);

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
