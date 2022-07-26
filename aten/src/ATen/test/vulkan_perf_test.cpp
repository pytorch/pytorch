#ifdef USE_VULKAN_API

#include <benchmark/benchmark.h>

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>

namespace {

static void add_op_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches = state.range(0);
  const auto channels = state.range(1);
  const auto height = state.range(2);
  const auto width = state.range(3);
  const auto in_cpu1 = at::rand({batches, channels, height, width}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({batches, channels, height, width}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan1 = in_cpu1.vulkan();
  const auto in_vulkan2 = in_cpu2.vulkan();

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    const auto vulkan_out = at::add(in_vulkan1, in_vulkan2).cpu();
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
#endif
}

static void CommonBenchmarkSettings(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMillisecond);
  b->ArgNames({"N", "C", "H", "W"});
}

} // namespace

BENCHMARK(add_op_benchmark)->Apply(CommonBenchmarkSettings)->UseManualTime()->Threads(1)->Iterations(100)->Args({3, 40, 221, 193});

BENCHMARK_MAIN();

#endif /* USE_VULKAN_API */
