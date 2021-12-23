#ifdef USE_VULKAN_API

#include <benchmark/benchmark.h>

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>

namespace {

static void cat_op_channel_perf(benchmark::State& state) {
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
  const auto in_cpu3 = at::rand({batches, channels, height, width}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan1 = in_cpu1.vulkan();
  const auto in_vulkan2 = in_cpu2.vulkan();
  const auto in_vulkan3 = in_cpu3.vulkan();

  // Act
  while (state.KeepRunning()) {
    const auto out_vulkan = at::cat({in_vulkan1, in_vulkan2, in_vulkan3}, 1);

    // to avoid out-of-memory issues, release resources by waiting and flushing all GPU operations
    at::native::vulkan::api::context()->wait(out_vulkan);
    at::native::vulkan::api::context()->flush();
  }
}

static void CommonBenchmarkSettings(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMillisecond);
  b->ArgNames({"N", "C", "H", "W"});
}

} // namespace

BENCHMARK(cat_op_channel_perf)->Apply(CommonBenchmarkSettings)->Threads(1)->Iterations(1000)->Args({3, 40, 221, 193}); // big multiple of 4 channels
BENCHMARK(cat_op_channel_perf)->Apply(CommonBenchmarkSettings)->Threads(1)->Iterations(1000)->Args({3, 20, 221, 193}); // big multiple of 4 channels
BENCHMARK(cat_op_channel_perf)->Apply(CommonBenchmarkSettings)->Threads(1)->Iterations(1000)->Args({3, 39, 221, 193}); // big non-multiple of 4 channels
BENCHMARK(cat_op_channel_perf)->Apply(CommonBenchmarkSettings)->Threads(1)->Iterations(5000)->Args({3, 4, 221, 193}); // small multiple of 4 channels
BENCHMARK(cat_op_channel_perf)->Apply(CommonBenchmarkSettings)->Threads(1)->Iterations(5000)->Args({3, 3, 221, 193}); // small non-multiple of 4 channels
BENCHMARK(cat_op_channel_perf)->Apply(CommonBenchmarkSettings)->Threads(3)->Iterations(1000)->Args({3, 40, 221, 193}); // big multiple of 4 channels (multi-thread)
BENCHMARK_MAIN();

#endif /* USE_VULKAN_API */
