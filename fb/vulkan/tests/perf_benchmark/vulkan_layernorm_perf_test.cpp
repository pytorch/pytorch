#include <unordered_map>
#ifdef USE_VULKAN_API

#include "vulkan_perf_utils.h"

namespace {

static void CommonMMBenchmarkSettings(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMillisecond);
  b->ArgNames({"N", "M", "P"});
}

static void layer_norm_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  c10::InferenceMode mode;

  // Arrange
  const auto c = state.range(0);
  const auto h = state.range(1);
  const auto w = state.range(2);

  const auto in_cpu =
      at::rand({c, h, w}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const auto weight_cpu =
      at::rand({c, h, w}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight_vulkan = weight_cpu.vulkan();

  const auto bias_cpu =
      at::rand({c, h, w}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_vulkan = bias_cpu.vulkan();

  // Act
  for (auto _ : state) {
    const auto vulkan_out =
        at::layer_norm(
            in_vulkan, {c, h, w}, weight_vulkan, bias_vulkan, 1e-05, false)
            .cpu();
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  extractTotalShaderResultsAndSetState(state);
  at::native::vulkan::api::context()->querypool().print_results();
#endif
}

} // namespace

const uint32_t BENCHMARK_MM_N = 75;
const uint32_t BENCHMARK_MM_M = 75;
const uint32_t BENCHMARK_MM_P = 75;
const uint32_t BENCHMARK_ITERATIONS = 50;

BENCHMARK(layer_norm_benchmark)
    ->Apply(CommonMMBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(BENCHMARK_ITERATIONS)
    ->Args({BENCHMARK_MM_N, BENCHMARK_MM_M, BENCHMARK_MM_P});

BENCHMARK_MAIN();

#endif /* USE_VULKAN_API */
