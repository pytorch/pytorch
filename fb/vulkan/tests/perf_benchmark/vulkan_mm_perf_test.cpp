#include <unordered_map>
#ifdef USE_VULKAN_API

#include "vulkan_perf_utils.h"

namespace {

static void CommonMMBenchmarkSettings(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMillisecond);
  b->ArgNames({"N", "M", "P"});
}

static void mm_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Arrange
  const auto n = state.range(0);
  const auto m = state.range(1);
  const auto p = state.range(2);

  const auto in_cpu1 = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan1 = in_cpu1.vulkan();
  // Act
  for (auto _ : state) {
    const auto vulkan_out = in_vulkan1.mm(in_cpu2).cpu();
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  extractTotalOpResultsAndSetState(state, "vulkan.mm");
  at::native::vulkan::api::context()->querypool().print_results();
#endif
}

static void addmm_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Arrange
  const auto n = state.range(0);
  const auto m = state.range(1);
  const auto p = state.range(2);

  const auto in_cpu1 = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan1 = in_cpu1.vulkan();
  const auto in_vulkan2 = in_cpu2.vulkan();

  const auto bias_vk =
      at::zeros({n, p}, at::device(at::kCPU).dtype(at::kFloat)).vulkan();

  // Act
  for (auto _ : state) {
    const auto vulkan_out =
        at::addmm(bias_vk, in_vulkan1, in_vulkan2, 1.0, 1.0).cpu();
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  extractTotalShaderResultsAndSetState(state);
  at::native::vulkan::api::context()->querypool().print_results();
#endif
}

static void create_linear_context_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Arrange
  const auto n = state.range(0);
  const auto m = state.range(1);
  const auto p = state.range(2);

  const auto weight = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));

  // Act
  for (auto _ : state) {
    auto prepack =
        callOpByName("vulkan_prepack::create_linear_context", "", weight, bias);

    const auto dummy = at::zeros({1}).vulkan();
    dummy.cpu(); // force sync?
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  float total_op_time =
      at::native::vulkan::api::context()->querypool().get_total_op_ns(
          "vulkan.nchw_to_image") /
      NANOSECONDS_IN_SECOND;
  total_op_time +=
      at::native::vulkan::api::context()->querypool().get_total_op_ns(
          "vulkan.image_to_nchw") /
      NANOSECONDS_IN_SECOND;
  total_op_time +=
      at::native::vulkan::api::context()->querypool().get_total_op_ns(
          "vulkan.convert_channels_to_height_packed") /
      NANOSECONDS_IN_SECOND;

  state.SetIterationTime(total_op_time);

  at::native::vulkan::api::context()->querypool().print_results();
#endif
}

static void run_linear_context_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Arrange
  const auto n = state.range(0);
  const auto m = state.range(1);
  const auto p = state.range(2);

  const auto input_cpu =
      at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));
  const auto weight = at::rand({p, m}, at::device(at::kCPU).dtype(at::kFloat));
  const auto bias = at::rand({n, p}, at::device(at::kCPU).dtype(at::kFloat));

  const auto prepack =
      callOpByName("vulkan_prepack::create_linear_context", "", weight, bias);

  // Act
  for (auto _ : state) {
    auto vulkan_output = callOpByName(
        "vulkan_prepack::run_linear_context",
        "",
        input_cpu.vulkan(),
        prepack[0]);
    auto out_vulkan = vulkan_output[0].toTensor().cpu(); // force sync?
  }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  extractTotalShaderResultsAndSetState(state);
  at::native::vulkan::api::context()->querypool().print_results();
#endif
}

} // namespace

const uint32_t BENCHMARK_MM_N = 500;
const uint32_t BENCHMARK_MM_M = 500;
const uint32_t BENCHMARK_MM_P = 500;
const uint32_t BENCHMARK_ITERATIONS = 5;
BENCHMARK(mm_benchmark)
    ->Apply(CommonMMBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(BENCHMARK_ITERATIONS)
    ->Args({BENCHMARK_MM_N, BENCHMARK_MM_M, BENCHMARK_MM_P});

BENCHMARK(addmm_benchmark)
    ->Apply(CommonMMBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(BENCHMARK_ITERATIONS)
    ->Args({BENCHMARK_MM_N, BENCHMARK_MM_M, BENCHMARK_MM_P});

BENCHMARK(create_linear_context_benchmark)
    ->Apply(CommonMMBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(BENCHMARK_ITERATIONS)
    ->Args({BENCHMARK_MM_N, BENCHMARK_MM_M, BENCHMARK_MM_P});

BENCHMARK(run_linear_context_benchmark)
    ->Apply(CommonMMBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(BENCHMARK_ITERATIONS)
    ->Args({BENCHMARK_MM_N, BENCHMARK_MM_M, BENCHMARK_MM_P});

BENCHMARK_MAIN();

#endif /* USE_VULKAN_API */
