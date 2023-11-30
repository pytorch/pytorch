#include <unordered_map>
#ifdef USE_VULKAN_API

#include <benchmark/benchmark.h>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Factory.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>

namespace {

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
static const float NANOSECONDS_IN_SECOND = 1000000000.0;
#endif

template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

template <class... Args>
inline std::vector<c10::IValue> callOpByHandle(
    const c10::OperatorHandle& op,
    Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  c10::Dispatcher::singleton().callBoxed(op, &stack);
  return stack;
}

template <class... Args>
inline std::vector<c10::IValue> callOpByName(
    const char* func_name,
    const char* overload_name,
    Args... args) {
  const c10::optional<c10::OperatorHandle> op_handle =
      c10::Dispatcher::singleton().findSchema({func_name, overload_name});
  assert(op_handle.has_value());
  return callOpByHandle(op_handle.value(), std::forward<Args>(args)...);
}

static void CommonMMBenchmarkSettings(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMillisecond);
  b->ArgNames({"N", "M", "P"});
}
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
// This function aggregate the latency of all invoked shaders except
// `vulkan.nchw_to_image` and `vulkan.image_to_nchw`, which are moving data
// between CPU and GPU memory.
static void extractTotalShaderResultsAndSetState(benchmark::State& state) {
  at::native::vulkan::api::context()->querypool().extract_results();

  uint64_t sum_shader_latency_in_nanoseconds = 0;
  auto result_aggregator =
      [&sum_shader_latency_in_nanoseconds](
          const at::native::vulkan::api::ShaderDuration& s) {
        if (s.kernel_name != "vulkan.nchw_to_image" &&
            s.kernel_name != "vulkan.image_to_nchw") {
          sum_shader_latency_in_nanoseconds += s.execution_duration_ns;
        }
      };
  at::native::vulkan::api::context()->querypool().shader_log_for_each(
      result_aggregator);

  float sum_shader_latency_in_seconds =
      sum_shader_latency_in_nanoseconds / NANOSECONDS_IN_SECOND;
  state.SetIterationTime(sum_shader_latency_in_seconds);
}
#endif

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
