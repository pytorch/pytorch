#ifdef USE_VULKAN_API

#include <benchmark/benchmark.h>

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Factory.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>

namespace {

at::Tensor vulkan_to_cpu(at::Tensor vulkan, at::Tensor in_cpu) {
  auto q_options = in_cpu.options();
  if (q_options.dtype().toScalarType() == c10::ScalarType::QUInt8) {
    auto output = at::native::empty_affine_quantized(
        in_cpu.sizes(),
        q_options.dtype().toScalarType(),
        q_options.layout(),
        q_options.device(),
        q_options.pinned_memory(),
        in_cpu.q_scale(),
        in_cpu.q_zero_point());
    at::native::vulkan::ops::copy_(output, vulkan);
    return output;
  } else {
    auto output = at::empty(in_cpu.sizes(), q_options);
    at::native::vulkan::ops::copy_(output, vulkan);
    return output;
  }
}

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
  const auto in_cpu1 = at::rand(
      {batches, channels, height, width},
      at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand(
      {batches, channels, height, width},
      at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan1 = in_cpu1.vulkan();
  const auto in_vulkan2 = in_cpu2.vulkan();

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_out = at::add(in_vulkan1, in_vulkan2).cpu();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("add") / 1000000.0);
#endif
}

static void add_op_q_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches = state.range(0);
  const auto channels = state.range(1);
  const auto height = state.range(2);
  const auto width = state.range(3);
  const auto in_cpu1 = at::rand(
      {batches, channels, height, width},
      at::device(at::kCPU).dtype(at::kFloat));
  const auto in_cpu2 = at::rand(
      {batches, channels, height, width},
      at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan1 = in_cpu1.vulkan();
  const auto in_vulkan2 = in_cpu2.vulkan();
  const double scale = 0.1;
  const int zero_point = 10;
  const auto out_cpu1 = at::quantize_per_tensor(
      in_cpu1, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan1 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan1, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  const double scale2 = 0.15;
  const int zero_point2 = 15;
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_add = at::native::vulkan::ops::quantized_add(
        out_vulkan1, out_vulkan2, scale2, zero_point2);
    const auto vulkan_out = vulkan_to_cpu(vulkan_add, out_cpu1);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("quantized_add") / 1000000.0);
#endif
}

static void CommonBenchmarkSettings(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMillisecond);
  b->ArgNames({"N", "C", "H", "W"});
}

} // namespace

BENCHMARK(add_op_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});
BENCHMARK(add_op_q_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});

BENCHMARK_MAIN();

#endif /* USE_VULKAN_API */
