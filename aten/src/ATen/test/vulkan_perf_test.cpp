#ifdef USE_VULKAN_API

#include <benchmark/benchmark.h>

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>

namespace {

// using Vulkan Timestamp Queries for the pure GPU execution time only
static void cat_op_channel_perf_gpu_only(benchmark::State& state) {
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
  for (auto _ : state) {
    at::native::vulkan::api::context()->querypool().enable();
    const auto vulkan_out = at::cat({in_vulkan1, in_vulkan2, in_vulkan3}, 1);
    vulkan_out.cpu();
    auto perf_info = at::native::vulkan::api::context()->querypool().disable(true);
    state.SetIterationTime(perf_info[0].execution_time_us / 1'000'000.); // us to sec
  }
}

static void gru_op_perf(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const int H_in = static_cast<int>(state.range(0));  // input_size
  const int H_out = static_cast<int>(state.range(1)); // hidden_size
  const int num_layers = static_cast<int>(state.range(2));
  const double gru_dropout = .0;
  const bool has_biases = true;
  const bool train = false;
  const bool bidirectional = false;
  const bool batch_first = true;
  const auto in_cpu = at::rand({1, 1, H_in}, at::device(at::kCPU).dtype(at::kFloat));
  const auto h0_cpu = at::rand({num_layers, 1, H_out}, at::device(at::kCPU).dtype(at::kFloat));

  c10::List<at::Tensor> weight_ih_l; // shape (3 * hidden_size, input_size)
  c10::List<at::Tensor> weight_hh_l; // shape (3 * hidden_size, hidden_size)
  c10::List<at::Tensor> bias_ih_l;   // shape (3 * hidden_size)
  c10::List<at::Tensor> bias_hh_l;   // shape (3 * hidden_size)
  for (int i = 0; i < num_layers; ++i) {
    weight_ih_l.emplace_back(at::rand({3 * H_out, H_in}, at::device(at::kCPU).dtype(at::kFloat)));
    weight_hh_l.emplace_back(at::rand({3 * H_out, H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_ih_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
    bias_hh_l.emplace_back(at::rand({3 * H_out}, at::device(at::kCPU).dtype(at::kFloat)));
  }

  // put this guard here to run inference inststead of training
  // to avoid the following error:
  //     C++ exception with description "0INTERNAL ASSERT FAILED at "xplat/caffe2/aten/src/ATen/core/boxing/KernelFunction.cpp":31, please report a bug to PyTorch. aten::gru.input has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering (see Note [Ambiguity in AutogradOther kernel]). If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated Autograd dispatch key for the backend.
  //     If you only want to run inference instead of training, add `c10::InferenceMode mode;` before model.forward(). Note this guard is only available in C++ but not Python at present.
  c10::InferenceMode mode;

  // Act
  while (state.KeepRunning()) {
    // weights/biases should be always on CPU.
    const auto out_vulkan = at::gru(in_cpu.vulkan(), h0_cpu.vulkan(), { weight_ih_l.get(0), weight_hh_l.get(0), bias_ih_l.get(0), bias_hh_l.get(0),
      weight_ih_l.get(1), weight_hh_l.get(1), bias_ih_l.get(1), bias_hh_l.get(1) },
      has_biases, num_layers, gru_dropout, train, bidirectional, batch_first);

    auto vulkan_output = std::get<0>(out_vulkan);
    auto vulkan_hidden = std::get<1>(out_vulkan);

    // to avoid out-of-memory issues, release resources by waiting and flushing all GPU operations
    at::native::vulkan::api::context()->wait(vulkan_output);
    at::native::vulkan::api::context()->wait(vulkan_hidden);
    at::native::vulkan::api::context()->flush();
  }
}

static void CommonBenchmarkSettings(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMillisecond);
  b->ArgNames({"N", "C", "H", "W"});
}

} // namespace

BENCHMARK(cat_op_channel_perf_gpu_only)->Apply(CommonBenchmarkSettings)->UseManualTime()->Threads(1)->Iterations(100)->Args({3, 40, 221, 193});  // big multiple of 4 channels
BENCHMARK(cat_op_channel_perf_gpu_only)->Apply(CommonBenchmarkSettings)->UseManualTime()->Threads(1)->Iterations(100)->Args({3, 20, 221, 193});  // big multiple of 4 channels
BENCHMARK(cat_op_channel_perf_gpu_only)->Apply(CommonBenchmarkSettings)->UseManualTime()->Threads(1)->Iterations(100)->Args({3, 39, 221, 193});  // big non-multiple of 4 channels
BENCHMARK(cat_op_channel_perf_gpu_only)->Apply(CommonBenchmarkSettings)->UseManualTime()->Threads(1)->Iterations(100)->Args({3, 4, 221, 193});   // small multiple of 4 channels
BENCHMARK(cat_op_channel_perf_gpu_only)->Apply(CommonBenchmarkSettings)->UseManualTime()->Threads(1)->Iterations(100)->Args({3, 3, 221, 193});   // small non-multiple of 4 channels
BENCHMARK(cat_op_channel_perf_gpu_only)->Apply(CommonBenchmarkSettings)->UseManualTime()->Threads(3)->Iterations(100)->Args({3, 40, 221, 193});  // big multiple of 4 channels (multi-thread)
BENCHMARK(gru_op_perf)->Apply(CommonBenchmarkSettings)->Threads(1)->Iterations(100)->Args({384, 384, 2});                                        // McLaren Model inputs

BENCHMARK_MAIN();

#endif /* USE_VULKAN_API */
