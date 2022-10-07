#ifdef USE_VULKAN_API

#include <benchmark/benchmark.h>

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Factory.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>

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

template <typename dest_t, typename src_t>
static inline dest_t safe_downcast(src_t v) {
  TORCH_CHECK(
      std::numeric_limits<dest_t>::min() <= v &&
          v <= std::numeric_limits<dest_t>::max(),
      "integer out of range");

  return static_cast<dest_t>(v);
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
  at::native::vulkan::api::context()->enable_op_profiling();
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
  at::native::vulkan::api::context()->enable_op_profiling();
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

static void conv2d_op_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches_in = safe_downcast<uint32_t>(state.range(0));
  const auto channels_in = safe_downcast<uint32_t>(state.range(1));
  const auto height_in = safe_downcast<uint32_t>(state.range(2));
  const auto width_in = safe_downcast<uint32_t>(state.range(3));
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
  // TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{batches_in, channels_in, height_in, width_in};

  struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{1, input.channels, 3, 3};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_out = at::conv2d(
                                input_cpu.vulkan(),
                                weights_cpu,
                                bias_cpu,
                                stride,
                                padding,
                                dilation,
                                groups)
                                .cpu();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("conv2d") / 1000000.0);
#endif
}

static void conv2d_op_q_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches_in = safe_downcast<uint32_t>(state.range(0));
  const auto channels_in = safe_downcast<uint32_t>(state.range(1));
  const auto height_in = safe_downcast<uint32_t>(state.range(2));
  const auto width_in = safe_downcast<uint32_t>(state.range(3));
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
  // TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{batches_in, channels_in, height_in, width_in};

  struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{1, input.channels, 3, 3};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const double w_scale = 0.1;
  const int w_zero_point = 10;

  const double b_scale = 0.1;
  const int b_zero_point = 10;

  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  const auto in_vulkan1 = input_cpu.vulkan();
  const double scale = 0.1;
  const int zero_point = 10;
  const auto out_vulkan1 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan1, scale, zero_point, c10::ScalarType::QUInt8);

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  const double scale2 = 0.15;
  const int zero_point2 = 15;
  const auto shape_match =
      at::rand({1, 1, 64, 199}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_conv2d = at::native::vulkan::ops::quantized_conv2d(
        out_vulkan1,
        weight_q,
        bias_q,
        stride,
        padding,
        dilation,
        groups,
        scale2,
        zero_point2);
    const auto vulkan_out = vulkan_to_cpu(vulkan_conv2d, shape_match);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("quantized_conv2d") / 1000000.0);
#endif
}

static void conv2dpw_op_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches_in = safe_downcast<uint32_t>(state.range(0));
  const auto channels_in = safe_downcast<uint32_t>(state.range(1));
  const auto height_in = safe_downcast<uint32_t>(state.range(2));
  const auto width_in = safe_downcast<uint32_t>(state.range(3));
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 1};
  constexpr std::array<int64_t, 2u> padding{0, 0};
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{batches_in, channels_in, height_in, width_in};

  struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{29, input.channels, 1, 1};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_out = at::conv2d(
                                input_cpu.vulkan(),
                                weights_cpu,
                                bias_cpu,
                                stride,
                                padding,
                                dilation,
                                groups)
                                .cpu();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("conv2d_pw_2x2") / 1000000.0);
#endif
}

static void conv2dpw_op_q_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches_in = safe_downcast<uint32_t>(state.range(0));
  const auto channels_in = safe_downcast<uint32_t>(state.range(1));
  const auto height_in = safe_downcast<uint32_t>(state.range(2));
  const auto width_in = safe_downcast<uint32_t>(state.range(3));
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 1};
  constexpr std::array<int64_t, 2u> padding{0, 0};
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{batches_in, channels_in, height_in, width_in};

  struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{29, input.channels, 1, 1};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const double w_scale = 0.1;
  const int w_zero_point = 10;

  const double b_scale = 0.1;
  const int b_zero_point = 10;

  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  const auto in_vulkan1 = input_cpu.vulkan();
  const double scale = 0.1;
  const int zero_point = 10;
  const auto out_vulkan1 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan1, scale, zero_point, c10::ScalarType::QUInt8);

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  const double scale2 = 0.15;
  const int zero_point2 = 15;
  const auto shape_match =
      at::rand({1, 29, 127, 397}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_conv2d = at::native::vulkan::ops::quantized_conv2d(
        out_vulkan1,
        weight_q,
        bias_q,
        stride,
        padding,
        dilation,
        groups,
        scale2,
        zero_point2);
    const auto vulkan_out = vulkan_to_cpu(vulkan_conv2d, shape_match);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("quantized_conv2d_pw_2x2") / 1000000.0);
#endif
}

static void conv2ddw_op_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches_in = safe_downcast<uint32_t>(state.range(0));
  const auto height_in = safe_downcast<uint32_t>(state.range(2));
  const auto width_in = safe_downcast<uint32_t>(state.range(3));
  constexpr int64_t groups = 7;
  constexpr std::array<int64_t, 2u> stride{2, 3};
  constexpr std::array<int64_t, 2u> padding{0, 4};
  constexpr std::array<int64_t, 2u> dilation{3, 1};

  struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{batches_in, groups, height_in, width_in};

  struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{groups, 1, 17, 7};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_out = at::conv2d(
                                input_cpu.vulkan(),
                                weights_cpu,
                                bias_cpu,
                                stride,
                                padding,
                                dilation,
                                groups)
                                .cpu();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("conv2d_dw") / 1000000.0);
#endif
}

static void conv2ddw_op_q_benchmark(benchmark::State& state) {
  // Guard
  if (!at::is_vulkan_available()) {
    return;
  }

  // Arrange
  const auto batches_in = safe_downcast<uint32_t>(state.range(0));
  const auto height_in = safe_downcast<uint32_t>(state.range(2));
  const auto width_in = safe_downcast<uint32_t>(state.range(3));
  constexpr int64_t groups = 7;
  constexpr std::array<int64_t, 2u> stride{2, 3};
  constexpr std::array<int64_t, 2u> padding{0, 4};
  constexpr std::array<int64_t, 2u> dilation{3, 1};

  struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          batches,
          channels,
          width,
          height,
      };
    }
  } input{batches_in, groups, height_in, width_in};

  struct {
    uint32_t output_channels;
    uint32_t input_channels;
    uint32_t width;
    uint32_t height;

    std::array<int64_t, 4u> size() const {
      return {
          output_channels,
          input_channels,
          width,
          height,
      };
    }
  } weights{groups, 1, 17, 7};

  const auto input_cpu =
      at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu =
      at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn(
      {weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const double w_scale = 0.1;
  const int w_zero_point = 10;

  const double b_scale = 0.1;
  const int b_zero_point = 10;

  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  const auto in_vulkan1 = input_cpu.vulkan();
  const double scale = 0.1;
  const int zero_point = 10;
  const auto out_vulkan1 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan1, scale, zero_point, c10::ScalarType::QUInt8);

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  const double scale2 = 0.15;
  const int zero_point2 = 15;
  const auto shape_match =
      at::rand({1, 7, 45, 67}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_conv2d = at::native::vulkan::ops::quantized_conv2d(
        out_vulkan1,
        weight_q,
        bias_q,
        stride,
        padding,
        dilation,
        groups,
        scale2,
        zero_point2);
    const auto vulkan_out = vulkan_to_cpu(vulkan_conv2d, shape_match);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("quantized_conv2d_dw") / 1000000.0);
#endif
}

static void sub_op_benchmark(benchmark::State& state) {
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
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_out = at::sub(in_vulkan1, in_vulkan2).cpu();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("sub") / 1000000.0);
#endif
}

static void sub_op_q_benchmark(benchmark::State& state) {
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
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  const double scale2 = 0.15;
  const int zero_point2 = 15;
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_sub = at::native::vulkan::ops::quantized_sub(
        out_vulkan1, out_vulkan2, scale2, zero_point2);
    const auto vulkan_out = vulkan_to_cpu(vulkan_sub, out_cpu1);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("quantized_sub") / 1000000.0);
#endif
}

static void mul_op_benchmark(benchmark::State& state) {
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
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_out = at::mul(in_vulkan1, in_vulkan2).cpu();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("mul") / 1000000.0);
#endif
}

static void mul_op_q_benchmark(benchmark::State& state) {
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
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  const double scale2 = 0.15;
  const int zero_point2 = 15;
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_mul = at::native::vulkan::ops::quantized_mul(
        out_vulkan1, out_vulkan2, scale2, zero_point2);
    const auto vulkan_out = vulkan_to_cpu(vulkan_mul, out_cpu1);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("quantized_mul") / 1000000.0);
#endif
}

static void div_op_benchmark(benchmark::State& state) {
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
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_out = at::div(in_vulkan1, in_vulkan2).cpu();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("div") / 1000000.0);
#endif
}

static void div_op_q_benchmark(benchmark::State& state) {
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
  at::native::vulkan::api::context()->enable_op_profiling();
  at::native::vulkan::api::context()->reset_querypool();
#endif

  // Act
  const double scale2 = 0.15;
  const int zero_point2 = 15;
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto vulkan_div = at::native::vulkan::ops::quantized_div(
        out_vulkan1, out_vulkan2, scale2, zero_point2);
    const auto vulkan_out = vulkan_to_cpu(vulkan_div, out_cpu1);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed.count());
  }

#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
  at::native::vulkan::api::context()->querypool().extract_results();
  at::native::vulkan::api::context()->querypool().print_results();
  state.SetIterationTime(at::native::vulkan::api::context()->querypool().get_total_op_ns("quantized_div") / 1000000.0);
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
BENCHMARK(conv2d_op_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({1, 17, 127, 397});
BENCHMARK(conv2d_op_q_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({1, 17, 127, 397});
BENCHMARK(conv2dpw_op_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({1, 17, 127, 397});
BENCHMARK(conv2dpw_op_q_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({1, 17, 127, 397});
BENCHMARK(conv2ddw_op_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(10)
    ->Args({1, 7, 137, 199});
BENCHMARK(conv2ddw_op_q_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(10)
    ->Args({1, 7, 137, 199});
BENCHMARK(sub_op_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});
BENCHMARK(sub_op_q_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});
BENCHMARK(mul_op_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});
BENCHMARK(mul_op_q_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});
BENCHMARK(div_op_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});
BENCHMARK(div_op_q_benchmark)
    ->Apply(CommonBenchmarkSettings)
    ->UseManualTime()
    ->Threads(1)
    ->Iterations(100)
    ->Args({3, 40, 221, 193});

BENCHMARK_MAIN();

#endif /* USE_VULKAN_API */
