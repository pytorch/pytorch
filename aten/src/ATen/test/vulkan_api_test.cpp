#ifdef USE_VULKAN_API

#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

// TODO: These functions should move to a common place.

namespace {

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float tolerance = 1e-2;
#else
  constexpr float tolerance = 1e-5;
#endif

  return diff.abs().max().item<float>() <= (tolerance * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

void showRtol(const at::Tensor& a, const at::Tensor& b) {
  const auto diff = (a - b).abs();

  float maxValue = a.abs().max().item<float>();
  maxValue = fmax(b.abs().max().item<float>(), maxValue);

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float tolerance = 1e-2;
#else
  constexpr float tolerance = 1e-5;
#endif

  const float maxDiff = maxValue * tolerance;
  std::cout << "Max Diff allowed: " << maxDiff << std::endl;
  if (diff.sizes().size() == 2) {
    for (int y = 0; y < diff.sizes()[0]; y++) {
      std::cout << y << ":";
      for (int x = 0; x < diff.sizes()[1]; x++) {
        float diff_xy = diff[y][x].item<float>();
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
        }
        else {
          std::cout << std::setw(5) << " ";
        }
      }
      std::cout << std::endl;
    }
  }
}

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

} // namespace

namespace {

TEST(VulkanAPITest, conv2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
  //TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  constexpr struct {
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
  } input {1, 3, 8, 8};

  constexpr struct {
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
  } weights {1, input.channels, 3, 3};

  const auto input_cpu = at::randn(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::randn(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::randn({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

  const auto output_cpu = at::conv2d(
      input_cpu,
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups);

  const auto output_vulkan = at::conv2d(
      input_cpu.vulkan(),
      weights_cpu,
      bias_cpu,
      stride,
      padding,
      dilation,
      groups).cpu();

  const bool check = almostEqual(output_cpu, output_vulkan);
  if (!check) {
    showRtol(output_cpu, output_vulkan);
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mclaren_encoder_block) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 2};
  constexpr std::array<int64_t, 2u> padding{0, 0};
  constexpr std::array<int64_t, 2u> output_padding{0, 0};
  //TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  const auto input_1_cpu = at::randn({1,4,1,161}, at::device(at::kCPU).dtype(at::kFloat))*5;
  const auto input_2_cpu = at::randn({1,4,1,161}, at::device(at::kCPU).dtype(at::kFloat))*5;

  const auto weights_1_cpu = at::randn({32,4,2,3}, at::device(at::kCPU).dtype(at::kFloat))*2;
  const auto bias_1_cpu = at::randn({32}, at::device(at::kCPU).dtype(at::kFloat))*2;

  const auto weights_2_cpu = at::randn({32,4,2,3}, at::device(at::kCPU).dtype(at::kFloat))*2;
  const auto bias_2_cpu = at::randn({32}, at::device(at::kCPU).dtype(at::kFloat))*2;

  const auto input_cpu = at::cat({input_1_cpu, input_2_cpu}, 2);
  const auto output_1_cpu = at::conv2d(
      input_cpu,
      weights_1_cpu,
      bias_1_cpu,
      stride,
      padding,
      dilation,
      groups);
  const auto output_2_cpu = at::conv2d(
      input_cpu,
      weights_2_cpu,
      bias_2_cpu,
      stride,
      padding,
      dilation,
      groups);
  const auto output_cpu = output_1_cpu * at::sigmoid(output_2_cpu);

  auto prepack = callOpByName(
      "mclaren_prepack::mclaren_encoder_block_prepack",
      "",
      weights_1_cpu,
      bias_1_cpu,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      weights_2_cpu,
      bias_2_cpu,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      false);

  auto op_out_ivalues =
      callOpByName("mclaren_prepack::mclaren_encoder_block_run", "", input_1_cpu.vulkan(), input_2_cpu.vulkan(), prepack[0]);
  auto output_vulkan = op_out_ivalues[0].toTensor().cpu();

  const bool check = almostEqual(output_cpu, output_vulkan);

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mclaren_decoder_block) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 2};
  constexpr std::array<int64_t, 2u> padding{1, 0};
  constexpr std::array<int64_t, 2u> output_padding{0, 1};
  //TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  const auto input_1_cpu = at::randn({1,4,1,39}, at::device(at::kCPU).dtype(at::kFloat))*5;
  const auto input_2_cpu = at::randn({1,4,1,39}, at::device(at::kCPU).dtype(at::kFloat))*5;

  const auto weights_1_cpu = at::randint(50, {4,1,2,3}, at::device(at::kCPU).dtype(at::kFloat))*2;
  const auto bias_1_cpu = at::randint(50, {1}, at::device(at::kCPU).dtype(at::kFloat))*2;

  const auto weights_2_cpu = at::randn({4,1,2,3}, at::device(at::kCPU).dtype(at::kFloat))*2;
  const auto bias_2_cpu = at::randn({1}, at::device(at::kCPU).dtype(at::kFloat))*2;

  const auto input_cpu = at::cat({input_1_cpu, input_2_cpu}, 2);
  const auto output_1_cpu = at::conv_transpose2d(
      input_cpu,
      weights_1_cpu,
      bias_1_cpu,
      stride,
      padding,
      output_padding,
      groups,
      dilation);
  const auto output_2_cpu = at::conv_transpose2d(
      input_cpu,
      weights_2_cpu,
      bias_2_cpu,
      stride,
      padding,
      output_padding,
      groups,
      dilation);
  const auto output_cpu = output_1_cpu * at::sigmoid(output_2_cpu);
  //const auto output_cpu = output_2_cpu;

  auto prepack = callOpByName(
      "mclaren_prepack::mclaren_encoder_block_prepack",
      "",
      weights_1_cpu,
      bias_1_cpu,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      weights_2_cpu,
      bias_2_cpu,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      true);

  auto op_out_ivalues =
      callOpByName("mclaren_prepack::mclaren_encoder_block_run", "", input_cpu.vulkan(), input_2_cpu.vulkan(), prepack[0]);
  auto output_vulkan = op_out_ivalues[0].toTensor().cpu();

  const bool check = almostEqual(output_cpu, output_vulkan);

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, transpose_conv2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 2};
  constexpr std::array<int64_t, 2u> padding{1, 0};
  constexpr std::array<int64_t, 2u> output_padding{0, 1};
  //TODO: Support conv2d with dilation != 1
  constexpr std::array<int64_t, 2u> dilation{1, 1};

  const auto input_1_cpu = at::randn({1,16,1,39}, at::device(at::kCPU).dtype(at::kFloat))*5;
  const auto input_2_cpu = at::randn({1,16,1,39}, at::device(at::kCPU).dtype(at::kFloat))*5;

  const auto weights_1_cpu = at::randn({16,8,2,3}, at::device(at::kCPU).dtype(at::kFloat))*2;
  const auto bias_1_cpu = at::randn({8}, at::device(at::kCPU).dtype(at::kFloat))*2;

  const auto weights_2_cpu = at::randn({16,8,2,3}, at::device(at::kCPU).dtype(at::kFloat))*2;
  const auto bias_2_cpu = at::randn({8}, at::device(at::kCPU).dtype(at::kFloat))*2;

  const auto input_cpu = at::cat({input_1_cpu, input_2_cpu}, 2);
  const auto output_cpu = at::conv_transpose2d(
      input_cpu,
      weights_1_cpu,
      bias_1_cpu,
      stride,
      padding,
      output_padding,
      groups,
      dilation);

  auto output_vulkan = at::conv_transpose2d(
      input_cpu.vulkan(),
      weights_1_cpu,
      bias_1_cpu,
      stride,
      padding,
      output_padding,
      groups,
      dilation).cpu();

  const bool check = almostEqual(output_cpu, output_vulkan);

  ASSERT_TRUE(check);
}

} // namespace

#endif /* USE_VULKAN_API */
