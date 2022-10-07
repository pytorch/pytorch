#ifdef USE_VULKAN_API

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/vulkan/api/api.h>
#include <gtest/gtest.h>

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Factory.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>

#include <c10/util/irange.h>

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

/* Unused function
bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.0f;
}
*/

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
    for (const auto y : c10::irange(diff.sizes()[0])) {
      std::cout << y << ":";
      for (const auto x : c10::irange(diff.sizes()[1])) {
        float diff_xy = diff[y][x].item<float>();
        if (diff_xy > maxDiff) {
          std::cout << std::setw(5) << x;
        } else {
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

class VulkanAPITest : public ::testing::Test {
 public:
  void SetUp() {
    if (!at::is_vulkan_available()) {
      GTEST_SKIP() << "Vulkan is not available";
    }
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    at::native::vulkan::api::context()->reset_querypool();
#endif
  }

  void TearDown() {
#if defined(USE_VULKAN_GPU_DIAGNOSTICS) && defined(__ANDROID__)
    try {
      at::native::vulkan::api::context()->querypool().extract_results();
      at::native::vulkan::api::context()->querypool().print_results();
    } catch (const std::exception& e) {
      std::cout << "Could not get querypool results!"
                << " Reason: " << e.what() << std::endl;
    }
#endif
  }
};

at::Tensor cpu_to_vulkan(at::Tensor in_cpu) {
  auto options = in_cpu.options();
  if (options.dtype().toScalarType() == c10::ScalarType::QUInt8) {
    auto ret = at::native::vulkan::ops::_empty_affine_quantized(
        in_cpu.sizes(),
        c10::ScalarType::QUInt8,
        options.layout(),
        options.device(),
        options.pinned_memory(),
        in_cpu.q_scale(),
        in_cpu.q_zero_point(),
        c10::MemoryFormat::Contiguous);
    at::native::vulkan::ops::copy_(ret, in_cpu);
    return ret;
  } else {
    auto ret = at::empty(in_cpu.sizes(), options);
    at::native::vulkan::ops::copy_(ret, in_cpu);
    return ret;
  }
}

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

TEST_F(VulkanAPITest, support_vulkan) {
  const double scale = 0.1;
  const int64_t zero_point = 10;

  auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 12 -
      6;
  auto in_cpu_quantized = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);

  auto in_vulkan_quantized = cpu_to_vulkan(in_cpu_quantized);
  at::native::vulkan::api::PipelineBarrier pipeline_barrier{};
  at::native::vulkan::ops::vTensor& v_self =
      at::native::vulkan::ops::convert(in_vulkan_quantized);
  if (in_cpu.dtype() == c10::kQUInt8) {
    v_self.image(
        pipeline_barrier,
        at::native::vulkan::api::PipelineStage::COMPUTE,
        at::native::vulkan::api::MemoryAccessType::READ);
    v_self.image(
        pipeline_barrier,
        at::native::vulkan::api::PipelineStage::COMPUTE,
        at::native::vulkan::api::MemoryAccessType::WRITE);
  }
  auto output = vulkan_to_cpu(in_vulkan_quantized, in_cpu_quantized);
  const auto check = almostEqual(
      at::native::int_repr_quantized_cpu(in_cpu_quantized),
      at::native::int_repr_quantized_cpu(output));

  if (!check) {
    showRtol(
        at::native::int_repr_quantized_cpu(in_cpu_quantized),
        at::native::int_repr_quantized_cpu(output));
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantize_per_tensor) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  auto output_for_quantized_vulkan = vulkan_to_cpu(out_vulkan, out_cpu);

  int rtol = 1;
  const auto check = at::allclose(
      at::native::int_repr_quantized_cpu(out_cpu),
      at::native::int_repr_quantized_cpu(output_for_quantized_vulkan),
      rtol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantize_dequantize) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;
  // quantize tensors
  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  // dequantize tensors
  const auto out_cpu_deq = at::dequantize(out_cpu);
  const auto out_vulkan_deq = at::native::vulkan::ops::dequantize(out_vulkan);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu);

  float rtol = 1;
  float atol = 0.5;
  const auto check =
      at::allclose(in_cpu, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);

  const auto check_two =
      at::allclose(out_cpu_deq, output_for_dequantized_vulkan, rtol, atol);

  if (!check_two) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check_two);
}

TEST_F(VulkanAPITest, quantized_add) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_added_tensors = callOpByName(
      "quantized::add",
      "",
      out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  float rtol = 0;
  float atol = 0.5;
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()), output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_add_broadcast) {
  const auto in_cpu =
      at::rand({2, 13, 1, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({2, 13, 32, 1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_added_tensors = callOpByName(
      "quantized::add",
      "",
      out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto in_cpu3 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

  float rtol = 0;
  float atol = 0.5;
   const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()), output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_add_broadcast1) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu =
      at::rand({2, 12, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({12, 1, 1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_added_tensors = callOpByName(
      "quantized::add",
      "",
      out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto in_cpu3 =
      at::rand({2, 12, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

  float rtol = 0;
  float atol = 0.5;
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()), output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_add_broadcast2) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu =
      at::rand({32, 1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({1, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_added_tensors = callOpByName(
      "quantized::add",
      "",
      out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto in_cpu3 =
      at::rand({32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

  float rtol = 0;
  float atol = 0.5;
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()), output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}


TEST_F(VulkanAPITest, quantized_add_broadcast3) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu =
      at::rand({32, 24}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({1}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_added_tensors = callOpByName(
      "quantized::add",
      "",
      out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto in_cpu3 =
      at::rand({32, 24}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu3);

  float rtol = 0;
  float atol = 0.5;
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()), output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_add_dif_params) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();
  const double scale = 0.1;
  const int zero_point = 10;
  const double scale2 = 0.2;
  const int zero_point2 = 20;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale2, zero_point2, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale2, zero_point2, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_added_tensors = callOpByName(
      "quantized::add",
      "",
      out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_added_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  float rtol = 0;
  float atol = 0.5;
  const auto check = at::allclose(
      at::dequantize(reg_added_tensors[0].toTensor()), output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d) {
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{2, 2};
  constexpr std::array<int64_t, 2u> padding{1, 1};
  // TODO: Support conv2d with dilation != 1
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
  } input{1, 3, 8, 8};

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
  } weights{1, input.channels, 3, 3};

  float r1 = 0.1;
  float r2 = 0.7;
  const auto input_cpu = (r1 - r2) *
          at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto weights_cpu = (r1 - r2) *
          at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto bias_cpu = (r1 - r2) *
          at::rand({weights.output_channels},
                   at::device(at::kCPU).dtype(at::kFloat)) +
      r2;

  const double w_scale = 0.1;
  const int w_zero_point = 10;

  const double b_scale = 0.1;
  const int b_zero_point = 10;

  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  const double scale = 0.10;
  const int zero_point = 10;
  const auto shape_match =
      at::rand({1, 1, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = input_cpu.vulkan();
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale2 = 0.15;
  const int zero_point2 = 15;
  const auto output_vulkan = at::native::vulkan::ops::quantized_conv2d(
      out_vulkan,
      weight_q,
      bias_q,
      stride,
      padding,
      dilation,
      groups,
      scale2,
      zero_point2);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(output_vulkan);
  auto output_for_dequantized_vulkan =
      vulkan_to_cpu(out_vulkan_deq, shape_match);

  float rtol = 0;
  float atol = 1.5;
  const auto check =
      at::allclose(output_cpu, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_pw) {
  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 1};
  constexpr std::array<int64_t, 2u> padding{0, 0};
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
  } input{1, 17, 127, 397};

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
  } weights{29, input.channels, 1, 1};

  float r1 = 0.1;
  float r2 = 0.7;
  const auto input_cpu = (r1 - r2) *
          at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto weights_cpu = (r1 - r2) *
          at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto bias_cpu = (r1 - r2) *
          at::rand({weights.output_channels},
                   at::device(at::kCPU).dtype(at::kFloat)) +
      r2;

  const double w_scale = 0.1;
  const int w_zero_point = 10;

  const double b_scale = 0.1;
  const int b_zero_point = 10;

  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  const double scale = 0.10;
  const int zero_point = 10;
  const auto shape_match =
      at::rand({1, 29, 127, 397}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = input_cpu.vulkan();
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale2 = 0.15;
  const int zero_point2 = 15;
  const auto output_vulkan = at::native::vulkan::ops::quantized_conv2d(
      out_vulkan,
      weight_q,
      bias_q,
      stride,
      padding,
      dilation,
      groups,
      scale2,
      zero_point2);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(output_vulkan);
  auto output_for_dequantized_vulkan =
      vulkan_to_cpu(out_vulkan_deq, shape_match);

  float rtol = 0;
  float atol = 1.5;
  const auto check =
      at::allclose(output_cpu, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, conv2d_dw) {
  constexpr int64_t groups = 7;
  constexpr std::array<int64_t, 2u> stride{2, 3};
  constexpr std::array<int64_t, 2u> padding{0, 4};
  constexpr std::array<int64_t, 2u> dilation{3, 1};

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
  } input{1, groups, 137, 199};

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
  } weights{groups, 1, 17, 7};

  float r1 = 0;
  float r2 = 0.2;
  const auto input_cpu = (r1 - r2) *
          at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto weights_cpu = (r1 - r2) *
          at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto bias_cpu = (r1 - r2) *
          at::rand({weights.output_channels},
                   at::device(at::kCPU).dtype(at::kFloat)) +
      r2;

  const double w_scale = 0.1;
  const int w_zero_point = 10;

  const double b_scale = 0.1;
  const int b_zero_point = 10;

  const auto weight_q = at::quantize_per_tensor(
      weights_cpu, w_scale, w_zero_point, c10::ScalarType::QUInt8);
  const auto bias_q = at::quantize_per_tensor(
      bias_cpu, b_scale, b_zero_point, c10::ScalarType::QUInt8);

  const auto output_cpu = at::conv2d(
      input_cpu, weights_cpu, bias_cpu, stride, padding, dilation, groups);

  const double scale = 0.10;
  const int zero_point = 10;
  const auto shape_match =
      at::rand({1, 7, 45, 67}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = input_cpu.vulkan();
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale2 = 0.15;
  const int zero_point2 = 15;
  const auto output_vulkan = at::native::vulkan::ops::quantized_conv2d(
      out_vulkan,
      weight_q,
      bias_q,
      stride,
      padding,
      dilation,
      groups,
      scale2,
      zero_point2);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(output_vulkan);
  auto output_for_dequantized_vulkan =
      vulkan_to_cpu(out_vulkan_deq, shape_match);

  float rtol = 0;
  float atol = 1;
  const auto check =
      at::allclose(output_cpu, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_sub) {
  float r1 = 4.0;
  float r2 = 7.0;

  float r3 = 2.0;
  float r4 = 5.0;
  const auto in_cpu = (r1 - r2) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 = (r3 - r4) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r4;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const auto reg_subtracted_tensors = at::sub(in_cpu, in_cpu2);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto vulk_subtracted_tensors = at::native::vulkan::ops::quantized_sub(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_subtracted_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  float rtol = 0;
  float atol = 0.5;
  const auto check = at::allclose(
      reg_subtracted_tensors, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_mul) {
  const auto in_cpu =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 =
      at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto reg_mul_tensors = callOpByName(
      "quantized::mul", "", out_cpu, out_cpu2, scale3, zero_point3);
  const auto vulk_mul_tensors = at::native::vulkan::ops::quantized_mul(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_mul_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  float rtol = 0;
  float atol = 1.5;
  const auto check = at::allclose(
      at::dequantize(reg_mul_tensors[0].toTensor()),
      output_for_dequantized_vulkan,
      rtol,
      atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_div) {
  float r1 = 2.0;
  float r2 = 3.5;

  float r3 = 4.0;
  float r4 = 5.5;
  const auto in_cpu = (r1 - r2) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r2;
  const auto in_vulkan = in_cpu.vulkan();
  const auto in_cpu2 = (r3 - r4) *
          at::rand({2, 13, 32, 27}, at::device(at::kCPU).dtype(at::kFloat)) +
      r4;
  const auto in_vulkan2 = in_cpu2.vulkan();

  const double scale = 0.1;
  const int zero_point = 10;

  const auto out_cpu = at::quantize_per_tensor(
      in_cpu, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_cpu2 = at::quantize_per_tensor(
      in_cpu2, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto out_vulkan2 = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan2, scale, zero_point, c10::ScalarType::QUInt8);

  const auto reg_div_tensors = at::div(in_cpu, in_cpu2);

  const double scale3 = 0.15;
  const int zero_point3 = 15;
  const auto vulk_div_tensors = at::native::vulkan::ops::quantized_div(
      out_vulkan, out_vulkan2, scale3, zero_point3);

  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(vulk_div_tensors);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  float rtol = 0;
  float atol = 1;
  const auto check =
      at::allclose(reg_div_tensors, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST_F(VulkanAPITest, quantized_upsample_nearest2d) {
  const auto in_cpu =
      at::rand({2, 13, 12, 27}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_nearest2d(in_cpu, {4, 6}, 1, 1);

  const double scale = 0.1;
  const int zero_point = 10;

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
      in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  const auto upsample_vulkan =
      at::native::vulkan::ops::quantized_upsample_nearest2d(
          out_vulkan, {4, 6}, 1, 1);

  const auto in_cpu2 =
      at::rand({2, 13, 4, 6}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_vulkan_deq =
      at::native::vulkan::ops::dequantize(upsample_vulkan);
  auto output_for_dequantized_vulkan = vulkan_to_cpu(out_vulkan_deq, in_cpu2);

  float rtol = 0;
  float atol = 1;
  const auto check =
      at::allclose(out_cpu, output_for_dequantized_vulkan, rtol, atol);

  if (!check) {
    std::cout << "Max Diff allowed: " << rtol << std::endl;
  }

  ASSERT_TRUE(check);
}

} // namespace

#endif /* USE_VULKAN_API */
