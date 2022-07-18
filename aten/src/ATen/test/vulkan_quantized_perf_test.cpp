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

class VulkanAPITest : public ::testing::Test {
 public:
#if defined(__ANDROID__) // to avoid `Undefined symbols for architecture arm64`
  // error
  static void SetUpTestSuite() {
    at::native::vulkan::api::context()->querypool().enable();
  }

  static void TearDownTestSuite() {
    at::native::vulkan::api::context()->querypool().disable(false);
  }
#endif
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

TEST_F(VulkanAPITest, reg_add) {
  for (int i = 0; i < 50; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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

    const auto reg_added_tensors = at::add(in_cpu, in_cpu2);
  }
}

TEST_F(VulkanAPITest, quantized_add) {
  for (int i = 0; i < 50; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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

    const auto vulk_added_tensors = at::native::vulkan::ops::quantized_add(
        out_vulkan, out_vulkan2, scale3, zero_point3);
  }
}

TEST_F(VulkanAPITest, reg_conv2d) {
  for (int i = 0; i < 5; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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
  }
}

TEST_F(VulkanAPITest, quantized_conv2d) {
  for (int i = 0; i < 5; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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

    const double scale = 0.10;
    const int zero_point = 10;
    const auto shape_match =
        at::rand({1, 1, 4, 4}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
    const auto in_vulkan = input_cpu.vulkan();
    const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
        in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

    const double scale2 = 0.15;
    const int zero_point2 = 15;
    const auto output_vulkan = at::native::vulkan::ops::conv2d(
        out_vulkan,
        weight_q,
        bias_q,
        stride,
        padding,
        dilation,
        groups,
        scale2,
        zero_point2);
  }
}

TEST_F(VulkanAPITest, reg_conv2d_pw) {
  for (int i = 0; i < 5; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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
  }
}

TEST_F(VulkanAPITest, quantized_conv2d_pw) {
  for (int i = 0; i < 5; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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

    const double scale = 0.10;
    const int zero_point = 10;
    const auto shape_match =
        at::rand({1, 29, 127, 397}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
    const auto in_vulkan = input_cpu.vulkan();
    const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
        in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

    const double scale2 = 0.15;
    const int zero_point2 = 15;
    const auto output_vulkan = at::native::vulkan::ops::conv2d(
        out_vulkan,
        weight_q,
        bias_q,
        stride,
        padding,
        dilation,
        groups,
        scale2,
        zero_point2);
  }
}

TEST_F(VulkanAPITest, reg_conv2d_dw) {
  for (int i = 0; i < 5; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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
        at::rand({1, 7, 45, 67}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
    const auto in_vulkan = input_cpu.vulkan();
    const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
        in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);
  }
}

TEST_F(VulkanAPITest, quantized_conv2d_dw) {
  for (int i = 0; i < 5; i++) {
    if (!at::is_vulkan_available()) {
      return;
    }

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

    const double scale = 0.10;
    const int zero_point = 10;
    const auto shape_match =
        at::rand({1, 7, 45, 67}, at::device(at::kCPU).dtype(at::kFloat)) * 6;
    const auto in_vulkan = input_cpu.vulkan();
    const auto out_vulkan = at::native::vulkan::ops::quantize_per_tensor(
        in_vulkan, scale, zero_point, c10::ScalarType::QUInt8);

    const double scale2 = 0.15;
    const int zero_point2 = 15;
    const auto output_vulkan = at::native::vulkan::ops::conv2d(
        out_vulkan,
        weight_q,
        bias_q,
        stride,
        padding,
        dilation,
        groups,
        scale2,
        zero_point2);
  }
}

} // namespace

#endif /* USE_VULKAN_API */
