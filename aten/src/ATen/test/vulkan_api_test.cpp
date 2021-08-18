#ifdef USE_VULKAN_API

#include <gtest/gtest.h>
#include <ATen/ATen.h>

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

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.0f;
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
  const int limit = 500;
  int printed = 0;
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
  else if (diff.sizes().size() == 4) {
    for (int n = 0; n < diff.sizes()[0] && printed < limit; n++) {
      for (int c = 0; c < diff.sizes()[1] && printed < limit; c++) {
        for (int y = 0; y < diff.sizes()[2] && printed < limit; y++) {
          for (int x = 0; x < diff.sizes()[3] && printed < limit; x++) {
            const float diff_xy = diff[n][c][y][x].item<float>();
            if (diff_xy > maxDiff) {
              const float a_xy = a[n][c][y][x].item<float>();
              const float b_xy = b[n][c][y][x].item<float>();
              std::cout << std::setw(8) << n << std::setw(8) << c << std::setw(8) << y << std::setw(8) << x << std::setw(24) << std::setprecision(4) << a_xy << std::setw(16) << std::setprecision(4) << b_xy << std::endl;;
              printed++;
            }
          }
        }
      }
    }
  }
}

void showRtolCustom(const at::Tensor& a, const at::Tensor& b, const at::Tensor& d) {
  const auto diff = (a - b).abs();

  float maxValue = a.abs().max().item<float>();
  maxValue = fmax(b.abs().max().item<float>(), maxValue);

#ifdef USE_VULKAN_FP16_INFERENCE
  constexpr float tolerance = 1e-2;
#else
  constexpr float tolerance = 1e-5;
#endif

  const float maxDiff = maxValue * tolerance;
  const int limit = 10;
  int printed = 0;
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
  else if (diff.sizes().size() == 4) {
    for (int n = 0; n < diff.sizes()[0] && printed < limit; n++) {
      for (int c = 0; c < diff.sizes()[1] && printed < limit; c++) {
        bool show_plane = false;
        for (int y = 0; y < diff.sizes()[2] && printed < limit; y++) {
          for (int x = 0; x < diff.sizes()[3] && printed < limit; x++) {
            const float diff_xy = diff[n][c][y][x].item<float>();
            if (diff_xy > maxDiff) {
              show_plane = true;
              const float a_xy = a[n][c][y][x].item<float>();
              const float b_xy = b[n][c][y][x].item<float>();
              const float c_xy = d[n][c][y][x].item<float>();
              std::cout << std::setw(8) << n
                        << std::setw(8) << c
                        << std::setw(8) << y
                        << std::setw(8) << x
                        << std::setw(24) << std::setprecision(4) << a_xy
                        << std::setw(16) << std::setprecision(4) << b_xy
                        << std::setw(16) << std::setprecision(4) << c_xy
                        << std::endl;
              printed++;
            }
          }
        }
        if (show_plane) {
          std::cout << b[n][c] << std::endl;
        }
      }
    }
  }
}

} // namespace

namespace {

TEST(VulkanAPITest, adaptive_avg_pool2d) {
  if (!at::is_vulkan_available()) {
    return;
  }
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({5, 7, 47, 31}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::adaptive_avg_pool2d(in_cpu, {3, 3});
  const auto out_vulkan = at::adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_broadcast0_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({16, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({16, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_broadcast1_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({3, 8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_scalar) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_scalar_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, addmm) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({179, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, addmm_expand) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr float alpha = 2.1f;
  constexpr float beta = 103.24;

  const auto bias_cpu = at::rand({1000}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m1_cpu = at::rand({1, 1280}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({1280, 1000}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = at::addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, avg_pool2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({3, 19, 43, 79}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::avg_pool2d(in_cpu, {5, 3}, {1, 2}, {2, 0}, true);
  const auto out_vulkan = at::avg_pool2d(in_cpu.vulkan(), {5, 3}, {1, 2}, {2, 0}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, clamp) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  const auto out_cpu = at::clamp(in_cpu, min_value, max_value);
  const auto out_vulkan = at::clamp(in_vulkan, min_value, max_value);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, clamp_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  cpu.clamp_(min_value, max_value);
  vulkan.clamp_(min_value, max_value);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

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

TEST(VulkanAPITest, conv2d_dw) {
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
  } input {1, groups, 137, 199};

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
  } weights {groups, 1, 17, 7};

  const auto input_cpu = at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::rand({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

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
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, conv2d_pw) {
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
  } input {1, 17, 127, 397};

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
  } weights {29, input.channels, 1, 1};

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
      groups);

  const bool check = almostEqual(output_cpu, output_vulkan.cpu());
  if (!check) {
    showRtol(output_cpu, output_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, conv2d_winograd) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 1};
  constexpr std::array<int64_t, 2u> padding{2, 2};
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
  } input {1, 10, 177, 232};

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
  } weights {13, input.channels, 3, 3};

  const auto input_cpu = at::rand(input.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto weights_cpu = at::rand(weights.size(), at::device(at::kCPU).dtype(at::kFloat));
  const auto bias_cpu = at::rand({weights.output_channels}, at::device(at::kCPU).dtype(at::kFloat));

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

TEST(VulkanAPITest, copy) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto cpu = at::rand({13, 17, 37, 19}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const auto check = exactlyEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, div_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat))+0.01;
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat))+0.01;
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.div_(b_cpu);
  a_vulkan.div_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, div_broadcast0_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({12, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat))+0.01;
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({12, 17, 29, 1}, at::device(at::kCPU).dtype(at::kFloat))+0.01;
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.div_(b_cpu);
  a_vulkan.div_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, div_broadcast1_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({3, 8, 29, 83}, at::device(at::kCPU).dtype(at::kFloat))+0.01;
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({8, 1, 1}, at::device(at::kCPU).dtype(at::kFloat))+0.01;
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.div_(b_cpu);
  a_vulkan.div_(b_vulkan);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(b_cpu, b_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, div_scalar) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::div(a_cpu, b_scalar);
  const auto c_vulkan = at::div(a_vulkan, b_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtol(c_cpu, c_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, div_scalar_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.div_(b_scalar);
  a_vulkan.div_(b_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    showRtol(a_cpu, a_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, empty) {
  if (!at::is_vulkan_available()) {
    return;
  }

  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

TEST(VulkanAPITest, hardsigmoid) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::hardsigmoid(in_cpu);
  const auto out_vulkan = at::hardsigmoid(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, hardsigmoid_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  auto vulkan = cpu.vulkan();

  at::hardsigmoid_(cpu);
  at::hardsigmoid_(vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, hardswish) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  const auto in_vulkan = in_cpu.vulkan();

  const auto out_cpu = at::hardswish(in_cpu);
  const auto out_vulkan = at::hardswish(in_vulkan);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, hardswish_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat))*12 - 6;
  auto vulkan = cpu.vulkan();

  at::native::hardswish_(cpu);
  at::hardswish_(vulkan);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    showRtol(cpu, vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, max_pool2d) {
  if (!at::is_vulkan_available()) {
    return;
  }
  c10::InferenceMode mode;

  const auto in_cpu = at::rand({5, 13, 55, 68}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::max_pool2d(in_cpu, {3, 4}, {2, 1}, {1, 1}, {1, 1}, false);
  const auto out_vulkan = at::max_pool2d(in_cpu.vulkan(), {3, 4}, {2, 1}, {1, 1}, {1,1}, false);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mean) {
  const auto in_cpu = at::rand({17, 3, 79, 53}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::mean(in_cpu, {-1, -2}, true);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::mean(in_vulkan, {-1, -2}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mean2d) {
  const auto in_cpu = at::rand({11, 7, 173, 37}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::mean(in_cpu, {-1, -2}, false);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::mean(in_vulkan, {-1, -2}, false);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}

/*
TEST(VulkanAPITest, mm) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto m1_cpu = at::rand({179, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto m2_cpu = at::rand({67, 163}, at::device(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = m1_cpu.mm(m2_cpu);

  const auto m1_vulkan = m1_cpu.vulkan();
  const auto out_vulkan = m1_vulkan.mm(m2_cpu);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    showRtol(out_cpu, out_vulkan.cpu());
  }

  ASSERT_TRUE(check);
}
*/

std::array<int64_t, 4u> in0_size{3, 4, 100, 100};
std::array<int64_t, 4u> in1_size{3, 4,   1, 100};

TEST(VulkanAPITest, add_broadcast1) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::zeros(in0_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::zeros(in1_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 1.0f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 1.0f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtolCustom(c_cpu, c_vulkan.cpu(), a_cpu);
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, div_broadcast1) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::ones(in0_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::ones(in1_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::div(a_cpu, b_cpu);
  const auto c_vulkan = at::div(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtolCustom(c_cpu, c_vulkan.cpu(), a_cpu);
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mul_broadcast1) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::ones(in0_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::ones(in1_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::mul(a_cpu, b_cpu);
  const auto c_vulkan = at::mul(a_vulkan, b_vulkan);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtolCustom(c_cpu, c_vulkan.cpu(), a_cpu);
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, sub_broadcast1) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::zeros(in0_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::zeros(in1_size, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::sub(a_cpu, b_cpu, 1.0f);
  const auto c_vulkan = at::sub(a_vulkan, b_vulkan, 1.0f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    showRtolCustom(c_cpu, c_vulkan.cpu(), a_cpu);
  }

  ASSERT_TRUE(check);
}

} // namespace

#endif /* USE_VULKAN_API */
