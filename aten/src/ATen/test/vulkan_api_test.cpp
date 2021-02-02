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

  return diff.abs().max().item<float>() < (tolerance * maxValue);
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

} // namespace

namespace {

TEST(VulkanAPITest, adaptive_avg_pool2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({5, 7, 47, 31}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::adaptive_avg_pool2d(in_cpu, {3, 3});
  const auto out_vulkan = at::adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.1f);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << c_cpu << std::endl;
    std::cout << "Got:\n" << c_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << a_cpu << std::endl;
    std::cout << "Got:\n" << a_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << c_cpu << std::endl;
    std::cout << "Got:\n" << c_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << a_cpu << std::endl;
    std::cout << "Got:\n" << a_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

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
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << cpu << std::endl;
    std::cout << "Got:\n" << vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, conv2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  constexpr int64_t groups = 1;
  constexpr std::array<int64_t, 2u> stride{1, 2};
  constexpr std::array<int64_t, 2u> padding{3, 0};
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
  } input {1, 37, 223, 227};

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
  } weights {83, input.channels, 13, 2};

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
    std::cout << "Expected:\n" << output_cpu << std::endl;
    std::cout << "Got:\n" << output_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << output_cpu << std::endl;
    std::cout << "Got:\n" << output_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << output_cpu << std::endl;
    std::cout << "Got:\n" << output_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << cpu << std::endl;
    std::cout << "Got:\n" << vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, empty) {
  if (!at::is_vulkan_available()) {
    return;
  }

  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

TEST(VulkanAPITest, mean) {
  const auto in_cpu = at::rand({17, 3, 79, 53}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::mean(in_cpu, {-1, -2}, true);

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::mean(in_vulkan, {-1, -2}, true);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
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
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mul_scalar) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::mul(a_cpu, b_scalar);
  const auto c_vulkan = at::mul(a_vulkan, b_scalar);

  const auto check = almostEqual(c_cpu, c_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << c_cpu << std::endl;
    std::cout << "Got:\n" << c_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, mul_scalar_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.mul_(b_scalar);
  a_vulkan.mul_(b_scalar);

  const auto check = almostEqual(a_cpu, a_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << a_cpu << std::endl;
    std::cout << "Got:\n" << a_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, reshape) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({47, 11, 83, 97}, at::device(at::kCPU).dtype(at::kFloat));
  const auto in_vulkan = in_cpu.vulkan();

  const std::array<int64_t, 2> shape{47 * 83, 11 * 97};

  const auto out_cpu = at::reshape(in_cpu, shape);
  const auto out_vulkan = at::reshape(in_vulkan, shape);

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
  std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, reshape_) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto cpu = at::rand({59, 41, 19, 67}, at::device(at::kCPU).dtype(at::kFloat));
  const auto vulkan = cpu.vulkan();

  const std::array<int64_t, 3> shape{59, 41 * 67, 19};

  cpu.reshape(shape);
  vulkan.reshape(shape);

  const auto check = almostEqual(cpu, vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << cpu << std::endl;
    std::cout << "Got:\n" << vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, upsample_nearest2d) {
  if (!at::is_vulkan_available()) {
    return;
  }

  const auto in_cpu = at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  const auto out_cpu = at::upsample_nearest2d(in_cpu, {4, 6});

  const auto in_vulkan = in_cpu.vulkan();
  const auto out_vulkan = at::upsample_nearest2d(in_vulkan, {4, 6});

  const auto check = almostEqual(out_cpu, out_vulkan.cpu());
  if (!check) {
    std::cout << "Expected:\n" << out_cpu << std::endl;
    std::cout << "Got:\n" << out_vulkan.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

enum class OpType {
  addmm,
  conv2d,
  hardtanh_,
  mean,
 };

class BaseOp {
 public:
  explicit BaseOp(const OpType type) : type_(type) {}
  virtual ~BaseOp() = default;

  virtual at::Tensor run(at::Tensor&) const = 0;
  virtual std::string toString() const = 0;

 private:
  OpType type_;
};

class Addmm final : public BaseOp {
 public:
  Addmm(
      const int64_t m1H,
      const int64_t m1W,
      const int64_t m2W,
      const float beta,
      const float alpha)
    : BaseOp(OpType::addmm),
      m2_(at::rand(c10::IntArrayRef({m1W, m2W}), at::device(at::kCPU).dtype(at::kFloat))),
      b_(at::rand(c10::IntArrayRef({m1H, m2W}), at::device(at::kCPU).dtype(at::kFloat))),
      beta_(beta),
      alpha_(alpha) {
  }

  at::Tensor run(at::Tensor& t) const override {
    if (t.is_vulkan()) {
      return at::addmm(b_, t, m2_, beta_, alpha_);
    }

    return at::addmm(b_, t, m2_, beta_, alpha_);
  }

  std::string toString() const override {
    return "addmm";
  }

 private:
  at::Tensor m2_;
  at::Tensor b_;
  float beta_;
  float alpha_;
};

class Conv2d final : public BaseOp {
 public:
  Conv2d(
      const c10::IntArrayRef wsizes,
      const int64_t groups,
      const int64_t stride,
      const int64_t padding)
      : BaseOp(OpType::conv2d),
        groups_(groups),
        stride_(stride),
        padding_(padding),
        w_(at::rand(wsizes, at::device(at::kCPU).dtype(at::kFloat))),
        b_(at::rand(wsizes[0], at::device(at::kCPU).dtype(at::kFloat))){
  }

  at::Tensor run(at::Tensor& t) const override {
    return at::conv2d(t, w_, b_, {stride_}, {padding_}, {1}, groups_);
  }

  std::string toString() const override {
    return "conv2d";
  }

 private:
  int64_t groups_;
  int64_t stride_;
  int64_t padding_;
  at::Tensor w_;
  at::Tensor b_;
};

class Hardtanh_ final : public BaseOp {
 public:
  Hardtanh_() : BaseOp(OpType::hardtanh_) {}

  at::Tensor run(at::Tensor& input) const override {
    return at::hardtanh_(input, 0, 6);
  }

  std::string toString() const override {
    return "hardtanh_";
  }
};

class Mean final : public BaseOp {
 public:
  Mean() : BaseOp(OpType::mean) {}

  at::Tensor run(at::Tensor& input) const override {
    return at::mean(input, {2, 3}, false);
  }

  std::string toString() const override {
    return "mean";
  }
};

class OpsList {
 public:
  OpsList() {}
  explicit OpsList(std::vector<std::unique_ptr<BaseOp>> ops)
    : ops_(std::move(ops)) {
  }

  auto run(const at::Tensor& input) {
    at::Tensor output = input;

    for (const auto& op : ops_) {
      output = op->run(output);
    }

    return output;
  }

  auto run(const at::Tensor& input, const at::Tensor& v_input) {
    at::Tensor output = input;
    at::Tensor v_output = v_input;

    for (const auto& op : ops_) {
      output = op->run(output);
      v_output = op->run(v_output);
    }

    return std::make_pair(output, v_output);
  }

 protected:
  std::vector<std::unique_ptr<BaseOp>> ops_;
};

class MobileNetV2 final : public OpsList {
 public:
  MobileNetV2() {
    ops_.emplace_back(new Conv2d({32, 3, 3, 3}, 1, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 1, 3, 3}, 32, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({16, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({96, 16, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 1, 3, 3}, 96, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({24, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({24, 144, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 144, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 192, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 384, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 2, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({160, 576, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Conv2d({320, 960, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Conv2d({1280, 320, 1, 1}, 1, 1, 0));
    ops_.emplace_back(new Hardtanh_());
    ops_.emplace_back(new Mean());
    ops_.emplace_back(new Addmm(1, 1280, 1000, 0, 1));
  }
};

TEST(VulkanAPITest, mobilenetv2) {
  if (!at::is_vulkan_available()) {
    return;
  }

  MobileNetV2 mn2;

  const auto input = at::rand({1, 3, 224, 224}, at::device(at::kCPU).dtype(at::kFloat));
  const auto output = mn2.run(input, input.vulkan());

  const auto check = almostEqual(output.first, output.second.cpu());
  if (!check) {
    std::cout << "Expected:\n" << output.first << std::endl;
    std::cout << "Got:\n" << output.second.cpu() << std::endl;
  }

  ASSERT_TRUE(check);
}

} // namespace

#endif /* USE_VULKAN_API */
