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

  return diff.abs().max().item<float>() < (2e-6 * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.0f;
}

} // namespace

namespace {

TEST(VulkanAPITest, add) {
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.1f);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, add_) {
  auto a_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({61, 17, 29, 83}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  ASSERT_TRUE(almostEqual(a_cpu, a_vulkan.cpu()));
}

TEST(VulkanAPITest, add_scalar) {
  const auto a_cpu = at::rand({13, 23, 59, 73}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, add_scalar_) {
  auto a_cpu = at::rand({47, 2, 23, 97}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  ASSERT_TRUE(almostEqual(a_cpu, a_vulkan.cpu()));
}

TEST(VulkanAPITest, conv2d) {
  auto OC = 2;
  auto C = 3;
  int64_t H = 3;
  int64_t W = 3;
  int64_t KH = 2;
  int64_t KW = 2;
  auto t_in = at::rand({1, C, H, W}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_w = at::rand({OC, C, KH, KW}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::zeros({OC}, at::device(at::kCPU).dtype(at::kFloat));
  int64_t groups = 1;
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};

  auto t_out_expected =
      at::conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
  auto tv_in = t_in.vulkan();
  auto tv_out = at::conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
  auto t_out = tv_out.cpu();
  bool check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, copy) {
  const auto cpu = at::rand({13, 17, 37, 19}, at::device(at::kCPU).dtype(at::kFloat));
  ASSERT_TRUE(exactlyEqual(cpu, cpu.vulkan().cpu()));
}

TEST(VulkanAPITest, empty) {
  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

} // namespace

#endif /* USE_VULKAN_API */
