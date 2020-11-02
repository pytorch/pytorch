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
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);

  ASSERT_TRUE(almostEqual(a_cpu, a_vulkan.cpu()));
}

TEST(VulkanAPITest, add_scalar) {
  const auto a_cpu = at::rand({1, 1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, add_scalar_) {
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);

  ASSERT_TRUE(almostEqual(a_cpu, a_vulkan.cpu()));
}

TEST(VulkanAPITest, mul_scalar) {
  const auto a_cpu = at::rand({17, 213, 213, 7}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::mul(a_cpu, b_scalar);
  const auto c_vulkan = at::mul(a_vulkan, b_scalar);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, mul_scalar_) {
  auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.mul_(b_scalar);
  a_vulkan.mul_(b_scalar);

  ASSERT_TRUE(almostEqual(a_cpu, a_vulkan.cpu()));
}

TEST(VulkanAPITest, clamp) {
  const auto a_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  const auto c_cpu = at::clamp(a_cpu, min_value, max_value);
  const auto c_vulkan = at::clamp(a_vulkan, min_value, max_value);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, clamp_) {
  const auto a_cpu = at::rand({17, 197, 302, 5}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float min_value = 0.2f;
  const float max_value = 0.8f;

  a_cpu.clamp_(min_value, max_value);
  a_vulkan.clamp_(min_value, max_value);

  ASSERT_TRUE(almostEqual(a_cpu, a_vulkan.cpu()));
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
