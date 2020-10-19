#include <gtest/gtest.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>

// TODO: These functions should move to a common place.

namespace {

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  double maxValue = 0.0;
  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < 2e-6 * maxValue;
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

} // namespace

namespace {

TEST(VulkanAPITest, add) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  const auto a_cpu = at::rand({1, 3, 64, 64}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_cpu = at::rand({1, 3, 64, 64}, at::device(at::kCPU).dtype(at::kFloat));
  const auto c_cpu = at::add(a_cpu, b_cpu, 2);
  const auto c_vulkan = at::add(a_cpu.vulkan(), b_cpu.vulkan(), 2);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, copy) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  const auto cpu = at::rand({1, 3, 64, 64}, at::device(at::kCPU).dtype(at::kFloat));

  ASSERT_TRUE(exactlyEqual(cpu, cpu.vulkan().cpu()));
}


TEST(VulkanAPITest, empty) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  ASSERT_NO_THROW(at::empty({1, 3, 64, 64}, at::device(at::kVulkan).dtype(at::kFloat)));
}

} // namespace

#endif /* USE_VULKAN_API */
