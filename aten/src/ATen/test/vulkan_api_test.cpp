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

TEST(VulkanAPITest, copy) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  {
    const auto vulkan = at::empty({1, 3, 64, 64}, at::device(at::kVulkan).dtype(at::kFloat));
    const auto cpu = vulkan.cpu();
  }

  {
    const auto cpu = at::empty({1, 3, 64, 64}, at::device(at::kCPU).dtype(at::kFloat));
    const auto vulkan = cpu.vulkan();
  }
}

TEST(VulkanAPITest, empty) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  ASSERT_NO_THROW(at::empty({1, 3, 64, 64}, at::device(at::kVulkan).dtype(at::kFloat)));
}

} // namespace

#endif /* USE_VULKAN_API */
