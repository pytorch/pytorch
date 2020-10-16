#include <gtest/gtest.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>
#include <test/cpp/jit/test_utils.h>

namespace {

TEST(VulkanAPITest, empty) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  ASSERT_NO_THROW(at::empty({1, 3, 64, 64}, at::device(at::kVulkan).dtype(at::kFloat)));
}

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

} // namespace

#endif /* USE_VULKAN_API */
