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

TEST(VulkanAPITest, add) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  const auto a_cpu = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_cpu = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  const auto c_cpu = at::add(a_cpu, b_cpu, 2);
  const auto a_vulkan = a_cpu.vulkan();
  const auto b_vulkan = b_cpu.vulkan();
  // const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2);

  // ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

} // namespace

#endif /* USE_VULKAN_API */
