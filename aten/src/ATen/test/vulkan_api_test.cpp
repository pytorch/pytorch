#include <gtest/gtest.h>

#include <ATen/ATen.h>

#ifdef USE_VULKAN_API

namespace {

TEST(VulkanAPITest, empty) {
  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

} // namespace

#endif /* USE_VULKAN_API */
