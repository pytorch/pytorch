#include <gtest/gtest.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>
#include <test/cpp/jit/test_utils.h>

namespace {

TEST(VulkanAPITest, Runtime) {
  const auto kMode = at::native::vulkan::api::Runtime::Type::Debug;
  ASSERT_NO_THROW(at::native::vulkan::api::Runtime{kMode});
}

TEST(VulkanAPITest, Context) {
  ASSERT_NO_THROW(at::native::vulkan::api::context());
}

TEST(VulkanAPITest, empty) {
  if (!at::native::vulkan::api::available()) {
    return;
  }

  ASSERT_NO_THROW(at::empty({1, 3, 64, 64}, at::device(at::kVulkan).dtype(at::kFloat)));
}

} // namespace

#endif /* USE_VULKAN_API */
