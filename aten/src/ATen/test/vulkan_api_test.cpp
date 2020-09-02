#include <gtest/gtest.h>

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>

namespace {

TEST(VulkanAPITest, Runtime) {
  const auto kMode = at::native::vulkan::api::Runtime::Type::Debug;
  ASSERT_NO_THROW(at::native::vulkan::api::Runtime{kMode});
}

TEST(VulkanAPITest, Context) {
  ASSERT_NO_THROW(at::native::vulkan::api::context());
}

} // namespace

#endif /* USE_VULKAN_API */
