#include <gtest/gtest.h>

#ifdef VULKAN_API_TEST

#include <ATen/native/vulkan/api/api.h>

namespace {

TEST(VulkanAPITest, Context) {
  constexpr bool kDebug = true;
  ASSERT_NO_THROW(at::native::vulkan::api::Context{kDebug});
}

} // namespace

#endif /* VULKAN_API_TEST */
