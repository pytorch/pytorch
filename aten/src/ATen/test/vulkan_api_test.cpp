#include <gtest/gtest.h>

#include <ATen/native/vulkan/api/api.h>

namespace {

TEST(VulkanAPITest, Context) {
  constexpr bool kDebug = true;
  ASSERT_NO_THROW(at::native::vulkan::api::Context{kDebug});
}

} // namespace
