#include <gtest/gtest.h>

#include "ATen/ATen.h"
//#include "ATen/native/vulkan/VulkanAten.h"

TEST(VulkanTest, UpsampleNearest2D) {
  if (!at::vulkan::is_available()) {
    return;
  }

  auto t_in =
      at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::upsample_nearest2d(t_in, {4, 6});
  auto tv_in = t_in.to(c10::TensorOptions{at::Device{at::kVulkan}}
                           .layout(at::kVulkanLayout)
                           .dtype(at::kFloat));

  auto tv_out = at::upsample_nearest2d(tv_in, {4, 6});
  auto t_out = tv_out.to(c10::TensorOptions{at::Device{at::kCPU}}
                             .layout(c10::kStrided)
                             .dtype(at::kFloat));

  ASSERT_TRUE(t_out.equal(t_out_expected));
}
