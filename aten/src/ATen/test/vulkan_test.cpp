#include <gtest/gtest.h>

#include "ATen/ATen.h"
#include "ATen/vulkan/Context.h"

TEST(VulkanTest, ToVulkanToCpu) {
  if (!at::vulkan::is_available())
    return;
  auto t =
      at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto tv = t.vulkan();
  ASSERT_TRUE(tv.options().device().type() == at::kVulkan);
  auto t2 = tv.cpu();
  ASSERT_TRUE(t2.options().device().type() == at::kCPU);
  ASSERT_TRUE(t2.equal(t));
}

TEST(VulkanTest, FailOnStrides) {
  if (!at::vulkan::is_available())
    return;
  auto t = at::empty({1, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto tv = t.vulkan();
  ASSERT_ANY_THROW(tv.strides());
  ASSERT_ANY_THROW(tv.stride(0));
}

TEST(VulkanTest, UpsampleNearest2D) {
  if (!at::vulkan::is_available())
    return;

  auto t_in =
      at::rand({1, 2, 2, 3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::upsample_nearest2d(t_in, {4, 6});
  auto tv_in =
      t_in.to(at::TensorOptions{at::Device{at::kVulkan}}.dtype(at::kFloat));

  auto tv_out = at::upsample_nearest2d(tv_in, {4, 6});
  auto t_out =
      tv_out.to(at::TensorOptions{at::Device{at::kCPU}}.dtype(at::kFloat));

  ASSERT_TRUE(t_out.equal(t_out_expected));
}

TEST(VulkanTest, Add) {
  if (!at::vulkan::is_available())
    return;
  auto t_in0 = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_in1 = at::rand({1, 2, 2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::add(t_in0, t_in1, 2);
  auto tv_in0 = t_in0.vulkan();
  auto tv_in1 = t_in1.vulkan();
  auto tv_out = at::add(tv_in0, tv_in1, 2);
  auto t_out = tv_out.cpu();

  ASSERT_TRUE(t_out.equal(t_out_expected));
}
