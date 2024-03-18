#include <gtest/gtest.h>

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

TEST(XPUGuardTest, GuardBehavior) {
  if (!has_xpu()) {
    return;
  }

  {
    auto device = c10::Device(c10::kXPU);
    const c10::DeviceGuard device_guard(device);
    EXPECT_EQ(c10::xpu::current_device(), 0);
  }

  std::vector<c10::xpu::XPUStream> streams0 = {
      c10::xpu::getStreamFromPool(), c10::xpu::getStreamFromPool(true)};
  EXPECT_EQ(streams0[0].device_index(), 0);
  EXPECT_EQ(streams0[1].device_index(), 0);
  c10::xpu::setCurrentXPUStream(streams0[0]);
  EXPECT_EQ(c10::xpu::getCurrentXPUStream(), streams0[0]);

  if (c10::xpu::device_count() <= 1) {
    return;
  }

  // Test DeviceGuard for XPU.
  std::vector<c10::xpu::XPUStream> streams1;
  {
    auto device = c10::Device(c10::kXPU, 1);
    const c10::DeviceGuard device_guard(device);
    streams1.push_back(c10::xpu::getStreamFromPool());
    streams1.push_back(c10::xpu::getStreamFromPool());
  }

  EXPECT_EQ(streams1[0].device_index(), 1);
  EXPECT_EQ(streams1[1].device_index(), 1);
  EXPECT_EQ(c10::xpu::current_device(), 0);
}
