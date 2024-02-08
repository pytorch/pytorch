#include <gtest/gtest.h>

#include <c10/xpu/XPUGuard.h>

bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

TEST(XPUGuardTest, GuardBehavior) {
  if (!has_xpu()) {
    return;
  }

  EXPECT_EQ(c10::xpu::current_device(), 0);
  std::vector<c10::xpu::XPUStream> streams0 = {
      c10::xpu::getStreamFromPool(), c10::xpu::getStreamFromPool(true)};
  EXPECT_EQ(streams0[0].device_index(), 0);
  EXPECT_EQ(streams0[1].device_index(), 0);
  c10::xpu::setCurrentXPUStream(streams0[0]);
  EXPECT_EQ(c10::xpu::getCurrentXPUStream(), streams0[0]);

  if (c10::xpu::device_count() <= 1) {
    return;
  }

  // Test for XPUGuard.
  std::vector<c10::xpu::XPUStream> streams1;
  {
    c10::xpu::XPUGuard device_guard(1);
    streams1.push_back(c10::xpu::getStreamFromPool());
    streams1.push_back(c10::xpu::getStreamFromPool());
  }

  EXPECT_EQ(streams1[0].device_index(), 1);
  EXPECT_EQ(streams1[1].device_index(), 1);
  EXPECT_EQ(c10::xpu::current_device(), 0);

  // Test for OptionalXPUGuard.
  {
    c10::xpu::OptionalXPUGuard device_guard;
    EXPECT_EQ(device_guard.current_device().has_value(), false);
    device_guard.set_index(1);
    EXPECT_EQ(c10::xpu::current_device(), 1);
  }

  EXPECT_EQ(c10::xpu::current_device(), 0);
  c10::xpu::setCurrentXPUStream(streams1[0]);

  // Test for XPUStreamGuard.
  {
    c10::xpu::XPUStreamGuard guard(streams1[1]);
    EXPECT_EQ(guard.current_device(), at::Device(at::kXPU, 1));
    EXPECT_EQ(c10::xpu::current_device(), 1);
    EXPECT_EQ(c10::xpu::getCurrentXPUStream(1), streams1[1]);
  }

  EXPECT_EQ(c10::xpu::current_device(), 0);
  EXPECT_EQ(c10::xpu::getCurrentXPUStream(), streams0[0]);
  EXPECT_EQ(c10::xpu::getCurrentXPUStream(1), streams1[0]);

  // Test for OptionalXPUStreamGuard.
  {
    c10::xpu::OptionalXPUStreamGuard guard;
    EXPECT_EQ(guard.current_stream().has_value(), false);
    guard.reset_stream(streams1[1]);
    EXPECT_EQ(guard.current_stream(), streams1[1]);
    EXPECT_EQ(c10::xpu::current_device(), 1);
    EXPECT_EQ(c10::xpu::getCurrentXPUStream(1), streams1[1]);
  }

  EXPECT_EQ(c10::xpu::current_device(), 0);
  EXPECT_EQ(c10::xpu::getCurrentXPUStream(), streams0[0]);
  EXPECT_EQ(c10::xpu::getCurrentXPUStream(1), streams1[0]);
}
