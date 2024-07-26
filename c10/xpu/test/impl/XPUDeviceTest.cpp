#include <gtest/gtest.h>

#include <c10/xpu/XPUFunctions.h>

bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

TEST(XPUDeviceTest, DeviceBehavior) {
  if (!has_xpu()) {
    return;
  }

  c10::xpu::set_device(0);
  EXPECT_EQ(c10::xpu::current_device(), 0);

  if (c10::xpu::device_count() <= 1) {
    return;
  }

  c10::xpu::set_device(1);
  EXPECT_EQ(c10::xpu::current_device(), 1);
  EXPECT_EQ(c10::xpu::exchange_device(0), 1);
  EXPECT_EQ(c10::xpu::current_device(), 0);
}

TEST(XPUDeviceTest, DeviceProperties) {
  if (!has_xpu()) {
    return;
  }

  c10::xpu::DeviceProp device_prop{};
  c10::xpu::get_device_properties(&device_prop, 0);

  EXPECT_TRUE(device_prop.max_compute_units > 0);
  EXPECT_TRUE(device_prop.gpu_eu_count > 0);
}

TEST(XPUDeviceTest, PointerGetDevice) {
  if (!has_xpu()) {
    return;
  }

  sycl::device& raw_device = c10::xpu::get_raw_device(0);
  void* ptr =
      sycl::malloc_device(8, raw_device, c10::xpu::get_device_context());

  EXPECT_EQ(c10::xpu::get_device_idx_from_pointer(ptr), 0);
  sycl::free(ptr, c10::xpu::get_device_context());

  int dummy = 0;
  ASSERT_THROW(c10::xpu::get_device_idx_from_pointer(&dummy), c10::Error);
}
