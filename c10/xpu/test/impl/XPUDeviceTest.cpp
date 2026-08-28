#include <gtest/gtest.h>

#include <c10/xpu/XPUFunctions.h>

static bool has_xpu() {
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

#if SYCL_COMPILER_VERSION >= 20260100
  EXPECT_TRUE(device_prop.xe_stack_count > 0);
  EXPECT_TRUE(device_prop.xe_regions_per_stack > 0);
  EXPECT_TRUE(device_prop.xe_clusters_per_region > 0);
  EXPECT_EQ(
      device_prop.xe_stack_count * device_prop.xe_regions_per_stack *
          device_prop.xe_clusters_per_region * device_prop.xe_cores_per_cluster,
      device_prop.gpu_eu_count / device_prop.gpu_eu_count_per_subslice);
  EXPECT_EQ(device_prop.eus_per_xe_core, device_prop.gpu_eu_count_per_subslice);
  EXPECT_TRUE(device_prop.max_lanes_per_hw_thread > 0);
  auto max_subgroup_size = *std::max_element(
      device_prop.sub_group_sizes.begin(), device_prop.sub_group_sizes.end());
  EXPECT_EQ(device_prop.max_lanes_per_hw_thread, max_subgroup_size);
#endif
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
