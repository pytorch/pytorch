#include <gtest/gtest.h>

#include <c10/xpu/XPUEvent.h>
#include <c10/xpu/test/impl/XPUTest.h>

static bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

TEST(XPUEventTest, IPCSupport) {
  if (!has_xpu()) {
    GTEST_SKIP() << "XPU not available, skipping test";
  }

#if SYCL_COMPILER_VERSION >= 20260200
  auto& device = c10::xpu::get_raw_device(c10::xpu::current_device());
  if (!device.has(sycl::aspect::ext_oneapi_per_event_profiling) ||
      !device.has(sycl::aspect::ext_oneapi_ipc_event)) {
    c10::xpu::XPUEvent event0(false, true);
    EXPECT_THROW(event0.record(), c10::Error);
    GTEST_SKIP() << "XPU IPC not supported, skipping test";
  }
  c10::xpu::XPUEvent event0(true, true);
  EXPECT_THROW(event0.record(), c10::Error);

  c10::xpu::XPUEvent event1(false, true);

  event1.record();
  EXPECT_EQ(event1.event().ext_oneapi_ipc_enabled(), true);
  auto handle = event1.ipc_handle();

  auto current_device = c10::xpu::current_device();
  c10::xpu::XPUEvent event2(current_device, handle);
  EXPECT_EQ(event2.event().ext_oneapi_ipc_enabled(), false);

  event1.synchronize();
  event2.synchronize();

  c10::xpu::XPUEvent event3(true);
  event3.record();
  event3.synchronize();
  EXPECT_THROW(event3.ipc_handle(), c10::Error);
  EXPECT_THROW(event3.elapsed_time(event2), c10::Error);
#else
  c10::xpu::XPUEvent event1(false, true);
  EXPECT_THROW(event1.record(), c10::Error);
#endif
}
