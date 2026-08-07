#include <gtest/gtest.h>

#include <c10/xpu/XPUEvent.h>
#include <c10/xpu/test/impl/XPUTest.h>

static bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

TEST(XPUEventTest, IPCSupport) {
  if (!has_xpu()) {
    return;
  }

#if SYCL_COMPILER_VERSION >= 20260200
  if (!c10::xpu::get_raw_device(c10::xpu::current_device())
           .has(sycl::aspect::ext_oneapi_ipc_event)) {
    c10::xpu::XPUEvent event0(false, true);
    EXPECT_THROW(event0.record(), c10::Error);
    return;
  }
  c10::xpu::XPUEvent event0(true, true);
  EXPECT_THROW(event0.record(), c10::Error);

  c10::xpu::XPUEvent event1(false, true);

  event1.record();
  EXPECT_EQ(event1.event().ext_oneapi_ipc_enabled(), true);
  auto handle = event1.ipc_handle();

  auto current_device = c10::xpu::current_device();
  c10::xpu::XPUEvent event2(current_device, handle);
  EXPECT_EQ(event2.event().ext_oneapi_ipc_enabled(), true);

  event1.synchronize();
  event2.synchronize();
#else
  c10::xpu::XPUEvent event1(false, true);
  EXPECT_THROW(event1.record(), c10::Error);
#endif
}
