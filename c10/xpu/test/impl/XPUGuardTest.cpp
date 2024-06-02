#include <gtest/gtest.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/test/impl/XPUTest.h>

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

TEST(XPUGuardTest, EventBehavior) {
  if (!has_xpu()) {
    return;
  }

  auto device = c10::Device(c10::kXPU, c10::xpu::current_device());
  c10::impl::VirtualGuardImpl impl(device.type());
  c10::Stream stream1 = impl.getStream(device);
  c10::Stream stream2 = impl.getStream(device);
  c10::Event event1(device.type());
  // event is lazily created.
  EXPECT_FALSE(event1.eventId());

  constexpr int numel = 1024;
  int hostData1[numel];
  initHostData(hostData1, numel);
  int hostData2[numel];
  clearHostData(hostData2, numel);

  auto xpu_stream1 = c10::xpu::XPUStream(stream1);
  int* deviceData1 = sycl::malloc_device<int>(numel, xpu_stream1);

  // Copy hostData1 to deviceData1 via stream1, and then copy deviceData1 to
  // hostData2 via stream2.
  xpu_stream1.queue().memcpy(deviceData1, hostData1, sizeof(int) * numel);
  // stream2 wait on stream1's completion.
  event1.record(stream1);
  event1.block(stream2);
  auto xpu_stream2 = c10::xpu::XPUStream(stream2);
  xpu_stream2.queue().memcpy(hostData2, deviceData1, sizeof(int) * numel);
  xpu_stream2.synchronize();

  EXPECT_TRUE(event1.query());
  validateHostData(hostData2, numel);
  event1.record(stream2);
  event1.synchronize();
  EXPECT_TRUE(event1.query());

  clearHostData(hostData2, numel);
  xpu_stream1.queue().memcpy(deviceData1, hostData1, sizeof(int) * numel);
  // stream2 wait on stream1's completion.
  event1.record(stream1);
  event1.block(stream2);
  // event1 will overwrite the previously captured state.
  event1.record(stream2);
  xpu_stream2.queue().memcpy(hostData2, deviceData1, sizeof(int) * numel);
  xpu_stream2.synchronize();
  EXPECT_TRUE(event1.query());
  validateHostData(hostData2, numel);

  clearHostData(hostData2, numel);
  // ensure deviceData1 and deviceData2 are different buffers.
  int* deviceData2 = sycl::malloc_device<int>(numel, xpu_stream1);
  sycl::free(deviceData1, c10::xpu::get_device_context());
  c10::Event event2(device.type());

  // Copy hostData1 to deviceData2 via stream1, and then copy deviceData2 to
  // hostData1 via stream1.
  xpu_stream1.queue().memcpy(deviceData2, hostData1, sizeof(int) * numel);
  event2.record(xpu_stream1);
  event2.synchronize();
  EXPECT_TRUE(event2.query());
  clearHostData(hostData1, numel);
  xpu_stream1.queue().memcpy(hostData1, deviceData2, sizeof(int) * numel);
  event2.record(xpu_stream1);
  event2.synchronize();
  EXPECT_TRUE(event2.query());
  EXPECT_NE(event1.eventId(), event2.eventId());
  ASSERT_THROW(event1.elapsedTime(event2), c10::Error);
  sycl::free(deviceData2, c10::xpu::get_device_context());
}
