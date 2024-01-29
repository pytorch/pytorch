#include <gtest/gtest.h>

#include <ATen/xpu/XPUEvent.h>
#include <c10/util/irange.h>

#define ASSERT_EQ_XPU(X, Y) \
  {                         \
    bool _isEQ = X == Y;    \
    ASSERT_TRUE(_isEQ);     \
  }

TEST(XpuEventTest, testXPUEventBehavior) {
  if (!at::xpu::is_available()) {
    return;
  }
  auto stream = c10::xpu::getStreamFromPool();
  at::xpu::XPUEvent event;

  ASSERT_TRUE(event.query());
  ASSERT_TRUE(!event.isCreated());

  event.recordOnce(stream);
  ASSERT_TRUE(event.isCreated());

  auto wait_stream0 = c10::xpu::getStreamFromPool();
  auto wait_stream1 = c10::xpu::getStreamFromPool();

  event.block(wait_stream0);
  event.block(wait_stream1);

  wait_stream0.synchronize();
  ASSERT_TRUE(event.query());
}

void eventSync(sycl::event& event) {
  event.wait();
}

void clearHostData(int* hostData, int numel) {
  for (const auto i : c10::irange(numel)) {
    hostData[i] = 0;
  }
}

void validateHostData(int* hostData, int numel) {
  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ_XPU(hostData[i], i);
  }
}

TEST(XpuEventTest, testXPUEventFunction) {
  if (!at::xpu::is_available()) {
    return;
  }

  constexpr int numel = 1024;
  int hostData[numel];
  for (const auto i : c10::irange(numel)) {
    hostData[i] = i;
  }

  auto stream = c10::xpu::getStreamFromPool();
  int* deviceData = sycl::malloc_device<int>(numel, stream);

  // H2D
  stream.queue().memcpy(deviceData, hostData, sizeof(int) * numel);
  at::xpu::XPUEvent event;
  event.record(stream);
  // To validate the implicit conversion of an XPUEvent to sycl::event.
  eventSync(event);
  ASSERT_TRUE(event.query());

  clearHostData(hostData, numel);

  // D2H
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  event.record(stream);
  event.synchronize();

  validateHostData(hostData, numel);

  clearHostData(hostData, numel);
  // D2H
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  // The event has already been created, so there will be no recording of the
  // stream via recordOnce() here.
  event.recordOnce(stream);
  ASSERT_TRUE(event.query());

  stream.synchronize();
  sycl::free(deviceData, c10::xpu::get_device_context());

  if (at::xpu::device_count() <= 1) {
    return;
  }
  c10::xpu::set_device(1);
  auto stream1 = c10::xpu::getStreamFromPool();
  ASSERT_THROW(event.record(stream1), c10::Error);
}
