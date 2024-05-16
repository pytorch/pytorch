#include <gtest/gtest.h>
#include <thread>
#include <chrono>

#include <ATen/xpu/XPUEvent.h>
#include <c10/util/irange.h>
#include <c10/xpu/test/impl/XPUTest.h>

TEST(XpuEventTest, testXPUEventBehavior) {
  if (!at::xpu::is_available()) {
    return;
  }
  auto stream = c10::xpu::getStreamFromPool();
  at::xpu::XPUEvent event;

  EXPECT_TRUE(event.query());
  EXPECT_TRUE(!event.isCreated());

  event.recordOnce(stream);
  EXPECT_TRUE(event.isCreated());

  auto wait_stream0 = c10::xpu::getStreamFromPool();
  auto wait_stream1 = c10::xpu::getStreamFromPool();

  event.block(wait_stream0);
  event.block(wait_stream1);

  wait_stream0.synchronize();
  EXPECT_TRUE(event.query());
}

TEST(XpuEventTest, testXPUEventCrossDevice) {
  if (at::xpu::device_count() <= 1) {
    return;
  }

  const auto stream0 = at::xpu::getStreamFromPool();
  at::xpu::XPUEvent event0;

  const auto stream1 = at::xpu::getStreamFromPool(false, 1);
  at::xpu::XPUEvent event1;

  event0.record(stream0);
  event1.record(stream1);

  event0 = std::move(event1);

  EXPECT_EQ(event0.device(), at::Device(at::kXPU, 1));

  event0.block(stream0);

  stream0.synchronize();
  ASSERT_TRUE(event0.query());
}

void eventSync(sycl::event& event) {
  event.wait();
}

TEST(XpuEventTest, testXPUEventFunction) {
  if (!at::xpu::is_available()) {
    return;
  }

  constexpr int numel = 1024;
  int hostData[numel];
  initHostData(hostData, numel);

  auto stream = c10::xpu::getStreamFromPool();
  int* deviceData = sycl::malloc_device<int>(numel, stream);

  // H2D
  stream.queue().memcpy(deviceData, hostData, sizeof(int) * numel);
  at::xpu::XPUEvent event;
  event.record(stream);
  // To validate the implicit conversion of an XPUEvent to sycl::event.
  eventSync(event);
  EXPECT_TRUE(event.query());

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
  EXPECT_TRUE(event.query());

  stream.synchronize();
  sycl::free(deviceData, c10::xpu::get_device_context());

  if (at::xpu::device_count() <= 1) {
    return;
  }
  c10::xpu::set_device(1);
  auto stream1 = c10::xpu::getStreamFromPool();
  ASSERT_THROW(event.record(stream1), c10::Error);
}

TEST(XpuEventTest, testXPUElapsedTime) {
  if (!at::xpu::is_available()) {
    return;
  }

  auto stream = c10::xpu::getStreamFromPool();
  at::xpu::XPUEvent startEvent(/*enable_timing=*/true);
  startEvent.recordOnce(stream);
  stream.synchronize();

  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  at::xpu::XPUEvent endEvent(/*enable_timing=*/true);
  endEvent.recordOnce(stream);
  stream.synchronize();

  auto elapsed_time = startEvent.elapsed_time(endEvent);
  EXPECT_GT(elapsed_time, float(0));
}

TEST(XpuEventTest, testXPUElapsedTimeDifferentStreams) {
  if (!at::xpu::is_available()) {
    return;
  }

  auto stream0 = c10::xpu::getStreamFromPool();
  at::xpu::XPUEvent startEvent(/*enable_timing=*/true);
  startEvent.recordOnce(stream0);
  stream0.synchronize();

  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  auto stream1 = c10::xpu::getStreamFromPool();
  at::xpu::XPUEvent endEvent(/*enable_timing=*/true);
  endEvent.recordOnce(stream1);
  stream1.synchronize();

  auto elapsed_time = startEvent.elapsed_time(endEvent);
  EXPECT_GT(elapsed_time, float(0));
}

TEST(XpuEventTest, testXPUElapsedTimeNotEnabled) {
  if (!at::xpu::is_available()) {
    return;
  }

  auto stream = c10::xpu::getStreamFromPool();
  at::xpu::XPUEvent startEvent(/*enable_timing=*/false);
  startEvent.recordOnce(stream);
  stream.synchronize();

  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  at::xpu::XPUEvent endEvent(/*enable_timing=*/true);
  endEvent.recordOnce(stream);
  stream.synchronize();

  EXPECT_ANY_THROW(startEvent.elapsed_time(endEvent));
}

TEST(XpuEventTest, testXPUElapsedTimeHighPriorityNotEnabled) {
  if (!at::xpu::is_available()) {
    return;
  }

  auto stream = c10::xpu::getStreamFromPool(/*high_priority=*/true);
  at::xpu::XPUEvent startEvent(/*enable_timing=*/true);
  startEvent.recordOnce(stream);
  stream.synchronize();

  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  at::xpu::XPUEvent endEvent(/*enable_timing=*/true);
  endEvent.recordOnce(stream);
  stream.synchronize();

  EXPECT_ANY_THROW(startEvent.elapsed_time(endEvent));
}
