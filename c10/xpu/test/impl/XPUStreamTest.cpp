#include <gtest/gtest.h>

#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUException.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/test/impl/XPUTest.h>

#include <optional>

#include <thread>
#include <unordered_set>

static bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

TEST(XPUStreamTest, CopyAndMoveTest) {
  if (!has_xpu()) {
    return;
  }

  c10::DeviceIndex device = -1;
  sycl::queue queue;
  c10::xpu::XPUStream copyStream = c10::xpu::getStreamFromPool();
  {
    auto s = c10::xpu::getStreamFromPool();
    device = s.device_index();
    queue = s.queue();

    copyStream = s;

    EXPECT_EQ(copyStream.device_index(), device);
    EXPECT_EQ(copyStream.queue(), queue);
  }

  EXPECT_EQ(copyStream.device_index(), device);
  EXPECT_EQ(copyStream.queue(), queue);

  // Tests that moving works as expected and preserves the stream
  c10::xpu::XPUStream moveStream = c10::xpu::getStreamFromPool();
  {
    auto s = c10::xpu::getStreamFromPool();
    device = s.device_index();
    queue = s.queue();

    moveStream = std::move(s);

    EXPECT_EQ(moveStream.device_index(), device);
    EXPECT_EQ(moveStream.queue(), queue);
  }

  EXPECT_EQ(moveStream.device_index(), device);
  EXPECT_EQ(moveStream.queue(), queue);
}

TEST(XPUStreamTest, StreamBehavior) {
  if (!has_xpu()) {
    return;
  }

  c10::xpu::XPUStream stream = c10::xpu::getStreamFromPool();
  EXPECT_EQ(stream.device_type(), c10::kXPU);
  c10::xpu::setCurrentXPUStream(stream);
  c10::xpu::XPUStream cur_stream = c10::xpu::getCurrentXPUStream();

  EXPECT_EQ(cur_stream, stream);
  EXPECT_EQ(stream.priority(), 0);

  auto [least_priority, greatest_priority] =
      c10::xpu::XPUStream::priority_range();
  EXPECT_EQ(least_priority, 1);
  EXPECT_EQ(greatest_priority, -1);

  stream = c10::xpu::getStreamFromPool(/* isHighPriority */ true);
  EXPECT_EQ(stream.priority(), -1);
  stream = c10::xpu::getStreamFromPool(/* isHighPriority */ false);
  EXPECT_EQ(stream.priority(), 0);

  stream = c10::xpu::getStreamFromPool(-1);
  EXPECT_EQ(stream.priority(), -1);
  stream = c10::xpu::getStreamFromPool(-10);
  EXPECT_EQ(stream.priority(), -1);
  stream = c10::xpu::getStreamFromPool(0);
  EXPECT_EQ(stream.priority(), 0);
  stream = c10::xpu::getStreamFromPool(1);
  EXPECT_EQ(stream.priority(), 1);
  stream = c10::xpu::getStreamFromPool(10);
  EXPECT_EQ(stream.priority(), 1);

  if (c10::xpu::device_count() <= 1) {
    return;
  }

  c10::xpu::set_device(0);
  stream = c10::xpu::getStreamFromPool(false, 1);
  EXPECT_EQ(stream.device_index(), 1);
  EXPECT_NE(stream.device_index(), c10::xpu::current_device());
}

static void thread_fun(std::optional<c10::xpu::XPUStream>& cur_thread_stream) {
  auto new_stream = c10::xpu::getStreamFromPool();
  c10::xpu::setCurrentXPUStream(new_stream);
  cur_thread_stream = {c10::xpu::getCurrentXPUStream()};
  EXPECT_EQ(*cur_thread_stream, new_stream);
}

// Ensures streams are thread local
TEST(XPUStreamTest, MultithreadStreamBehavior) {
  if (!has_xpu()) {
    return;
  }
  std::optional<c10::xpu::XPUStream> s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  c10::xpu::XPUStream cur_stream = c10::xpu::getCurrentXPUStream();

  EXPECT_TRUE(s0);
  EXPECT_TRUE(s1);
  EXPECT_NE(cur_stream, s0);
  EXPECT_NE(cur_stream, s1);
  EXPECT_NE(s0, s1);
}

// Ensure queue pool round-robin fashion
TEST(XPUStreamTest, StreamPoolRoundRobinTest) {
  if (!has_xpu()) {
    return;
  }

  std::vector<c10::xpu::XPUStream> streams{};
  for ([[maybe_unused]] const auto _ : c10::irange(200)) {
    streams.emplace_back(c10::xpu::getStreamFromPool());
  }

  std::unordered_set<sycl::queue> queue_set{};
  bool hasDuplicates = false;
  for (const auto i : c10::irange(streams.size())) {
    auto& queue = streams[i].queue();
    auto result_pair = queue_set.insert(queue);
    if (!result_pair.second) { // already existed
      hasDuplicates = true;
    } else { // newly inserted
      EXPECT_TRUE(!hasDuplicates);
    }
  }
  EXPECT_TRUE(hasDuplicates);

  auto stream = c10::xpu::getStreamFromPool(/* isHighPriority */ true);
  auto result_pair = queue_set.insert(stream.queue());
  EXPECT_TRUE(result_pair.second);
}

static void asyncMemCopy(
    sycl::queue& queue,
    int* dst,
    int* src,
    size_t numBytes) {
  queue.memcpy(dst, src, numBytes);
}

TEST(XPUStreamTest, StreamFunction) {
  if (!has_xpu()) {
    return;
  }

  constexpr int numel = 1024;
  // NOLINTNEXTLINE(*-c-arrays)
  int hostData[numel];
  initHostData(hostData, numel);

  auto stream = c10::xpu::getStreamFromPool();
  EXPECT_TRUE(stream.query());
  int* deviceData = sycl::malloc_device<int>(numel, stream);

  // H2D
  asyncMemCopy(stream, deviceData, hostData, sizeof(int) * numel);
  c10::xpu::syncStreamsOnDevice();
  EXPECT_TRUE(stream.query());

  clearHostData(hostData, numel);

  // D2H
  asyncMemCopy(stream, hostData, deviceData, sizeof(int) * numel);
  c10::xpu::syncStreamsOnDevice();

  validateHostData(hostData, numel);

  stream = c10::xpu::getStreamFromPool(-1);

  clearHostData(hostData, numel);

  // D2H
  asyncMemCopy(stream, hostData, deviceData, sizeof(int) * numel);
  c10::xpu::syncStreamsOnDevice();

  validateHostData(hostData, numel);
  sycl::free(deviceData, c10::xpu::get_device_context());
}

// Verifies external streams can be created and used
TEST(XPUStreamTest, ExternalTest) {
  if (!has_xpu()) {
    return;
  }

  c10::DeviceGuard device_guard(c10::Device(c10::DeviceType::XPU, 0));

  using namespace sycl::ext::oneapi::property;
  sycl::queue* stream = new sycl::queue(
      c10::xpu::get_device_context(),
      c10::xpu::get_raw_device(0),
      c10::xpu::asyncHandler,
      {sycl::property::queue::in_order(), queue::priority_normal()});

  at::xpu::XPUStream myStream = at::xpu::getStreamFromExternal(stream, 0);

  at::xpu::setCurrentXPUStream(myStream);
  at::xpu::XPUStream curStream = at::xpu::getCurrentXPUStream();

  EXPECT_EQ(myStream.priority(), 0);
  ASSERT_TRUE(curStream == myStream);
  ASSERT_TRUE(&(curStream.queue()) == stream);

  sycl::queue* q_ptr = curStream;
  ASSERT_TRUE(q_ptr == stream);

  delete stream;
}

// Verifies different external streams can be used for different devices at the
// same time
TEST(XPUStreamTest, ExternalMultiDeviceTest) {
  if (!has_xpu()) {
    return;
  }
  if (c10::xpu::device_count() < 2)
    return;
  sycl::queue* stream_0 = nullptr;
  sycl::queue* stream_1 = nullptr;

  using namespace sycl::ext::oneapi::property;
  {
    c10::DeviceGuard device_guard(c10::Device(c10::DeviceType::XPU, 0));
    stream_0 = new sycl::queue(
        c10::xpu::get_device_context(),
        c10::xpu::get_raw_device(0),
        c10::xpu::asyncHandler,
        {sycl::property::queue::in_order(), queue::priority_normal()});
  }
  {
    c10::DeviceGuard device_guard(c10::Device(c10::DeviceType::XPU, 1));
    stream_1 = new sycl::queue(
        c10::xpu::get_device_context(),
        c10::xpu::get_raw_device(1),
        c10::xpu::asyncHandler,
        {sycl::property::queue::in_order(), queue::priority_normal()});
  }
  at::xpu::XPUStream myStream0 = at::xpu::getStreamFromExternal(stream_0, 0);
  at::xpu::XPUStream myStream1 = at::xpu::getStreamFromExternal(stream_1, 1);

  at::xpu::setCurrentXPUStream(myStream0);
  ASSERT_TRUE(at::xpu::getCurrentXPUStream(0) == myStream0);
  at::xpu::setCurrentXPUStream(myStream1);
  ASSERT_TRUE(at::xpu::getCurrentXPUStream(0) == myStream0);
  ASSERT_TRUE(at::xpu::getCurrentXPUStream(1) == myStream1);

  delete stream_0;
  delete stream_1;
}

TEST(XPUStreamTest, ExternalStreamDifferentPointersTest) {
  if (!has_xpu()) {
    return;
  }

  using namespace sycl::ext::oneapi::property;
  sycl::queue ext_queue = sycl::queue(
      c10::xpu::get_device_context(),
      c10::xpu::get_raw_device(0),
      c10::xpu::asyncHandler,
      {sycl::property::queue::in_order(), queue::priority_normal()});

  // Ponters to queue and its copies will lead to distinct external XPUStreams.
  auto queue_ptr1 = std::make_unique<sycl::queue>(ext_queue);
  auto queue_ptr2 = std::make_unique<sycl::queue>(ext_queue);

  at::xpu::XPUStream myStream1 =
      at::xpu::getStreamFromExternal(queue_ptr1.get(), 0);
  at::xpu::XPUStream myStream2 =
      at::xpu::getStreamFromExternal(queue_ptr2.get(), 0);

  EXPECT_NE(myStream1, myStream2);
}
