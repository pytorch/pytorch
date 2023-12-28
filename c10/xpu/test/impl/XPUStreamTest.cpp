#include <gtest/gtest.h>

#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUStream.h>

#include <thread>

#define ASSERT_EQ_XPU(X, Y) \
  {                         \
    bool _isEQ = X == Y;    \
    ASSERT_TRUE(_isEQ);     \
  }

#define ASSERT_NE_XPU(X, Y) \
  {                         \
    bool isNE = X == Y;     \
    ASSERT_FALSE(isNE);     \
  }

bool has_xpu() {
  return c10::xpu::device_count() > 0;
}

TEST(XPUStreamTest, CopyAndMoveTest) {
  if (!has_xpu()) {
    return;
  }

  int32_t device = -1;
  sycl::queue queue;
  uint8_t queue_index;
  c10::xpu::XPUStream copyStream = c10::xpu::getStreamFromPool();
  {
    auto s = c10::xpu::getStreamFromPool();
    device = s.device_index();
    queue = s.queue();
    queue_index = s.queue_index();

    copyStream = s;

    ASSERT_EQ_XPU(copyStream.device_index(), device);
    ASSERT_EQ_XPU(copyStream.queue(), queue);
    ASSERT_EQ_XPU(copyStream.queue_index(), queue_index);
  }

  ASSERT_EQ_XPU(copyStream.device_index(), device);
  ASSERT_EQ_XPU(copyStream.queue(), queue);
  ASSERT_EQ_XPU(copyStream.queue_index(), queue_index);

  // Tests that moving works as expected and preserves the stream
  c10::xpu::XPUStream moveStream = c10::xpu::getStreamFromPool();
  {
    auto s = c10::xpu::getStreamFromPool();
    device = s.device_index();
    queue = s.queue();
    queue_index = s.queue_index();

    moveStream = std::move(s);

    ASSERT_EQ_XPU(moveStream.device_index(), device);
    ASSERT_EQ_XPU(moveStream.queue(), queue);
    ASSERT_EQ_XPU(moveStream.queue_index(), queue_index);
  }

  ASSERT_EQ_XPU(moveStream.device_index(), device);
  ASSERT_EQ_XPU(moveStream.queue(), queue);
  ASSERT_EQ_XPU(moveStream.queue_index(), queue_index);
}

TEST(XPUStreamTest, StreamBehavior) {
  if (!has_xpu()) {
    return;
  }

  c10::xpu::XPUStream stream = c10::xpu::getStreamFromPool();
  ASSERT_EQ_XPU(stream.device_type(), c10::kXPU);
  c10::xpu::setCurrentXPUStream(stream);
  c10::xpu::XPUStream cur_stream = c10::xpu::getCurrentXPUStream();

  ASSERT_EQ_XPU(cur_stream, stream);

  ASSERT_THROW(
      c10::xpu::getStreamFromPool(/* isHighPriority= */ true), c10::Error);

  if (c10::xpu::device_count() <= 1) {
    return;
  }

  c10::xpu::set_device(0);
  stream = c10::xpu::getStreamFromPool(false, 1);
  ASSERT_EQ_XPU(stream.device_index(), 1);
  ASSERT_NE_XPU(stream.device_index(), c10::xpu::current_device());
}

void thread_fun(c10::optional<at::xpu::XPUStream>& cur_thread_stream) {
  auto new_stream = c10::xpu::getStreamFromPool();
  c10::xpu::setCurrentXPUStream(new_stream);
  cur_thread_stream = {c10::xpu::getCurrentXPUStream()};
  ASSERT_EQ_XPU(*cur_thread_stream, new_stream);
}

// Ensures streams are thread local
TEST(XPUStreamTest, MultithreadStreamBehavior) {
  if (!has_xpu()) {
    return;
  }
  c10::optional<c10::xpu::XPUStream> s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  c10::xpu::XPUStream cur_stream = at::xpu::getCurrentXPUStream();

  ASSERT_NE_XPU(cur_stream, *s0);
  ASSERT_NE_XPU(cur_stream, *s1);
  ASSERT_NE_XPU(s0, s1);
}

// Ensure queue pool round-robin fashion
TEST(XPUStreamTest, StreamPoolRoundRobinTest) {
  if (!has_xpu()) {
    return;
  }

  auto queue_index = c10::xpu::getStreamFromPool().queue_index();

  for (C10_UNUSED const auto _ : c10::irange(200)) {
    auto next_queue_index = c10::xpu::getStreamFromPool().queue_index();
    ASSERT_EQ_XPU(
        (queue_index + 1) % c10::xpu::kStreamsPerPool, next_queue_index);
    queue_index = next_queue_index;
  }
}
