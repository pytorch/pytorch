#include <gtest/gtest.h>

#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUStream.h>

#include <thread>
#include <unordered_set>

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
  c10::xpu::XPUStream copyStream = c10::xpu::getStreamFromPool();
  {
    auto s = c10::xpu::getStreamFromPool();
    device = s.device_index();
    queue = s.queue();

    copyStream = s;

    ASSERT_EQ_XPU(copyStream.device_index(), device);
    ASSERT_EQ_XPU(copyStream.queue(), queue);
  }

  ASSERT_EQ_XPU(copyStream.device_index(), device);
  ASSERT_EQ_XPU(copyStream.queue(), queue);

  // Tests that moving works as expected and preserves the stream
  c10::xpu::XPUStream moveStream = c10::xpu::getStreamFromPool();
  {
    auto s = c10::xpu::getStreamFromPool();
    device = s.device_index();
    queue = s.queue();

    moveStream = std::move(s);

    ASSERT_EQ_XPU(moveStream.device_index(), device);
    ASSERT_EQ_XPU(moveStream.queue(), queue);
  }

  ASSERT_EQ_XPU(moveStream.device_index(), device);
  ASSERT_EQ_XPU(moveStream.queue(), queue);
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

  if (c10::xpu::device_count() <= 1) {
    return;
  }

  c10::xpu::set_device(0);
  stream = c10::xpu::getStreamFromPool(false, 1);
  ASSERT_EQ_XPU(stream.device_index(), 1);
  ASSERT_NE_XPU(stream.device_index(), c10::xpu::current_device());
}

void thread_fun(c10::optional<c10::xpu::XPUStream>& cur_thread_stream) {
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

  c10::xpu::XPUStream cur_stream = c10::xpu::getCurrentXPUStream();

  ASSERT_NE_XPU(cur_stream, *s0);
  ASSERT_NE_XPU(cur_stream, *s1);
  ASSERT_NE_XPU(s0, s1);
}

// Ensure queue pool round-robin fashion
TEST(XPUStreamTest, StreamPoolRoundRobinTest) {
  if (!has_xpu()) {
    return;
  }

  std::vector<c10::xpu::XPUStream> streams{};
  for (C10_UNUSED const auto _ : c10::irange(200)) {
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
      ASSERT_TRUE(!hasDuplicates);
    }
  }
  ASSERT_TRUE(hasDuplicates);
}

void asyncMemCopy(sycl::queue& queue, int* dst, int* src, size_t numBytes) {
  queue.memcpy(dst, src, numBytes);
}

TEST(XPUStreamTest, StreamFunction) {
  if (!has_xpu()) {
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
  asyncMemCopy(stream, deviceData, hostData, sizeof(int) * numel);
  c10::xpu::device_synchronize();

  for (const auto i : c10::irange(numel)) {
    hostData[i] = 0;
  }

  // D2H
  asyncMemCopy(stream, hostData, deviceData, sizeof(int) * numel);
  c10::xpu::device_synchronize();

  for (const auto i : c10::irange(numel)) {
    ASSERT_EQ_XPU(hostData[i], i);
  }
  sycl::free(deviceData, c10::xpu::get_device_context());
}
