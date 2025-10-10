#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Event.h>
#include <c10/core/impl/InlineEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <c10/util/irange.h>

#include <cuda_runtime.h>

#include <functional>
#include <future>
#include <thread>
#include <unordered_set>

#define ASSERT_EQ_CUDA(X, Y) \
  {                          \
    bool isTRUE = X == Y;    \
    ASSERT_TRUE(isTRUE);     \
  }

#define ASSERT_NE_CUDA(X, Y) \
  {                          \
    bool isFALSE = X == Y;   \
    ASSERT_FALSE(isFALSE);   \
  }

/*
   Tests related to ATen streams.
   */
// Verifies streams are live through copying and moving
TEST(TestStream, CopyAndMoveTest) {
  if (!at::cuda::is_available()) return;
  int32_t device = -1;
  cudaStream_t cuda_stream;

  // Tests that copying works as expected and preserves the stream
  at::cuda::CUDAStream copyStream = at::cuda::getStreamFromPool();
  {
    auto s = at::cuda::getStreamFromPool();
    device = s.device_index();
    cuda_stream = s.stream();

    copyStream = s;

    ASSERT_EQ_CUDA(copyStream.device_index(), device);
    ASSERT_EQ_CUDA(copyStream.stream(), cuda_stream);
  }

  ASSERT_EQ_CUDA(copyStream.device_index(), device);
  ASSERT_EQ_CUDA(copyStream.stream(), cuda_stream);

  // Tests that moving works as expected and preserves the stream
  at::cuda::CUDAStream moveStream = at::cuda::getStreamFromPool();
  {
    auto s = at::cuda::getStreamFromPool();
    device = s.device_index();
    cuda_stream = s.stream();

    moveStream = std::move(s);

    ASSERT_EQ_CUDA(moveStream.device_index(), device);
    ASSERT_EQ_CUDA(moveStream.stream(), cuda_stream);
  }

  ASSERT_EQ_CUDA(moveStream.device_index(), device);
  ASSERT_EQ_CUDA(moveStream.stream(), cuda_stream);
}

// Verifies stream priority is handled properly
TEST(TestStream, StreamPriorityTest) {
  if (!at::cuda::is_available()) return;
  auto [least_priority, greatest_priority] =
      at::cuda::CUDAStream::priority_range();
  EXPECT_EQ(least_priority, 0);

  auto stream = at::cuda::getStreamFromPool(-1);
  EXPECT_EQ(stream.priority(), -1);
  EXPECT_GT(10, at::cuda::max_compile_time_stream_priorities);
  stream = at::cuda::getStreamFromPool(-10);
  EXPECT_EQ(stream.priority(), greatest_priority);
  stream = at::cuda::getStreamFromPool(0);
  EXPECT_EQ(stream.priority(), 0);
  stream = at::cuda::getStreamFromPool(10);
  EXPECT_EQ(stream.priority(), 0);
}

// Verifies streams are set properly
TEST(TestStream, GetAndSetTest) {
  if (!at::cuda::is_available()) return;
  at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();

  // Sets and gets
  at::cuda::setCurrentCUDAStream(myStream);
  at::cuda::CUDAStream curStream = at::cuda::getCurrentCUDAStream();

  ASSERT_EQ_CUDA(myStream, curStream);

  // Gets, sets, and gets default stream
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  curStream = at::cuda::getCurrentCUDAStream();

  ASSERT_NE_CUDA(defaultStream, myStream);
  ASSERT_EQ_CUDA(curStream, defaultStream);
}

void thread_fun(std::optional<at::cuda::CUDAStream>& cur_thread_stream) {
  auto new_stream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(new_stream);
  cur_thread_stream = {at::cuda::getCurrentCUDAStream()};
  ASSERT_EQ_CUDA(*cur_thread_stream, new_stream);
}

// Ensures streams are thread local
TEST(TestStream, MultithreadGetAndSetTest) {
  if (!at::cuda::is_available()) return;
  std::optional<at::cuda::CUDAStream> s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStream default_stream = at::cuda::getDefaultCUDAStream();

  ASSERT_EQ_CUDA(cur_stream, default_stream);
  ASSERT_NE_CUDA(cur_stream, *s0);
  ASSERT_NE_CUDA(cur_stream, *s1);
  ASSERT_NE_CUDA(s0, s1);
}

// CUDA Guard
TEST(TestStream, CUDAGuardTest) {
  if (!at::cuda::is_available()) return;
  if (at::cuda::getNumGPUs() < 2) {
    return;
  }

  // -- begin setup

  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);
  std::vector<at::cuda::CUDAStream> streams0 = {
      at::cuda::getDefaultCUDAStream(), at::cuda::getStreamFromPool()};
  ASSERT_EQ_CUDA(streams0[0].device_index(), 0);
  ASSERT_EQ_CUDA(streams0[1].device_index(), 0);
  at::cuda::setCurrentCUDAStream(streams0[0]);

  std::vector<at::cuda::CUDAStream> streams1;
  {
    at::cuda::CUDAGuard device_guard(1);
    streams1.push_back(at::cuda::getDefaultCUDAStream());
    streams1.push_back(at::cuda::getStreamFromPool());
  }
  ASSERT_EQ_CUDA(streams1[0].device_index(), 1);
  ASSERT_EQ_CUDA(streams1[1].device_index(), 1);
  at::cuda::setCurrentCUDAStream(streams1[0]);

  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);

  // -- end setup

  // Setting a stream changes the current device and the stream on that device
  {
    at::cuda::CUDAStreamGuard guard(streams1[1]);
    ASSERT_EQ_CUDA(guard.current_device(), at::Device(at::kCUDA, 1));
    ASSERT_EQ_CUDA(at::cuda::current_device(), 1);
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), streams1[1]);
  }

  // Device and stream are now reset
  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), streams1[0]);

  // Setting only the device changes only the current device and not the stream
  {
    at::cuda::CUDAGuard guard(/*device=*/1);
    ASSERT_EQ_CUDA(guard.current_device(), at::Device(at::kCUDA, 1));
    ASSERT_EQ_CUDA(at::cuda::current_device(), 1);
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), streams1[0]);
  }

  ASSERT_EQ_CUDA(at::cuda::current_device(), 0);
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(0), streams0[0]);
}

// Streampool Round Robin
TEST(TestStream, StreamPoolTest) {
  if (!at::cuda::is_available()) return;
  std::vector<at::cuda::CUDAStream> streams{};
  for (const auto i : c10::irange(200)) {
    (void)i;
    streams.emplace_back(at::cuda::getStreamFromPool());
  }

  std::unordered_set<cudaStream_t> stream_set{};
  bool hasDuplicates = false;
  for (const auto i: c10::irange(streams.size())) {
    cudaStream_t cuda_stream = streams[i];
    auto result_pair = stream_set.insert(cuda_stream);
    if (!result_pair.second)
      hasDuplicates = true;
  }

  ASSERT_TRUE(hasDuplicates);
}

// Multi-GPU
TEST(TestStream, MultiGPUTest) {
  if (!at::cuda::is_available()) return;
  if (at::cuda::getNumGPUs() < 2)
    return;

  at::cuda::CUDAStream s0 = at::cuda::getStreamFromPool(true, 0);
  at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, 1);

  at::cuda::setCurrentCUDAStream(s0);
  at::cuda::setCurrentCUDAStream(s1);

  ASSERT_EQ_CUDA(s0, at::cuda::getCurrentCUDAStream());

  at::cuda::CUDAGuard device_guard{1};
  ASSERT_EQ_CUDA(s1, at::cuda::getCurrentCUDAStream());
}

// CUDAEvent Syncs
TEST(TestStream, CUDAEventSyncTest) {
  if (!at::cuda::is_available()) return;
  const auto stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAEvent event;

  ASSERT_TRUE(event.query());

  event.recordOnce(stream);

  const auto wait_stream0 = at::cuda::getStreamFromPool();
  const auto wait_stream1 = at::cuda::getStreamFromPool();

  event.block(wait_stream0);
  event.block(wait_stream1);

  cudaStreamSynchronize(wait_stream0);
  ASSERT_TRUE(event.query());
}

// Cross-Device Events
TEST(TestStream, CrossDeviceTest) {
  if (!at::cuda::is_available()) return;
  if (at::cuda::getNumGPUs() < 2)
    return;

  const auto stream0 = at::cuda::getStreamFromPool();
  at::cuda::CUDAEvent event0;

  at::cuda::set_device(1);
  const auto stream1 = at::cuda::getStreamFromPool();
  at::cuda::CUDAEvent event1;

  event0.record(stream0);
  event1.record(stream1);

  event0 = std::move(event1);

  ASSERT_EQ_CUDA(event0.device(), at::Device(at::kCUDA, 1));

  event0.block(stream0);

  cudaStreamSynchronize(stream0);
  ASSERT_TRUE(event0.query());
}

// Generic Events
TEST(TestStream, GenericInlineCUDAEventTest) {
  if (!at::cuda::is_available()) return;

  c10::impl::InlineEvent<c10::cuda::impl::CUDAGuardImpl> event{c10::DeviceType::CUDA};
  c10::Stream stream = at::cuda::getStreamFromPool();

  event.record(stream);

  const c10::Stream wait_stream0 = at::cuda::getStreamFromPool();
  const c10::Stream wait_stream1 = at::cuda::getStreamFromPool();

  event.block(wait_stream0);
  event.block(wait_stream1);

  const at::cuda::CUDAStream cuda_stream{wait_stream0};
  cudaStreamSynchronize(cuda_stream);

  ASSERT_TRUE(event.query());
}

TEST(TestStream, GenericVirtualCUDAEventTest) {
  if (!at::cuda::is_available()) return;

  c10::Event event{c10::DeviceType::CUDA};
  c10::Stream stream = at::cuda::getStreamFromPool();

  event.recordOnce(stream);

  const c10::Stream wait_stream0 = at::cuda::getStreamFromPool();
  const c10::Stream wait_stream1 = at::cuda::getStreamFromPool();

  wait_stream0.wait(event);
  wait_stream1.wait(event);

  const at::cuda::CUDAStream cuda_stream{wait_stream0};
  cudaStreamSynchronize(cuda_stream);

  ASSERT_TRUE(event.query());
  ASSERT_TRUE(event.flag() == c10::EventFlag::PYTORCH_DEFAULT);
}

// Verifies external streams can be created and used
TEST(TestStream, ExternalTest) {
  if (!at::cuda::is_available())
    return;
  at::cuda::CUDAGuard device_guard(0);

  cudaStream_t cuda_stream;
  cudaStreamCreateWithPriority(&cuda_stream, cudaStreamNonBlocking, -1);

  at::cuda::CUDAStream myStream =
      at::cuda::getStreamFromExternal(cuda_stream, 0);

  at::cuda::setCurrentCUDAStream(myStream);
  at::cuda::CUDAStream curStream = at::cuda::getCurrentCUDAStream();

  ASSERT_EQ_CUDA(curStream, myStream);
  ASSERT_EQ_CUDA(curStream.stream(), cuda_stream);

  cudaStreamDestroy(cuda_stream);
}

// Verifies different external streams can be used for different devices at the
// same time
TEST(TestStream, ExternalMultiDeviceTest) {
  if (!at::cuda::is_available())
    return;
  if (at::cuda::getNumGPUs() < 2)
    return;
  cudaStream_t cuda_stream_0;
  cudaStream_t cuda_stream_1;
  {
    at::cuda::CUDAGuard device_guard(0);
    cudaStreamCreateWithPriority(&cuda_stream_0, cudaStreamNonBlocking, -1);
  }
  {
    at::cuda::CUDAGuard device_guard(1);
    cudaStreamCreateWithPriority(&cuda_stream_1, cudaStreamNonBlocking, -1);
  }
  at::cuda::CUDAStream myStream0 =
      at::cuda::getStreamFromExternal(cuda_stream_0, 0);
  at::cuda::CUDAStream myStream1 =
      at::cuda::getStreamFromExternal(cuda_stream_1, 1);

  at::cuda::setCurrentCUDAStream(myStream0);
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(0), myStream0);
  at::cuda::setCurrentCUDAStream(myStream1);
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(0), myStream0);
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(1), myStream1);

  cudaStreamDestroy(cuda_stream_0);
  cudaStreamDestroy(cuda_stream_1);
}

// Verifies external streams work with guards, even nested ones
TEST(TestStream, ExternalGuardTest) {
  if (!at::cuda::is_available())
    return;
  at::cuda::CUDAGuard device_guard(0);

  cudaStream_t a_cuda_stream;
  cudaStream_t another_cuda_stream;
  cudaStreamCreateWithPriority(&a_cuda_stream, cudaStreamNonBlocking, -1);
  cudaStreamCreateWithPriority(&another_cuda_stream, cudaStreamNonBlocking, -1);
  at::cuda::CUDAStream myFirstStream =
      at::cuda::getStreamFromExternal(a_cuda_stream, 0);
  at::cuda::CUDAStream mySecondStream =
      at::cuda::getStreamFromExternal(another_cuda_stream, 0);

  at::cuda::CUDAStream originalStream = at::cuda::getCurrentCUDAStream();
  {
    at::cuda::CUDAStreamGuard outerGuard(myFirstStream);
    ASSERT_EQ_CUDA(outerGuard.original_stream(), originalStream);
    ASSERT_EQ_CUDA(outerGuard.current_stream(), myFirstStream);
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), myFirstStream);
    {
      at::cuda::CUDAStreamGuard innerGuard(mySecondStream);
      ASSERT_EQ_CUDA(innerGuard.original_stream(), myFirstStream);
      ASSERT_EQ_CUDA(innerGuard.current_stream(), mySecondStream);
      ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), mySecondStream);
    }
    ASSERT_EQ_CUDA(outerGuard.original_stream(), originalStream);
    ASSERT_EQ_CUDA(outerGuard.current_stream(), myFirstStream);
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), myFirstStream);
    outerGuard.reset_stream(mySecondStream);
    ASSERT_EQ_CUDA(outerGuard.original_stream(), originalStream);
    ASSERT_EQ_CUDA(outerGuard.current_stream(), mySecondStream);
    ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), mySecondStream);
  }
  ASSERT_EQ_CUDA(at::cuda::getCurrentCUDAStream(), originalStream);

  cudaStreamDestroy(a_cuda_stream);
  cudaStreamDestroy(another_cuda_stream);
}

// Verifies that different threads stage their external streams to different
// places in memory and thus don't interfere
TEST(TestStream, ExternalMultiThreadTest) {
  if (!at::cuda::is_available())
    return;
  at::cuda::CUDAGuard device_guard(0);

  cudaStream_t cuda_stream_a;
  cudaStream_t cuda_stream_b;
  cudaStreamCreateWithPriority(&cuda_stream_a, cudaStreamNonBlocking, -1);
  cudaStreamCreateWithPriority(&cuda_stream_b, cudaStreamNonBlocking, -1);
  at::cuda::CUDAStream myStreamA =
      at::cuda::getStreamFromExternal(cuda_stream_a, 0);
  at::cuda::CUDAStream myStreamB =
      at::cuda::getStreamFromExternal(cuda_stream_b, 0);

  std::promise<void> aToBProm;
  std::promise<void> bToAProm;
  std::optional<at::cuda::CUDAStream> foundStream;

  std::thread threadA([&]() {
    at::cuda::CUDAGuard device_guard(0);
    at::cuda::setCurrentCUDAStream(myStreamA);
    aToBProm.set_value();
    bToAProm.get_future().wait();
    foundStream = at::cuda::getCurrentCUDAStream();
  });

  std::thread threadB([&]() {
    at::cuda::CUDAGuard device_guard(0);
    aToBProm.get_future().wait();
    at::cuda::setCurrentCUDAStream(myStreamB);
    bToAProm.set_value();
  });

  threadA.join();
  threadB.join();

  ASSERT_EQ_CUDA(*foundStream, myStreamA);

  cudaStreamDestroy(cuda_stream_a);
  cudaStreamDestroy(cuda_stream_b);
}
