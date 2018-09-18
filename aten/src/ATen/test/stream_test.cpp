#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAGuard.h"
#include "ATen/cuda/CUDAEvent.h"

#include "cuda_runtime.h"

#include <functional>
#include <thread>
#include <unordered_set>

/*
Tests related to ATen streams.
*/
CATCH_TEST_CASE(
    "Copying and Moving Streams",
    "Verifies streams are live through copying and moving") {
  int32_t device = -1;
  cudaStream_t cuda_stream;

  // Tests that copying works as expected and preserves the stream
  at::cuda::CUDAStream copyStream;
  {
    auto s = at::cuda::createCUDAStream();
    device = s.device();
    cuda_stream = s.stream();

    copyStream = s;

    CATCH_REQUIRE(copyStream.internals() == s.internals());
    CATCH_REQUIRE(copyStream.device() == device);
    CATCH_REQUIRE(copyStream.stream() == cuda_stream);
  }

  CATCH_REQUIRE(copyStream.internals());
  CATCH_REQUIRE(copyStream.device() == device);
  CATCH_REQUIRE(copyStream.stream() == cuda_stream);

  // Tests that moving works as expected and preserves the stream
  at::cuda::CUDAStream moveStream;
  {
    auto s = at::cuda::createCUDAStream();
    device = s.device();
    cuda_stream = s.stream();

    moveStream = std::move(s);

    CATCH_REQUIRE(moveStream.device() == device);
    CATCH_REQUIRE(moveStream.stream() == cuda_stream);
  }

  CATCH_REQUIRE(moveStream.internals());
  CATCH_REQUIRE(moveStream.device() == device);
  CATCH_REQUIRE(moveStream.stream() == cuda_stream);
}

CATCH_TEST_CASE("Getting and Setting Streams", "Verifies streams are set properly") {
  at::cuda::CUDAStream myStream = at::cuda::createCUDAStream();

  // Sets and gets
  at::cuda::setCurrentCUDAStream(myStream);
  at::cuda::CUDAStream curStream = at::cuda::getCurrentCUDAStream();

  CATCH_REQUIRE(myStream == curStream);

  // Gets, sets, and gets default stream
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  curStream = at::cuda::getCurrentCUDAStream();

  CATCH_REQUIRE(defaultStream != myStream);
  CATCH_REQUIRE(curStream == defaultStream);
}

void thread_fun(at::cuda::CUDAStream& cur_thread_stream) {
  auto new_stream = at::cuda::createCUDAStream();
  at::cuda::setCurrentCUDAStream(new_stream);
  cur_thread_stream = at::cuda::getCurrentCUDAStream();
  CATCH_REQUIRE(cur_thread_stream == new_stream);
}

CATCH_TEST_CASE(
    "Multithread Getting and Setting",
    "Ensures streams are thread local") {
  at::cuda::CUDAStream s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStream default_stream = at::cuda::getDefaultCUDAStream();

  CATCH_REQUIRE(cur_stream == default_stream);
  CATCH_REQUIRE(cur_stream != s0);
  CATCH_REQUIRE(cur_stream != s1);
  CATCH_REQUIRE(s0 != s1);
}

CATCH_TEST_CASE("CUDAGuard") {
  if (at::cuda::getNumGPUs() < 2) {
    return;
  }

  // -- begin setup

  CATCH_REQUIRE(at::cuda::current_device() == 0);
  std::vector<at::cuda::CUDAStream> streams0 = {
      at::cuda::getDefaultCUDAStream(),
      at::cuda::createCUDAStream()};
  CATCH_REQUIRE(streams0[0].device() == 0);
  CATCH_REQUIRE(streams0[1].device() == 0);
  at::cuda::setCurrentCUDAStream(streams0[0]);

  std::vector<at::cuda::CUDAStream> streams1;
  {
    at::DeviceGuard device_guard(1);
    streams1.push_back(at::cuda::getDefaultCUDAStream());
    streams1.push_back(at::cuda::createCUDAStream());
  }
  CATCH_REQUIRE(streams1[0].device() == 1);
  CATCH_REQUIRE(streams1[1].device() == 1);
  at::cuda::setCurrentCUDAStream(streams1[0]);

  CATCH_REQUIRE(at::cuda::current_device() == 0);

  // -- end setup

  // Test that all original streams are recorded.
  {
    at::cuda::CUDAGuard guard;
    CATCH_REQUIRE(guard.original_streams().empty());
    guard.set_stream(streams0[0]);
    CATCH_REQUIRE(
        guard.original_streams().size() == at::cuda::getNumGPUs());
    CATCH_REQUIRE(guard.original_streams()[0] == streams0[0]);
    CATCH_REQUIRE(guard.original_streams()[1] == streams1[0]);
  }

  // Setting a stream changes the current device and the stream on that device
  {
    at::cuda::CUDAGuard guard(streams1[1]);
    CATCH_REQUIRE(guard.last_device() == 1);
    CATCH_REQUIRE(at::cuda::current_device() == 1);
    CATCH_REQUIRE(at::cuda::getCurrentCUDAStream(1) == streams1[1]);
  }

  // Device and stream are now reset
  CATCH_REQUIRE(at::cuda::current_device() == 0);
  CATCH_REQUIRE(at::cuda::getCurrentCUDAStream(1) == streams1[0]);

  // Setting only the device changes only the current device and not the stream
  {
    at::cuda::CUDAGuard guard(/*device=*/1);
    CATCH_REQUIRE(guard.last_device() == 1);
    CATCH_REQUIRE(at::cuda::current_device() == 1);
    CATCH_REQUIRE(at::cuda::getCurrentCUDAStream(1) == streams1[0]);
  }

  CATCH_REQUIRE(at::cuda::current_device() == 0);
  CATCH_REQUIRE(at::cuda::getCurrentCUDAStream(0) == streams0[0]);

  // Setting the stream first, and then the device, first changes the devices
  // back, and then resets the stream on the initial device.

  {
    at::cuda::CUDAGuard guard(streams0[1]);
    guard.set_device(1);
  }

  CATCH_REQUIRE(at::cuda::current_device() == 0);
  CATCH_REQUIRE(at::cuda::getCurrentCUDAStream(0) == streams0[0]);
  CATCH_REQUIRE(at::cuda::getCurrentCUDAStream(1) == streams1[0]);
}

CATCH_TEST_CASE("CUDAGuardIsMovable") {
  if (at::cuda::getNumGPUs() < 2) {
    return;
  }
  const auto stream = at::cuda::createCUDAStream();
  const auto device_count = at::cuda::getNumGPUs();
  at::cuda::CUDAGuard first(stream);
  first.set_device(1);
  at::cuda::CUDAGuard second(std::move(first));
  CATCH_REQUIRE(second.original_streams().size() == device_count);
  CATCH_REQUIRE(second.original_device() == 0);
  CATCH_REQUIRE(second.last_device() == 1);
  at::cuda::CUDAGuard third;
  third = std::move(second);
  CATCH_REQUIRE(third.original_streams().size() == device_count);
  CATCH_REQUIRE(third.original_device() == 0);
  CATCH_REQUIRE(third.last_device() == 1);
}

CATCH_TEST_CASE("Streampool Round Robin") {
  std::vector<at::cuda::CUDAStream> streams{};
  for (int i = 0; i < 200; ++i) {
    streams.emplace_back(at::cuda::detail::CUDAStream_createStream());
  }

  std::unordered_set<cudaStream_t> stream_set{};
  bool hasDuplicates = false;
  for (auto i = decltype(streams.size()){0}; i < streams.size(); ++i) {
    cudaStream_t cuda_stream = streams[i];
    auto result_pair = stream_set.insert(cuda_stream);
    if (!result_pair.second) hasDuplicates = true;
  }

  CATCH_REQUIRE(hasDuplicates);
}

CATCH_TEST_CASE("Multi-GPU") {
  if (at::cuda::getNumGPUs() < 2) return;

  at::cuda::CUDAStream s0 = at::cuda::createCUDAStream(true, 0);
  at::cuda::CUDAStream s1 = at::cuda::createCUDAStream(false, 1);

  at::cuda::setCurrentCUDAStream(s0);
  at::cuda::setCurrentCUDAStream(s1);

  CATCH_REQUIRE(s0 == at::cuda::getCurrentCUDAStream());

  at::DeviceGuard device_guard{1};
  CATCH_REQUIRE(s1 == at::cuda::getCurrentCUDAStream());
}

CATCH_TEST_CASE("CUDAEvent Syncs") {
  const auto stream = at::cuda::createCUDAStream();
  at::cuda::CUDAEvent event;

  CATCH_REQUIRE(!event.happened());

  event.recordOnce(stream);

  const auto wait_stream0 = at::cuda::createCUDAStream();
  const auto wait_stream1 = at::cuda::createCUDAStream();

  wait_stream0.synchronize_with(event);
  wait_stream1.synchronize_with(event);

  cudaStreamSynchronize(wait_stream0);
  CATCH_REQUIRE(event.happened());
}

CATCH_TEST_CASE("Cross-Device Events") {
  if (at::cuda::getNumGPUs() < 2) return;

  const auto stream0 = at::cuda::createCUDAStream();
  at::cuda::CUDAEvent event0;

  at::cuda::set_device(1);
  const auto stream1 = at::cuda::createCUDAStream();
  at::cuda::CUDAEvent event1;

  event0.record(stream0);
  event1.record(stream1);
  
  event0 = std::move(event1);
  
  CATCH_REQUIRE(event0.device() == 1);

  stream0.synchronize_with(event0);
  
  cudaStreamSynchronize(stream0);
  CATCH_REQUIRE(event0.happened());
}
