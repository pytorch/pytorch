#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAGuard.h"

#include "cuda_runtime.h"

#include <functional>
#include <thread>

/*
Tests related to ATen streams.
*/
TEST_CASE(
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

    REQUIRE(copyStream.internals() == s.internals());
    REQUIRE(copyStream.device() == device);
    REQUIRE(copyStream.stream() == cuda_stream);
  }

  REQUIRE(copyStream.internals());
  REQUIRE(copyStream.device() == device);
  REQUIRE(copyStream.stream() == cuda_stream);

  // Tests that moving works as expected and preserves the stream
  at::cuda::CUDAStream moveStream;
  {
    auto s = at::cuda::createCUDAStream();
    device = s.device();
    cuda_stream = s.stream();

    moveStream = std::move(s);

    REQUIRE(moveStream.device() == device);
    REQUIRE(moveStream.stream() == cuda_stream);
  }

  REQUIRE(moveStream.internals());
  REQUIRE(moveStream.device() == device);
  REQUIRE(moveStream.stream() == cuda_stream);
}

TEST_CASE("Getting and Setting Streams", "Verifies streams are set properly") {
  at::cuda::CUDAStream myStream = at::cuda::createCUDAStream();

  // Sets and gets
  at::cuda::setCurrentCUDAStream(myStream);
  at::cuda::CUDAStream curStream = at::cuda::getCurrentCUDAStream();

  REQUIRE(myStream == curStream);

  // Gets, sets, and gets default stream
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  curStream = at::cuda::getCurrentCUDAStream();

  REQUIRE(defaultStream != myStream);
  REQUIRE(curStream == defaultStream);
}

TEST_CASE("Stream API retain/free", "Ensures streams are destroyed properly") {
  auto ptr = at::cuda::detail::CUDAStream_createAndRetainWithOptions(
      at::cuda::CUDAStream::DEFAULT_FLAGS, at::cuda::CUDAStream::DEFAULT_PRIORITY);

  at::cuda::detail::CUDAStream_free(ptr);
  REQUIRE(ptr == nullptr);
}

void thread_fun(at::cuda::CUDAStream& cur_thread_stream) {
  auto new_stream = at::cuda::createCUDAStream();
  at::cuda::setCurrentCUDAStream(new_stream);
  cur_thread_stream = at::cuda::getCurrentCUDAStream();
  REQUIRE(cur_thread_stream == new_stream);
}

TEST_CASE(
    "Multithread Getting and Setting",
    "Ensures streams are thread local") {
  at::cuda::CUDAStream s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStream default_stream = at::cuda::getDefaultCUDAStream();

  REQUIRE(cur_stream == default_stream);
  REQUIRE(cur_stream != s0);
  REQUIRE(cur_stream != s1);
  REQUIRE(s0 != s1);
}

TEST_CASE("CUDAGuard") {
  if (at::cuda::getNumGPUs() < 2) {
    return;
  }

  // -- begin setup

  REQUIRE(at::cuda::current_device() == 0);
  std::vector<at::cuda::CUDAStream> streams0 = {
      at::cuda::getDefaultCUDAStream(),
      at::cuda::createCUDAStream()};
  REQUIRE(streams0[0].device() == 0);
  REQUIRE(streams0[1].device() == 0);
  at::cuda::setCurrentCUDAStreamOnDevice(0, streams0[0]);

  std::vector<at::cuda::CUDAStream> streams1;
  {
    at::DeviceGuard device_guard(1);
    streams1.push_back(at::cuda::getDefaultCUDAStream());
    streams1.push_back(at::cuda::createCUDAStream());
  }
  REQUIRE(streams1[0].device() == 1);
  REQUIRE(streams1[1].device() == 1);
  at::cuda::setCurrentCUDAStreamOnDevice(1, streams1[0]);

  REQUIRE(at::cuda::current_device() == 0);

  // -- end setup

  // Test that all original streams are recorded.
  {
    at::cuda::CUDAGuard guard;
    REQUIRE(guard.original_streams().empty());
    guard.set_stream(streams0[0]);
    REQUIRE(
        guard.original_streams().size() == at::cuda::getNumGPUs());
    REQUIRE(guard.original_streams()[0] == streams0[0]);
    REQUIRE(guard.original_streams()[1] == streams1[0]);
  }

  // Setting a stream changes the current device and the stream on that device
  {
    at::cuda::CUDAGuard guard(streams1[1]);
    REQUIRE(guard.last_device() == 1);
    REQUIRE(at::cuda::current_device() == 1);
    REQUIRE(at::cuda::getCurrentCUDAStreamOnDevice(1) == streams1[1]);
  }

  // Device and stream are now reset
  REQUIRE(at::cuda::current_device() == 0);
  REQUIRE(at::cuda::getCurrentCUDAStreamOnDevice(1) == streams1[0]);

  // Setting only the device changes only the current device and not the stream
  {
    at::cuda::CUDAGuard guard(/*device=*/1);
    REQUIRE(guard.last_device() == 1);
    REQUIRE(at::cuda::current_device() == 1);
    REQUIRE(at::cuda::getCurrentCUDAStreamOnDevice(1) == streams1[0]);
  }

  REQUIRE(at::cuda::current_device() == 0);
  REQUIRE(at::cuda::getCurrentCUDAStreamOnDevice(0) == streams0[0]);

  // Setting the stream first, and then the device, first changes the devices
  // back, and then resets the stream on the initial device.

  {
    at::cuda::CUDAGuard guard(streams0[1]);
    guard.set_device(1);
  }

  REQUIRE(at::cuda::current_device() == 0);
  REQUIRE(at::cuda::getCurrentCUDAStreamOnDevice(0) == streams0[0]);
  REQUIRE(at::cuda::getCurrentCUDAStreamOnDevice(1) == streams1[0]);
}

TEST_CASE("CUDAGuardIsMovable") {
  if (at::cuda::getNumGPUs() < 2) {
    return;
  }
  const auto stream = at::cuda::createCUDAStream();
  const auto device_count = at::cuda::getNumGPUs();
  at::cuda::CUDAGuard first(stream);
  first.set_device(1);
  at::cuda::CUDAGuard second(std::move(first));
  REQUIRE(second.original_streams().size() == device_count);
  REQUIRE(second.original_device() == 0);
  REQUIRE(second.last_device() == 1);
  at::cuda::CUDAGuard third;
  third = std::move(second);
  REQUIRE(third.original_streams().size() == device_count);
  REQUIRE(third.original_device() == 0);
  REQUIRE(third.last_device() == 1);
}
