#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"

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
  at::CUDAStream copyStream;
  {
    auto s = at::globalContext().createCUDAStream();
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
  at::CUDAStream moveStream;
  {
    auto s = at::globalContext().createCUDAStream();
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
  at::CUDAStream myStream = at::globalContext().createCUDAStream();

  // Sets and gets
  at::globalContext().setCurrentCUDAStream(myStream);
  at::CUDAStream curStream = at::globalContext().getCurrentCUDAStream();

  REQUIRE(myStream == curStream);

  // Gets, sets, and gets default stream
  at::CUDAStream defaultStream = at::globalContext().getDefaultCUDAStream();
  at::globalContext().setCurrentCUDAStream(defaultStream);
  curStream = at::globalContext().getCurrentCUDAStream();

  REQUIRE(defaultStream != myStream);
  REQUIRE(curStream == defaultStream);
}

TEST_CASE("Stream API retain/free", "Ensures streams are destroyed properly") {
  auto ptr = at::detail::CUDAStream_createAndRetainWithOptions(
      at::CUDAStream::DEFAULT_FLAGS, at::CUDAStream::DEFAULT_PRIORITY);

  at::detail::CUDAStream_free(ptr);
  REQUIRE(ptr == nullptr);
}

void thread_fun(at::CUDAStream& cur_thread_stream) {
  auto new_stream = at::globalContext().createCUDAStream();
  at::globalContext().setCurrentCUDAStream(new_stream);
  cur_thread_stream = at::globalContext().getCurrentCUDAStream();
  REQUIRE(cur_thread_stream == new_stream);
}

TEST_CASE(
    "Multithread Getting and Setting",
    "Ensures streams are thread local") {
  at::CUDAStream s0, s1;

  std::thread t0{thread_fun, std::ref(s0)};
  std::thread t1{thread_fun, std::ref(s1)};
  t0.join();
  t1.join();

  at::CUDAStream cur_stream = at::globalContext().getCurrentCUDAStream();
  at::CUDAStream default_stream = at::globalContext().getDefaultCUDAStream();

  REQUIRE(cur_stream == default_stream);
  REQUIRE(cur_stream != s0);
  REQUIRE(cur_stream != s1);
  REQUIRE(s0 != s1);
}

TEST_CASE("CUDAGuard") {
  if (at::globalContext().getNumGPUs() < 2) {
    return;
  }

  // -- begin setup

  REQUIRE(at::current_device() == 0);
  std::vector<at::CUDAStream> streams0 = {
      at::globalContext().getDefaultCUDAStream(),
      at::globalContext().createCUDAStream()};
  REQUIRE(streams0[0].device() == 0);
  REQUIRE(streams0[1].device() == 0);
  at::globalContext().setCurrentCUDAStreamOnDevice(0, streams0[0]);

  std::vector<at::CUDAStream> streams1;
  {
    at::DeviceGuard device_guard(1);
    streams1.push_back(at::globalContext().getDefaultCUDAStream());
    streams1.push_back(at::globalContext().createCUDAStream());
  }
  REQUIRE(streams1[0].device() == 1);
  REQUIRE(streams1[1].device() == 1);
  at::globalContext().setCurrentCUDAStreamOnDevice(1, streams1[0]);

  REQUIRE(at::current_device() == 0);

  // -- end setup

  // Test that all original streams are recorded.
  {
    at::CUDAGuard guard;
    REQUIRE(guard.original_streams().empty());
    guard.set_stream(streams0[0]);
    REQUIRE(
        guard.original_streams().size() == at::globalContext().getNumGPUs());
    REQUIRE(guard.original_streams()[0] == streams0[0]);
    REQUIRE(guard.original_streams()[1] == streams1[0]);
  }

  // Setting a stream changes the current device and the stream on that device
  {
    at::CUDAGuard guard(streams1[1]);
    REQUIRE(guard.last_device() == 1);
    REQUIRE(at::current_device() == 1);
    REQUIRE(at::globalContext().getCurrentCUDAStreamOnDevice(1) == streams1[1]);
  }

  // Device and stream are now reset
  REQUIRE(at::current_device() == 0);
  REQUIRE(at::globalContext().getCurrentCUDAStreamOnDevice(1) == streams1[0]);

  // Setting only the device changes only the current device and not the stream
  {
    at::CUDAGuard guard(/*device=*/1);
    REQUIRE(guard.last_device() == 1);
    REQUIRE(at::current_device() == 1);
    REQUIRE(at::globalContext().getCurrentCUDAStreamOnDevice(1) == streams1[0]);
  }

  REQUIRE(at::current_device() == 0);
  REQUIRE(at::globalContext().getCurrentCUDAStreamOnDevice(0) == streams0[0]);

  // Setting the stream first, and then the device, first changes the devices
  // back, and then resets the stream on the initial device.

  {
    at::CUDAGuard guard(streams0[1]);
    guard.set_device(1);
  }

  REQUIRE(at::current_device() == 0);
  REQUIRE(at::globalContext().getCurrentCUDAStreamOnDevice(0) == streams0[0]);
  REQUIRE(at::globalContext().getCurrentCUDAStreamOnDevice(1) == streams1[0]);
}

TEST_CASE("CUDAGuardIsMovable") {
  if (at::globalContext().getNumGPUs() < 2) {
    return;
  }
  const auto stream = at::globalContext().createCUDAStream();
  const auto device_count = at::globalContext().getNumGPUs();
  at::CUDAGuard first(stream);
  first.set_device(1);
  at::CUDAGuard second(std::move(first));
  REQUIRE(second.original_streams().size() == device_count);
  REQUIRE(second.original_device() == 0);
  REQUIRE(second.last_device() == 1);
  at::CUDAGuard third;
  third = std::move(second);
  REQUIRE(third.original_streams().size() == device_count);
  REQUIRE(third.original_device() == 0);
  REQUIRE(third.last_device() == 1);
}
