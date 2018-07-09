#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"

#include "cuda_runtime.h"

#include <thread>
#include <functional>

/*
Tests related to ATen streams.
*/
TEST_CASE("Copying and Moving Streams", "Verifies streams are live through copying and moving") {
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
    at::CUDAStream::DEFAULT_FLAGS
  , at::CUDAStream::DEFAULT_PRIORITY);

  at::detail::CUDAStream_free(ptr);
  REQUIRE(ptr == nullptr);
}

void thread_fun(at::CUDAStream& cur_thread_stream) {
  auto new_stream = at::globalContext().createCUDAStream();
  at::globalContext().setCurrentCUDAStream(new_stream);
  cur_thread_stream = at::globalContext().getCurrentCUDAStream();
  REQUIRE(cur_thread_stream == new_stream);
}

TEST_CASE("Multithread Getting and Setting", "Ensures streams are thread local") {
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
