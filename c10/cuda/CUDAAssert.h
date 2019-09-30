#pragma once

#include <cstdint>
#include <mutex>

namespace c10 {
namespace cuda {

constexpr size_t C10_ASSERT_BUFFER_SIZE = 2048;
constexpr size_t C10_ASSERT_ARG_ALIGN_SIZE = sizeof(double);

// This class holds the assert state associated with a single CUDA stream.
// The `error` field is set to a non-zero value with a CAS operation in the
// kernel when a C10_KERNEL_ASSERT() fails. A device sync should be executed
// prior to accessing the assert report in `buffer`.
struct CUDAAssert {
  volatile int32_t error; // non-zero indicates error, set in device kernel
  uint32_t length; // number of bytes of the assert report in buffer
  uint32_t persistent; // a non-zero value here indicates that a hard device
                       // side assert was triggered
  std::mutex* mutex; // synchronize host side decoding & reset
  alignas(C10_ASSERT_ARG_ALIGN_SIZE) char buffer[C10_ASSERT_BUFFER_SIZE];
};

void checkAssertError(c10::cuda::CUDAAssert* assert_state);

} // namespace cuda
} // namespace c10
