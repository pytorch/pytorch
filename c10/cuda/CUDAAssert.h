#pragma once

#include <cstdint>

namespace c10 {
namespace cuda {

constexpr size_t C10_ASSERT_BUFFER_SIZE = 2048;
constexpr size_t C10_ASSERT_ARG_ALIGN_SIZE = sizeof(double);

// This class holds the assert state associated with a single CUDA stream.
// The `error` field is set to a non-zero value with a CAS operation in the
// kernel when a C10_KERNEL_ASSERT() fails. A device sync should be executed
// prior to accessing the assert report in `buffer`.
struct CUDAAssert {
  volatile int32_t error; // error signal, non-zero when assert failed
  uint32_t length; // number of bytes of the assert report in buffer
  alignas(C10_ASSERT_ARG_ALIGN_SIZE) char buffer[C10_ASSERT_BUFFER_SIZE];
};

void check_assert_error(c10::cuda::CUDAAssert* assert_state);

} // namespace cuda
} // namespace c10
