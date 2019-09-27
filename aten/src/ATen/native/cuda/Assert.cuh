#pragma once

#include <c10/cuda/CUDAAssert.h>

namespace at {
namespace native {
namespace assert {

using namespace c10::cuda;

__device__ CUDAAssert* global_assert_state = nullptr;

static inline C10_HOST_DEVICE char* device_strcpy(
    char* dst,
    const char* src,
    int n) {
  int i = 0;
  while (i + 1 < n && src[i]) {
    dst[i] = src[i];
    ++i;
  }
  if (i < n) {
    dst[i] = '\0';
  }
  return dst;
}

// If the assert error state was not set before this function returns true,
// otherwise false.
static inline C10_HOST_DEVICE bool assert_failed(
    CUDAAssert* assert_state,
    CUDAAssertKind kind,
    const char* message,
    uint32_t line,
    const char* file) {
#ifdef __CUDA_ARCH__
  if (atomicCAS(const_cast<int32_t*>(&assert_state->error), 0, 1) != 0) {
    return false;
  }
#else
  assert(false); // should never be called
#endif

  // error message and kind
  device_strcpy(assert_state->message, message, MAX_ASSERT_MESSAGE_LENGTH);
  assert_state->kind = kind;

  // call site information
  device_strcpy(assert_state->file, file, MAX_ASSERT_FILE_LENGTH);
  assert_state->line = line;

  return true;
}

static inline C10_DEVICE bool graceful_assert_(
    bool condition,
    const char* message,
    uint32_t line,
    const char* file) {
  if (!condition) {
    assert_failed(
        global_assert_state, CUDAAssertKind::ASSERTION_FAILED, message, line, file);
  }

  return !global_assert_state->error; // return false if we are in error state, signals
                               // kernel to quit
}

static inline C10_DEVICE void graceful_index_error_(
    int64_t index,
    int64_t axis,
    int64_t size,
    const char* message,
    uint32_t line,
    const char* file) {
  if (assert_failed(
          global_assert_state,
          CUDAAssertKind::INDEX_OUT_OF_BOUNDS,
          message,
          line,
          file)) {
    // capture index details
    CUDAAssertDetailIndexError& details =
        global_assert_state->details.index_error;
    details.index = index;
    details.axis = axis;
    details.size = size;
  }
}

static inline C10_HOST_DEVICE void graceful_division_by_zero_(
    const char* message,
    uint32_t line,
    const char* file) {
#ifdef __CUDA_ARCH__
  assert_failed(
      global_assert_state, CUDAAssertKind::ZERO_DIVISION, message, line, file);
#endif
}

} // namespace assert
} // namespace native
} // namespace at

inline C10_HOST void enableCudaAssert(c10::cuda::CUDAAssert* state) {
  // only set pointer value
  cudaMemcpyToSymbol(
      at::native::assert::global_assert_state, &state, sizeof(state));
}

#define C10_KERNEL_ASSERT_ENABLE                    \
  do {                                              \
    auto stream = at::cuda::getCurrentCUDAStream(); \
    auto assert_state = stream.assert_state();      \
    enableCudaAssert(assert_state);                 \
  } while (0);

#define C10_KERNEL_ASSERT(exp, msg) \
  at::native::assert::graceful_assert_(         \
      exp, msg, static_cast<uint32_t>(__LINE__), __FILE__)

#define C10_KERNEL_ERROR_INDEX_ERROR(index, axis, size, msg) \
  at::native::assert::graceful_index_error_(         \
      index, axis, size, msg, static_cast<uint32_t>(__LINE__), __FILE__)

#define C10_KERNEL_ERROR_DIVISION_BY_ZERO(msg) \
  at::native::assert::graceful_division_by_zero_(    \
      msg, static_cast<uint32_t>(__LINE__), __FILE__)
