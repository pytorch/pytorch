#pragma once

#include <c10/cuda/CUDAAssert.h>

namespace at {
namespace native {
namespace assert {

using namespace c10::cuda;

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
    CUDAAssert* assert_state,
    bool condition,
    const char* message,
    uint32_t line,
    const char* file) {
  if (!condition) {
    assert_failed(
        assert_state, CUDAAssertKind::ASSERTION_FAILED, message, line, file);
  }

  return !assert_state->error; // return false if we are in error state, signals
                               // kernel to quit
}

static inline C10_DEVICE void graceful_index_error_(
    CUDAAssert* assert_state,
    int64_t index,
    int64_t axis,
    int64_t size,
    const char* message,
    uint32_t line,
    const char* file) {
  if (assert_failed(
          assert_state,
          CUDAAssertKind::INDEX_OUT_OF_BOUNDS,
          message,
          line,
          file)) {
    // capture index details
    CUDAAssertDetailIndexError& details = assert_state->details.index_error;
    details.index = index;
    details.axis = axis;
    details.size = size;
  }
}

static inline C10_HOST_DEVICE void graceful_devision_by_zero_(
    CUDAAssert* assert_state,
    const char* message,
    uint32_t line,
    const char* file) {
  assert_failed(
      assert_state, CUDAAssertKind::ZERO_DIVISION, message, line, file);
}

} // namespace assert
} // namespace native
} // namespace at

#define graceful_assert(assert_state, exp, msg) \
  at::native::assert::graceful_assert_(         \
      assert_state, exp, msg, static_cast<uint32_t>(__LINE__), __FILE__)

#define graceful_index_error(assert_state, index, axis, size, msg) \
  at::native::assert::graceful_index_error_(                       \
      assert_state,                                                \
      index,                                                       \
      axis,                                                        \
      size,                                                        \
      msg,                                                         \
      static_cast<uint32_t>(__LINE__),                             \
      __FILE__)

#define graceful_devision_by_zero(assert_state, msg) \
  at::native::assert::graceful_devision_by_zero_(    \
      assert_state, msg, static_cast<uint32_t>(__LINE__), __FILE__)
