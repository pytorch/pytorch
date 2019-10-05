#pragma once

#include <c10/cuda/CUDAAssert.h>
#include <c10/macros/Macros.h>
#include <cstddef>

namespace at {
namespace native {
namespace assert {

using namespace c10::cuda;

#ifdef ENABLE_C10_KERNEL_ASSERT_STRING_SUPPORT

static inline C10_HOST_DEVICE char* write_string(
    char* dst,
    const char* src,
    size_t n) {
  size_t i = 0;
  if (src && n > 0) {
    while (i < n - 1 && src[i]) {
      dst[i] = src[i];
      ++i;
    }
    dst[i] = '\0'; // ensure strings are null-terminated
    ++i;
  }
  return dst + i;
}

inline C10_HOST_DEVICE char* copy_string(
    char* dst,
    char* buffer_end,
    const char* src) {
  if (dst) {
    auto size_ptr =
        reinterpret_cast<size_t*>(dst); // store pointer of string size field
    dst += C10_ASSERT_ARG_ALIGN_SIZE;
    if (dst < buffer_end) {
      char* start = dst;

      // write string into buffer and store size
      dst = write_string(dst, src, buffer_end - dst);
      *size_ptr = (size_t)(dst - start);

      // fix alignment
      ptrdiff_t offset = dst - start;
      ptrdiff_t misalign = offset % C10_ASSERT_ARG_ALIGN_SIZE;
      if (misalign > 0) {
        dst += C10_ASSERT_ARG_ALIGN_SIZE - misalign;
      }

      // again check position after alignment correction
      if (dst <= buffer_end) {
        return dst;
      }
    }
  }

  return nullptr; // insufficient buffer
}


inline C10_HOST_DEVICE char* copy_args(
    char* dst,
    char* const end,
    const char* arg) {
  return copy_string(dst, end, arg);
}

#endif

static inline C10_HOST_DEVICE char* copy_args(char* dst, char* const end) {
  return dst;
}

template <typename T>
C10_HOST_DEVICE char* copy_args(char* dst, char* const end, T arg) {
  if (!dst || dst + C10_ASSERT_ARG_ALIGN_SIZE >= end) {
    return nullptr;
  }

  // write length of arg
  *reinterpret_cast<size_t*>(dst) = sizeof(T);
  dst += C10_ASSERT_ARG_ALIGN_SIZE;

  // write argument value
  *reinterpret_cast<T*>(dst) = arg;
  dst += C10_ASSERT_ARG_ALIGN_SIZE;

  return dst;
}

template <typename T, typename... Args>
C10_HOST_DEVICE char* copy_args(
    char* dst,
    char* const end,
    T first,
    Args... args) {
  dst = copy_args(dst, end, first);
  return copy_args(dst, end, args...);
}

template <typename... Args>
C10_HOST_DEVICE __noinline__ void assert_fail(
    CUDAAssert* assert_state,
    AssertError error_code,
    bool persistent,
    AssertSource file,
    uint32_t line,
    AssertFormat format,
    Args... args) {
  assert(error_code != AssertError::OK);
#ifdef __CUDA_ARCH__
  if (atomicCAS(
          const_cast<int32_t*>(&assert_state->error),
          0,
          static_cast<int32_t>(error_code)) != 0) {
    return;
  }
#else
  assert(false); // should never be called
#endif

  char* buffer = assert_state->buffer;
  char* dst = buffer;
  char* const end = dst + sizeof(assert_state->buffer);
  dst = copy_args(dst, end, line, args...);

  assert_state->persistent = persistent;
  assert_state->file = file;
  assert_state->line = line;
  assert_state->format = format;
  assert_state->length = dst ? dst - buffer : 0;
}

// handle case without format string, e.g. C10_KERNEL_ASSERT(false)
static inline C10_HOST_DEVICE void assert_fail(
    CUDAAssert* assert_state,
    AssertError error_code,
    bool persistent,
    AssertSource file,
    uint32_t line,
    AssertFormat format) {
  assert_fail(assert_state, error_code, persistent, file, line, format, 0);
}

static inline CUDAAssert* prepare_kernel_assert() {
  auto current_stream = getCurrentCUDAStream();
  return current_stream.assert_state();
}

} // namespace assert
} // namespace native
} // namespace at

#define C10_PREPARE_KERNEL_ASSERT \
  auto __c10_assert_state = at::native::assert::prepare_kernel_assert();

#define C10_KERNEL_ASSERT(exp, format, ...) \
  do {                                      \
    if (!(exp)) {                           \
      at::native::assert::assert_fail(      \
          __c10_assert_state,               \
          c10::cuda::AssertError::Error,    \
          true, /*persistent*/              \
          __c10_assert_source,              \
          static_cast<uint32_t>(__LINE__),  \
          format,                           \
          ##__VA_ARGS__);                   \
      assert(exp);                          \
      assert(false);                        \
    }                                       \
  } while (false)

#define C10_KERNEL_ASSERT_RETURN_TYPE_(rval, exp, error, format, ...) \
  if (!(exp)) {                                                       \
    at::native::assert::assert_fail(                                  \
        __c10_assert_state,                                           \
        error,                                                        \
        false, /*persistent*/                                         \
        __c10_assert_source,                                          \
        static_cast<uint32_t>(__LINE__),                              \
        format,                                                       \
        ##__VA_ARGS__);                                               \
    return rval;                                                      \
  }

#define C10_KERNEL_ASSERT_RETURN_FORMAT_(rval, exp, format, ...) \
  C10_KERNEL_ASSERT_RETURN_TYPE_(                                \
      rval, exp, c10::cuda::AssertError::Error, format, ##__VA_ARGS__)

#define C10_KERNEL_ASSERT_RETURN_(rval, exp, ...) \
  C10_KERNEL_ASSERT_RETURN_FORMAT_(               \
      rval, exp, c10::cuda::AssertFormat::Default, ##__VA_ARGS__)

#define C10_KERNEL_ASSERT_RETURN(exp, ...) \
  C10_KERNEL_ASSERT_RETURN_(, exp, ##__VA_ARGS__)


#define C10_KERNEL_ASSERT_SOFT_TYPE(error, exp, format, ...) \
  (exp) ? ((void)0)                                          \
        : at::native::assert::assert_fail(                   \
              __c10_assert_state,                            \
              error,                                         \
              false, /*persistent*/                          \
              __c10_assert_source,                           \
              static_cast<uint32_t>(__LINE__),               \
              format,                                        \
              ##__VA_ARGS__)

#define C10_KERNEL_ASSERT_SOFT_FORMAT(exp, format, ...) \
  C10_KERNEL_ASSERT_SOFT_TYPE(                          \
      c10::cuda::AssertError::Error, exp, format, ##__VA_ARGS__)

#define C10_KERNEL_ASSERT_SOFT(exp, ...) \
  C10_KERNEL_ASSERT_SOFT_TYPE(           \
      c10::cuda::AssertError::Error,     \
      exp,                               \
      c10::cuda::AssertFormat::Default,  \
      ##__VA_ARGS__)

#define C10_KERNEL_INDEX_ERROR_SOFT(index, axis, size) \
  C10_KERNEL_ASSERT_SOFT_TYPE(                         \
      c10::cuda::AssertError::IndexError,              \
      false,                                           \
      c10::cuda::AssertFormat::IndexOutOfBounds,       \
      index,                                           \
      axis,                                            \
      size)
