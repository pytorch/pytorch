#pragma once
#include <c10/metal/common.h>

namespace c10 {
namespace metal {
C10_METAL_CONSTEXPR unsigned error_message_count = 30;
struct ErrorMessage {
  char file[128];
  char func[128];
  char message[250];
  unsigned int line;
};

struct ErrorMessages {
#ifdef __METAL__
  ::metal::atomic<unsigned int> count;
#else
  unsigned int count;
#endif
  ErrorMessage msg[error_message_count];
};

#ifdef __METAL__
namespace detail {
static uint strncpy(device char* dst, constant const char* src, unsigned len) {
  uint i = 0;
  while (src[i] != 0 && i < len - 1) {
    dst[i] = src[i];
    i++;
  }
  dst[i] = 0;
  return i;
}

inline uint print_arg(
    device char* ptr,
    unsigned len,
    constant const char* arg) {
  return strncpy(ptr, arg, len);
}

// Returns number length as string in base10
static inline uint base10_length(long num) {
  uint rc = 1;
  if (num < 0) {
    num = -num;
    rc += 1;
  }
  while (num > 9) {
    num /= 10;
    rc++;
  }
  return rc;
}

// Converts signed integer to string
inline uint print_arg(device char* ptr, unsigned len, long arg) {
  const auto arg_len = base10_length(arg);
  if (arg_len >= len)
    return 0;
  if (arg < 0) {
    ptr[0] = '-';
    arg = -arg;
  }
  uint idx = 1;
  do {
    ptr[arg_len - idx] = '0' + (arg % 10);
    arg /= 10;
    idx++;
  } while (arg > 0);
  ptr[arg_len] = 0;
  return arg_len;
}

template <typename T>
inline void print_args(device char* ptr, unsigned len, T arg) {
  print_arg(ptr, len, arg);
}

template <typename T, typename... Args>
inline void print_args(device char* ptr, unsigned len, T arg, Args... args) {
  const auto rc = print_arg(ptr, len, arg);
  print_args(ptr + rc, len - rc, args...);
}

} // namespace detail

template <typename... Args>
static void report_error(
    device ErrorMessages* msgs,
    constant const char* file,
    int line,
    constant const char* func,
    Args... args) {
  const auto idx =
      atomic_fetch_add_explicit(&msgs->count, 1, ::metal::memory_order_relaxed);
  if (idx >= error_message_count) {
    return;
  }
  device auto* msg = &msgs->msg[idx];
  detail::strncpy(msg->file, file, 128);
  detail::strncpy(msg->func, func, 128);
  detail::print_args(msg->message, 250, args...);
  msg->line = line;
}

#define TORCH_REPORT_ERROR(buf, ...) \
  ::c10::metal::report_error(buf, __FILE__, __LINE__, __func__, __VA_ARGS__)
#endif
} // namespace metal
} // namespace c10
