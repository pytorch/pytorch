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
static void strncpy(device char* dst, constant const char* src, unsigned len) {
  uint i = 0;
  while (src[i] != 0 && i < len - 1) {
    dst[i] = src[i];
    i++;
  }
  dst[i] = 0;
}

void report_error(
    device ErrorMessages* msgs,
    constant const char* file,
    int line,
    constant const char* func,
    constant const char* message) {
  const auto idx =
      atomic_fetch_add_explicit(&msgs->count, 1, ::metal::memory_order_relaxed);
  if (idx >= error_message_count) {
    return;
  }
  device auto* msg = &msgs->msg[idx];
  strncpy(msg->file, file, 128);
  strncpy(msg->func, func, 128);
  strncpy(msg->message, message, 250);
  msg->line = line;
}
#endif
} // namespace metal
} // namespace c10
