#pragma once

#if defined(__ELF__) && (defined(__x86_64__) || defined(__i386__)) && \
    !(defined(TORCH_DISABLE_SDT) && TORCH_DISABLE_SDT)

#define TORCH_HAVE_SDT 1

#include <c10/util/static_tracepoint_elfx86.h>

#define TORCH_SDT(name, ...) \
  TORCH_SDT_PROBE_N(         \
      pytorch, name, 0, TORCH_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
// Use TORCH_SDT_DEFINE_SEMAPHORE(name) to define the semaphore
// as global variable before using the TORCH_SDT_WITH_SEMAPHORE macro
#define TORCH_SDT_WITH_SEMAPHORE(name, ...) \
  TORCH_SDT_PROBE_N(                        \
      pytorch, name, 1, TORCH_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
#define TORCH_SDT_IS_ENABLED(name) (TORCH_SDT_SEMAPHORE(pytorch, name) > 0)

#else

#define TORCH_HAVE_SDT 0

#define TORCH_SDT(name, ...) \
  do {                       \
  } while (0)
#define TORCH_SDT_WITH_SEMAPHORE(name, ...) \
  do {                                      \
  } while (0)
#define TORCH_SDT_IS_ENABLED(name) (false)
#define TORCH_SDT_DEFINE_SEMAPHORE(name)
#define TORCH_SDT_DECLARE_SEMAPHORE(name)

#endif
