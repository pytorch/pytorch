#pragma once

#if defined(__ELF__) && (defined(__x86_64__) || defined(__i386__) || defined(__aarch64__) || \
     defined(__arm__)) && !(defined(TORCH_DISABLE_SDT) && TORCH_DISABLE_SDT)

#define TORCH_HAVE_SDT 1

#include <c10/util/static_tracepoint_elf.h>

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

// Mark variadic macro args as unused from https://stackoverflow.com/a/31470425
#define TORCH_UNUSED0()
#define TORCH_UNUSED1(a) (void)(a)
#define TORCH_UNUSED2(a, b) (void)(a), TORCH_UNUSED1(b)
#define TORCH_UNUSED3(a, b, c) (void)(a), TORCH_UNUSED2(b, c)
#define TORCH_UNUSED4(a, b, c, d) (void)(a), TORCH_UNUSED3(b, c, d)
#define TORCH_UNUSED5(a, b, c, d, e) (void)(a), TORCH_UNUSED4(b, c, d, e)
#define TORCH_UNUSED6(a, b, c, d, e, f) (void)(a), TORCH_UNUSED5(b, c, d, e, f)
#define TORCH_UNUSED7(a, b, c, d, e, f, g) \
  (void)(a), TORCH_UNUSED6(b, c, d, e, f, g)
#define TORCH_UNUSED8(a, b, c, d, e, f, g, h) \
  (void)(a), TORCH_UNUSED7(b, c, d, e, f, g, h)

#define TORCH_VA_NUM_ARGS_IMPL(_0, _1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define TORCH_VA_NUM_ARGS(...) \
  TORCH_VA_NUM_ARGS_IMPL(100, ##__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define TORCH_ALL_UNUSED_IMPL_(nargs) TORCH_UNUSED##nargs
#define TORCH_ALL_UNUSED_IMPL(nargs) TORCH_ALL_UNUSED_IMPL_(nargs)

#if defined(_MSC_VER)
#define TORCH_ALL_UNUSED(...)
#else
#define TORCH_ALL_UNUSED(...) \
  TORCH_ALL_UNUSED_IMPL(TORCH_VA_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)
#endif

#define TORCH_SDT(name, ...) \
  do {                                 \
    TORCH_ALL_UNUSED(__VA_ARGS__);     \
  } while (0)
#define TORCH_SDT_WITH_SEMAPHORE(name, ...) \
  do {                                                \
    TORCH_ALL_UNUSED(__VA_ARGS__);                    \
  } while (0)
#define TORCH_SDT_IS_ENABLED(name) (false)
#define TORCH_SDT_SEMAPHORE(name) \
  folly_sdt_semaphore_##pytorch##_##name
#define TORCH_SDT_DEFINE_SEMAPHORE(name)    \
  extern "C" {                                        \
  unsigned short TORCH_SDT_SEMAPHORE(name); \
  }
#define TORCH_SDT_DECLARE_SEMAPHORE(name) \
  extern "C" unsigned short TORCH_SDT_SEMAPHORE(name)

#endif
