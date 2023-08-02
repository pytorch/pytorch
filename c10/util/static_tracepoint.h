#pragma once

#if defined(__ELF__) && (defined(__x86_64__) || defined(__i386__)) && \
    !CAFFE_DISABLE_SDT

#define CAFFE_HAVE_SDT 1

#include <c10/util/static_tracepoint_elfx86.h>

#define CAFFE_SDT(name, ...) \
  CAFFE_SDT_PROBE_N(         \
      caffe2, name, 0, CAFFE_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
// Use CAFFE_SDT_DEFINE_SEMAPHORE(name) to define the semaphore
// as global variable before using the CAFFE_SDT_WITH_SEMAPHORE macro
#define CAFFE_SDT_WITH_SEMAPHORE(name, ...) \
  CAFFE_SDT_PROBE_N(                        \
      caffe2, name, 1, CAFFE_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
#define CAFFE_SDT_IS_ENABLED(name) (CAFFE_SDT_SEMAPHORE(caffe2, name) > 0)

#else

#define CAFFE_HAVE_SDT 0

#define CAFFE_SDT(name, ...) \
  do {                       \
  } while (0)
#define CAFFE_SDT_WITH_SEMAPHORE(name, ...) \
  do {                                      \
  } while (0)
#define CAFFE_SDT_IS_ENABLED(name) (false)
#define CAFFE_SDT_DEFINE_SEMAPHORE(name)
#define CAFFE_SDT_DECLARE_SEMAPHORE(name)

#endif
