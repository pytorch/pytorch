#pragma once

#if defined(__ELF__) && (defined(__x86_64__) || defined(__i386__))
#include <caffe2/core/static_tracepoint_elfx86.h>

#define CAFFE_SDT(name, ...)                                         \
  CAFFE_SDT_PROBE_N(                                                 \
    caffe2, name, CAFFE_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
#else
#define CAFFE_SDT(name, ...) do {} while(0)
#endif
