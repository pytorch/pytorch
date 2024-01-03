// This file is here merely to check that the flags are not mixed up: for
// example, if your compiler did not specify -mavx2, you should not provide
// the CAFFE2_PERF_WITH_AVX2 macro.

#include "caffe2/core/common.h"

#ifdef CAFFE2_PERF_WITH_AVX2
#ifndef __AVX2__
#error( \
    "You found a build system error: CAFFE2_PERF_WITH_AVX2 is defined" \
    "but __AVX2__ is not defined (via e.g. -mavx2).");
#endif // __AVX2__
#endif // CAFFE2_PERF_WITH_AVX2

#ifdef __AVX2__
#ifndef CAFFE2_PERF_WITH_AVX2
#error( \
    "You found a build system error: __AVX2__ is defined (via e.g. -mavx2) " \
    "but CAFFE2_PERF_WITH_AVX2 is not defined.");
#endif // CAFFE2_PERF_WITH_AVX2
#endif
