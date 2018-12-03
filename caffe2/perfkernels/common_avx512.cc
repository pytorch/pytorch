// This file is here merely to check that the flags are not mixed up: for
// example, if your compiler did not specify -mavx512f and -mavx512dq,
// you should not provide the CAFFE2_PERF_WITH_AVX512 macro.

#include "caffe2/core/common.h"

#ifdef CAFFE2_PERF_WITH_AVX512
#if !defined(__AVX512F__) || !defined(__AVX512DQ__)
#error( \
    "You found a build system error: CAFFE2_PERF_WITH_AVX512 is defined" \
    "but __AVX512F__ or __AVX512DQ__ is not defined" \
    "(via e.g. -mavx512f and -mavx512dq).");
#endif
#endif // CAFFE2_PERF_WITH_AVX512

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#ifndef CAFFE2_PERF_WITH_AVX512
#error( \
    "You found a build system error: __AVX512F__ and __AVX512DQ__ is defined" \
    "(via e.g. -mavx512f and -mavx512dq) " \
    "but CAFFE2_PERF_WITH_AVX512 is not defined.");
#endif // CAFFE2_PERF_WITH_AVX512
#endif
