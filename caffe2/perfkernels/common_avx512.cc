// This file is here merely to check that the flags are not mixed up: for
// example, if your compiler did not specify -mavx512f, -mavx512dq, and
// -mavx512vl you should not provide the CAFFE2_PERF_WITH_AVX512 macro.

#include "caffe2/core/common.h"

#ifdef CAFFE2_PERF_WITH_AVX512
#if !defined(__AVX512F__) || !defined(__AVX512DQ__) || !defined(__AVX512VL__)
#error( \
    "You found a build system error: CAFFE2_PERF_WITH_AVX512 is defined" \
    "but __AVX512F__, __AVX512DQ__, or __AVX512VL is not defined" \
    "(via e.g. -mavx512f, -mavx512dq, and -mavx512vl).");
#endif
#endif // CAFFE2_PERF_WITH_AVX512

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
#ifndef CAFFE2_PERF_WITH_AVX512
#error( \
    "You found a build system error: __AVX512F__, __AVX512DQ__, __AVX512VL__ " \
    "is defined (via e.g. -mavx512f, -mavx512dq, and -mavx512vl) " \
    "but CAFFE2_PERF_WITH_AVX512 is not defined.");
#endif // CAFFE2_PERF_WITH_AVX512
#endif
