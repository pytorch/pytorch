// This file is here merely to check that the flags are not mixed up: for
// example, if your compiler did not specify -march=armv8-a+sve, you should not
// provide the CAFFE2_PERF_WITH_SVE macro.

#include "caffe2/core/common.h"

#ifdef CAFFE2_PERF_WITH_SVE
#ifndef __ARM_FEATURE_SVE
#error( \
    "You found a build system error: CAFFE2_PERF_WITH_SVE is defined" \
    "but __ARM_FEATURE_SVE is not defined (via e.g. -march=armv8-a+sve).");
#endif // __ARM_FEATURE_SVE
#endif // CAFFE2_PERF_WITH_SVE

#ifdef __ARM_FEATURE_SVE
#ifndef CAFFE2_PERF_WITH_SVE
#error( \
    "You found a build system error: __SVE__ is defined \
    (via e.g. -march=armv8-a+sve) " \
    "but CAFFE2_PERF_WITH_SVE is not defined.");
#endif // CAFFE2_PERF_WITH_SVE
#endif
