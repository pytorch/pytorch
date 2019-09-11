#include <torch/script.h>

#if TARGET_OS_IPHONE
    #define AT_NNPACK_ENABLED() 1
    #define USE_NNPACK ON
    #undef CAFFE2_PERF_WITH_AVX512
#endif
