#pragma once

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#if !defined(__MAC_14_0) && \
    (!defined(MAC_OS_X_VERSION_14_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_14_0))

// define BFloat16 enums for MacOS13
#define MPSDataTypeBFloat16 (MPSDataTypeAlternateEncodingBit | MPSDataTypeFloat16)
#endif
