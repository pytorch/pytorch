#pragma once

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#if !defined(__MAC_26_0) && (!defined(MAC_OS_X_VERSION_26_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_26_0))
constexpr NSInteger MTLGPUFamilyApple10 = 1010;
#endif
