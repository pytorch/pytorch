#pragma once

#include <sstream>
#include <string>

#include "c10/macros/Macros.h"

#ifdef __CUDACC__
// Designates functions callable from the host (CPU) and the device (GPU)
#define AT_HOST_DEVICE __host__ __device__
#define AT_DEVICE __device__
#define AT_HOST __host__
#else
#define AT_HOST_DEVICE
#define AT_HOST
#define AT_DEVICE
#endif

#ifdef __HIP_PLATFORM_HCC__
#define HIP_HOST_DEVICE __host__ __device__
#else
#define HIP_HOST_DEVICE
#endif

#if defined(__ANDROID__)
#define AT_ANDROID 1
#define AT_MOBILE 1
#elif (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define AT_IOS 1
#define AT_MOBILE 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define AT_IOS 1
#define AT_MOBILE 0
#else
#define AT_MOBILE 0
#endif // ANDROID / IOS / MACOS
