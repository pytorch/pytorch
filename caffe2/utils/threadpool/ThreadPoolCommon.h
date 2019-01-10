#ifndef CAFFE2_UTILS_THREADPOOL_COMMON_H_
#define CAFFE2_UTILS_THREADPOOL_COMMON_H_

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// caffe2 depends upon NNPACK, which depends upon this threadpool, so
// unfortunately we can't reference core/common.h here

// This is copied from core/common.h's definition of C10_MOBILE
// Define enabled when building for iOS or Android devices
#if defined(__ANDROID__)
#define C10_ANDROID 1
#elif (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define C10_IOS 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define C10_IOS 1
#else
#endif // ANDROID / IOS / MACOS

#endif  // CAFFE2_UTILS_THREADPOOL_COMMON_H_
