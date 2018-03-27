#ifndef CAFFE2_UTILS_THREADPOOL_COMMON_H_
#define CAFFE2_UTILS_THREADPOOL_COMMON_H_

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// caffe2 depends upon NNPACK, which depends upon this threadpool, so
// unfortunately we can't reference core/common.h here

// This is copied from core/common.h's definition of CAFFE2_MOBILE
// Define enabled when building for iOS or Android devices
#if defined(__ANDROID__)
#define CAFFE2_ANDROID 1
#elif (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define CAFFE2_IOS 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define CAFFE2_IOS 1
#else
#endif // ANDROID / IOS / MACOS

#endif  // CAFFE2_UTILS_THREADPOOL_COMMON_H_
