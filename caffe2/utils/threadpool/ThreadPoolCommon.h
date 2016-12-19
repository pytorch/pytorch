#ifndef CAFFE2_UTILS_THREADPOOL_COMMON_H_
#define CAFFE2_UTILS_THREADPOOL_COMMON_H_

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// caffe2 depends upon NNPACK, which depends upon this threadpool, so
// unfortunately we can't reference core/common.h here

// This is copied from core/common.h's definition of CAFFE2_MOBILE
// Define enabled when building for iOS or Android devices
#if !defined(CAFFE2_THREADPOOL_MOBILE)
#if defined(__ANDROID__)
#define CAFFE2_ANDROID 1
#define CAFFE2_THREADPOOL_MOBILE 1
#elif (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define CAFFE2_IOS 1
#define CAFFE2_THREADPOOL_MOBILE 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define CAFFE2_IOS 1
#define CAFFE2_THREADPOOL_MOBILE 1
#else
#define CAFFE2_THREADPOOL_MOBILE 0
#endif // ANDROID / IOS / MACOS
#endif // CAFFE2_THREADPOOL_MOBILE

#endif  // CAFFE2_UTILS_THREADPOOL_COMMON_H_
