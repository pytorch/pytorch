#pragma once

// Detect address sanitizer as some stuff doesn't work with it

#undef CAFFE2_ASAN_ENABLED

// for clang
#if defined(__has_feature)
#if ((__has_feature(address_sanitizer)))
#define CAFFE2_ASAN_ENABLED 1
#endif
#endif

// for gcc
#if defined(__SANITIZE_ADDRESS__)
#if __SANITIZE_ADDRESS__
#if !defined(CAFFE2_ASAN_ENABLED)
#define CAFFE2_ASAN_ENABLED 1
#endif
#endif
#endif

#if !defined(CAFFE2_ASAN_ENABLED)
#define CAFFE2_ASAN_ENABLED 0
#endif

// Define sanitization macro
#if !CAFFE2_ASAN_ENABLED
#define CAFFE2_NO_SANITIZE(...)
#else
#define CAFFE2_NO_SANITIZE(...) __attribute__((no_sanitize(__VA_ARGS__)))
#endif
