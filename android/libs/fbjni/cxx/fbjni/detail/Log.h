/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
/** @file ALog.h
 *
 *  Very simple (android only) logging. Define LOG_TAG to enable the macros.
 */

#pragma once

#ifdef __ANDROID__

#include <android/log.h>

namespace facebook {
namespace jni {
namespace log_ {
// the weird name of this namespace is to avoid a conflict with the
// function named log.

inline void loge(const char* tag, const char* msg) noexcept {
  __android_log_write(ANDROID_LOG_ERROR, tag, msg);
}

template<typename... ARGS>
inline void loge(const char* tag, const char* msg, ARGS... args) noexcept {
  __android_log_print(ANDROID_LOG_ERROR, tag, msg, args...);
}

inline void logf(const char* tag, const char* msg) noexcept {
  __android_log_write(ANDROID_LOG_FATAL, tag, msg);
}

template<typename... ARGS>
inline void logf(const char* tag, const char* msg, ARGS... args) noexcept {
  __android_log_print(ANDROID_LOG_FATAL, tag, msg, args...);
}

template<typename... ARGS>
[[noreturn]]
inline void logassert(const char* tag, const char* msg, ARGS... args) noexcept {
  __android_log_assert(0, tag, msg, args...);
}


#ifdef LOG_TAG
# define FBJNI_LOGE(...) ::facebook::jni::log_::loge(LOG_TAG, __VA_ARGS__)
# define FBJNI_LOGF(...) ::facebook::jni::log_::logf(LOG_TAG, __VA_ARGS__)
# define FBJNI_ASSERT(cond) do { if (!(cond)) ::facebook::jni::log_::logassert(LOG_TAG, "%s", #cond); } while(0)
#else
# define FBJNI_LOGE(...) ::facebook::jni::log_::loge("log", __VA_ARGS__)
# define FBJNI_LOGF(...) ::facebook::jni::log_::logf("log", __VA_ARGS__)
# define FBJNI_ASSERT(cond) do { if (!(cond)) ::facebook::jni::log_::logassert("log", "%s", #cond); } while(0)
#endif

}}}

#else
#include <stdlib.h>

# define FBJNI_LOGE(...) ((void)0)
# define FBJNI_LOGF(...) (abort())
# define FBJNI_ASSERT(cond) ((void)0)
#endif
