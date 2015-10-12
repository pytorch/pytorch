#ifndef CAFFE2_CORE_LOGGING_H_
#define CAFFE2_CORE_LOGGING_H_

#include "caffe2/core/flags.h"

CAFFE2_DECLARE_int(caffe2_log_level);

// CAFFE2_LOG_THRESHOLD is a compile time flag that would allow us to turn off
// logging at compile time so no logging message below that level is produced
// at all. The value should be between INT_MIN and CAFFE_FATAL.
#ifndef CAFFE2_LOG_THRESHOLD
// If we have not defined the compile time log threshold, we keep all the
// log cases.
#define CAFFE2_LOG_THRESHOLD INT_MIN
#endif  // CAFFE2_LOG_THRESHOLD

namespace caffe2 {
// Functions that we use for initialization.
bool InitCaffeLogging(int* argc, char** argv);

constexpr bool IsUsingGoogleLogging() {
#ifdef CAFFE2_USE_GOOGLE_GLOG
  return true;
#else
  return false;
#endif
}

constexpr bool IsThrowingExceptionAtFatal() {
#ifdef CAFFE2_THROW_ON_FATAL
  return true;
#else
  return false;
#endif
}

}  // namespace caffe2

// Below are different implementations for glog and non-glog cases.

#ifdef CAFFE2_USE_GOOGLE_GLOG
#include "caffe2/core/logging_is_google_glog.h"
#else   // !CAFFE2_USE_GOOGLE_GLOG
#include "caffe2/core/logging_is_not_google_glog.h"
#endif  // CAFFE2_USE_GOOGLE_GLOG

#endif  // CAFFE2_CORE_LOGGING_H_
