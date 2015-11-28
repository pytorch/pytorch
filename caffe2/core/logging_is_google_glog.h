#ifndef CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_
#define CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_

// Using google glog.
#include <glog/logging.h>

#ifdef CAFFE2_THROW_ON_FATAL
static_assert(false, "If you use CAFFE2_USE_GOOGLE_GLOG, "
                     "you should not specify CAFFE2_THROW_ON_FATAL.");
#endif  // CAFFE2_THROW_ON_FATAL

static_assert(CAFFE2_LOG_THRESHOLD <= google::FATAL,
              "CAFFE2_LOG_THRESHOLD should at most be FATAL.");

// Re-route everything to glog's corresponding macros.
#define LOG_ENABLE_COMPILE_TIME_THRESHOLD(n) \
  if (google::n >= CAFFE2_LOG_THRESHOLD) LOG(n)
#define CAFFE_VLOG_IS_ON(n) ((-n) >= CAFFE2_LOG_THRESHOLD)
#define CAFFE_LOG_INFO LOG_ENABLE_COMPILE_TIME_THRESHOLD(INFO)
#define CAFFE_LOG_WARNING LOG_ENABLE_COMPILE_TIME_THRESHOLD(WARNING)
#define CAFFE_LOG_ERROR LOG_ENABLE_COMPILE_TIME_THRESHOLD(ERROR)
#define CAFFE_LOG_FATAL LOG(FATAL)
#define CAFFE_VLOG(n) if (CAFFE_VLOG_IS_ON(n)) VLOG(n)

#define CAFFE_CHECK(...) CHECK(__VA_ARGS__)
#define CAFFE_DCHECK(...) DCHECK(__VA_ARGS__)
#define CAFFE_CHECK_EQ(...) CHECK_EQ(__VA_ARGS__)
#define CAFFE_CHECK_NE(...) CHECK_NE(__VA_ARGS__)
#define CAFFE_CHECK_LE(...) CHECK_LE(__VA_ARGS__)
#define CAFFE_CHECK_LT(...) CHECK_LT(__VA_ARGS__)
#define CAFFE_CHECK_GE(...) CHECK_GE(__VA_ARGS__)
#define CAFFE_CHECK_GT(...) CHECK_GT(__VA_ARGS__)
#define CAFFE_DCHECK_EQ(...) DCHECK_EQ(__VA_ARGS__)
#define CAFFE_DCHECK_NE(...) DCHECK_NE(__VA_ARGS__)
#define CAFFE_DCHECK_LE(...) DCHECK_LE(__VA_ARGS__)
#define CAFFE_DCHECK_LT(...) DCHECK_LT(__VA_ARGS__)
#define CAFFE_DCHECK_GE(...) DCHECK_GE(__VA_ARGS__)
#define CAFFE_DCHECK_GT(...) DCHECK_GT(__VA_ARGS__)
#define CAFFE_CHECK_NOTNULL(...) CHECK_NOTNULL(__VA_ARGS__)

#endif  // CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_
