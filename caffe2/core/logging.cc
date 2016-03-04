#ifdef ANDROID
#include <android/log.h>
#endif  // ANDROID

#include "caffe2/core/logging.h"

#ifdef CAFFE2_USE_GOOGLE_GLOG

CAFFE2_DEFINE_int(caffe2_log_level, google::ERROR,
                  "The minimum log level that caffe2 will output.");

namespace caffe2 {
bool InitCaffeLogging(int* argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();
  // TODO(Yangqing): add a compatibility support for caffe2_log_level
  // instead of having to use glog's specifications.
  return true;
}
}  // namespace caffe2

#else  // !CAFFE2_USE_GOOGLE_GLOG

CAFFE2_DEFINE_int(caffe2_log_level, CAFFE_ERROR,
                  "The minimum log level that caffe2 will output.");

namespace caffe2 {
bool InitCaffeLogging(int* argc, char** argv) {
  // When doing InitCaffeLogging, we will assume that caffe's flag paser has
  // already finished.
  if (!CommandLineFlagsHasBeenParsed()) {
    std::cerr << "InitCaffeLogging() has to be called after "
                 "ParseCaffeCommandLineFlags. Modify your program to make sure "
                 "of this." << std::endl;
    return false;
  }
  if (FLAGS_caffe2_log_level > CAFFE_FATAL) {
    std::cerr << "The log level of Caffe2 has to be no larger than CAFFE_FATAL("
              << CAFFE_FATAL << "). Capping it to CAFFE_FATAL." << std::endl;
    FLAGS_caffe2_log_level = CAFFE_FATAL;
  }
  return true;
}

MessageLogger::MessageLogger(const char *file, int line, int severity)
  : severity_(severity) {
  if (severity_ < FLAGS_caffe2_log_level) {
    // Nothing needs to be logged.
    return;
  }
#ifdef ANDROID
  tag_ = "native";
#else  // !ANDROID
  tag_ = "";
#endif  // ANDROID
  // Pre-pend the stream with the file and line number.
  StripBasename(std::string(file), &filename_only_);
  /*
  time_t rawtime;
  struct tm * timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  std::chrono::nanoseconds ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch());
  */
  stream_ << "["
          << CAFFE_SEVERITY_PREFIX[std::min(4, CAFFE_FATAL - severity_)]
          //<< (timeinfo->tm_mon + 1) * 100 + timeinfo->tm_mday
          //<< std::setfill('0')
          //<< " " << std::setw(2) << timeinfo->tm_hour
          //<< ":" << std::setw(2) << timeinfo->tm_min
          //<< ":" << std::setw(2) << timeinfo->tm_sec
          //<< "." << std::setw(9) << ns.count() % 1000000000
          << " " << filename_only_ << ":" << line << "] ";
}

// Output the contents of the stream to the proper channel on destruction.
MessageLogger::~MessageLogger() {
  if (severity_ < FLAGS_caffe2_log_level) {
    // Nothing needs to be logged.
    return;
  }
  stream_ << "\n";
#ifdef ANDROID
  static const int android_log_levels[] = {
      ANDROID_LOG_FATAL,    // CAFFE_LOG_FATAL
      ANDROID_LOG_ERROR,    // CAFFE_LOG_ERROR
      ANDROID_LOG_WARN,     // CAFFE_LOG_WARNING
      ANDROID_LOG_INFO,     // CAFFE_LOG_INFO
      ANDROID_LOG_DEBUG,    // CAFFE_VLOG(1)
      ANDROID_LOG_VERBOSE,  // CAFFE_VLOG(2) .. CAFFE_VLOG(N)
  };
  int android_level_index = CAFFE_FATAL - std::min(CAFFE_FATAL, severity_);
  int level = android_log_levels[std::min(android_level_index, 5)];
  // Output the log string the Android log at the appropriate level.
  __android_log_print(level, tag_, stream_.str().c_str());
  // Indicate termination if needed.
  if (severity_ == CAFFE_FATAL) {
    __android_log_print(ANDROID_LOG_FATAL, tag_, "terminating.\n");
  }
#else  // !ANDROID
  if (severity_ >= FLAGS_caffe2_log_level) {
    // If not building on Android, log all output to std::cerr.
    std::cerr << stream_.str();
  }
#endif  // ANDROID
  if (severity_ == CAFFE_FATAL) {
    DealWithFatal();
  }
}

void MessageLogger::StripBasename(
    const std::string &full_path, std::string *filename) {
  const char kSeparator = '/';
  size_t pos = full_path.rfind(kSeparator);
  if (pos != std::string::npos) {
    *filename = full_path.substr(pos + 1, std::string::npos);
  } else {
    *filename = full_path;
  }
}

}  // namespace caffe2

#endif  // !CAFFE2_USE_GOOGLE_GLOG
