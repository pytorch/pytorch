#include "caffe2/core/logging.h"

#ifdef CAFFE2_USE_GOOGLE_GLOG

CAFFE2_DEFINE_int(caffe2_log_level, google::ERROR,
                  "The minimum log level that caffe2 will output.");

namespace caffe2 {
bool InitCaffeLogging(int* argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();
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
}  // namespace caffe2

#endif  // CAFFE2_USE_GOOGLE_GLOG