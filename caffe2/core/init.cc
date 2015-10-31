#include "caffe2/core/init.h"

namespace caffe2 {
std::stringstream& GlobalInitStream() {
  static std::stringstream ss;
  return ss;
}

bool GlobalInit(int* pargc, char** argv) {
  static bool global_init_was_already_run = false;
  bool success = true;
  if (global_init_was_already_run) {
    std::cerr << "GlobalInit has already been called: did you double-call?";
    return false;
  }
  success &= ParseCaffeCommandLineFlags(pargc, argv);
  success &= InitCaffeLogging(pargc, argv);
  // All other initialization functions.
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredInitFunctions();
  global_init_was_already_run = true;
  if (success) {
    CAFFE_LOG_INFO << GlobalInitStream().str();
  } else {
    CAFFE_LOG_ERROR << GlobalInitStream().str();
    // TODO: if we fail GlobalInit(), should we continue?
    abort();
  }
  // Clear the global init stream.
  GlobalInitStream().str(std::string());
  return success;
}
}  // namespace caffe2