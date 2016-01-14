#include "caffe2/core/init.h"

#ifndef CAFFE2_BUILD_STRING
#define CAFFE2_BUILD_STRING "build_version_not_set"
#endif

namespace caffe2 {

bool GlobalInit(int* pargc, char** argv) {
  static bool global_init_was_already_run = false;
  bool success = true;
  if (global_init_was_already_run) {
    std::cerr << "GlobalInit has already been called: did you double-call?";
    return false;
  }
  success &= ParseCaffeCommandLineFlags(pargc, argv);
  success &= InitCaffeLogging(pargc, argv);
  // Print out the current build version.
  CAFFE_LOG_INFO << "Caffe2 build version: " << CAFFE2_BUILD_STRING;
  // All other initialization functions.
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredInitFunctions();
  global_init_was_already_run = true;
  if (!success) {
    // TODO: if we fail GlobalInit(), should we continue?
    abort();
  }
  return success;
}
}  // namespace caffe2
