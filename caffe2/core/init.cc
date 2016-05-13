#include "caffe2/core/init.h"

#ifndef CAFFE2_BUILD_STRING
#define CAFFE2_BUILD_STRING "build_version_not_set"
#endif

namespace caffe2 {

bool GlobalInit(int* pargc, char*** pargv) {
  static bool global_init_was_already_run = false;
  if (global_init_was_already_run) {
    CAFFE_VLOG(1) << "GlobalInit has already been called: did you double-call?";
    return true;
  }
  bool success = true;
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredEarlyInitFunctions(pargc, pargv);
  CHECK(success) << "Failed to run some early init functions for caffe.";
  success &= ParseCaffeCommandLineFlags(pargc, pargv);
  success &= InitCaffeLogging(pargc, *pargv);
  // Print out the current build version.
  CAFFE_VLOG(1) << "Caffe2 build version: " << CAFFE2_BUILD_STRING;
  // All other initialization functions.
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredInitFunctions(pargc, pargv);
  CHECK(success) << "Failed to run some early init functions for caffe.";
  global_init_was_already_run = true;
  // TODO: if we fail GlobalInit(), should we continue?

  return success;
}
}  // namespace caffe2
