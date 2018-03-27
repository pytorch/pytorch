#include "caffe2/core/init.h"
#include "caffe2/core/operator.h" // for StaticLinkingProtector

#include <iomanip>

CAFFE2_DEFINE_bool(caffe2_version, false,
                   "Print Caffe2 version and build options on startup");

namespace caffe2 {

namespace internal {
Caffe2InitializeRegistry* Caffe2InitializeRegistry::Registry() {
  static Caffe2InitializeRegistry gRegistry;
  return &gRegistry;
}
}

bool GlobalInit(int* pargc, char*** pargv) {
  static bool global_init_was_already_run = false;
  static StaticLinkingProtector g_protector;
  if (global_init_was_already_run) {
    VLOG(1) << "GlobalInit has already been called: did you double-call?";
    return true;
  }
  global_init_was_already_run = true;
  bool success = true;
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredEarlyInitFunctions(pargc, pargv);
  CAFFE_ENFORCE(success,
                "Failed to run some early init functions for caffe2.");
  success &= ParseCaffeCommandLineFlags(pargc, pargv);
  success &= InitCaffeLogging(pargc, *pargv);
  // Print out the current build version. Using cerr as LOG(INFO) might be off
  if (VLOG_IS_ON(1) || FLAGS_caffe2_version) {
    std::cerr << "Caffe2 build configuration: " << std::endl;
    for (const auto& it : GetBuildOptions()) {
      std::cerr << "  " << std::setw(25) << std::left << it.first << " : "
                << it.second << std::endl;
    }
  }
  // All other initialization functions.
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredInitFunctions(pargc, pargv);
  if (!success) {
    global_init_was_already_run = false;
  }
  CAFFE_ENFORCE(success,
                "Failed to run some init functions for caffe2.");
  // TODO: if we fail GlobalInit(), should we continue?
  return success;
}
}  // namespace caffe2
