#include "caffe2/core/init.h"
#include "caffe2/core/operator.h" // for StaticLinkingProtector
#include "caffe2/core/scope_guard.h"

#include <iomanip>
#include <mutex>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(
    caffe2_version,
    false,
    "Print Caffe2 version and build options on startup");

namespace caffe2 {

namespace internal {
// Keep track of stages of initialization to differentiate between
// (a) Re-entrant calls to GlobalInit (e.g. caller registers a Caffe2 init
// function which might in turn indirectly invoke GlobalInit).
// (b) Successive calls to GlobalInit, which are handled as documented in
// init.h.
// Note that this is NOT attempting to address thread-safety, see comments
// in init.h.
enum class State {
  Uninitialized,
  Initializing,
  Initialized,
};

Caffe2InitializeRegistry* Caffe2InitializeRegistry::Registry() {
  static Caffe2InitializeRegistry gRegistry;
  return &gRegistry;
}

State& GlobalInitState() {
  static State state = State::Uninitialized;
  return state;
}
} // namespace internal

bool GlobalInitAlreadyRun() {
  return internal::GlobalInitState() == internal::State::Initialized;
}

bool GlobalInit(int* pargc, char*** pargv) {
  C10_LOG_API_USAGE_ONCE("caffe2.global_init");
  static std::recursive_mutex init_mutex;
  std::lock_guard<std::recursive_mutex> guard(init_mutex);
  internal::State& init_state = internal::GlobalInitState();
  static StaticLinkingProtector g_protector;
  bool success = true;

  // NOTE: if init_state == internal::State::Initializing at this point, do
  // nothing because that indicates a re-entrant call
  if (init_state == internal::State::Initialized) {
    VLOG(1) << "GlobalInit has already been called: re-parsing gflags only.";
    // Reparse command line flags
    success &= c10::ParseCommandLineFlags(pargc, pargv);
    UpdateLoggingLevelsFromFlags();
  } else if (init_state == internal::State::Uninitialized) {
    init_state = internal::State::Initializing;
    auto init_state_guard = MakeGuard([&] {
      // If an exception is thrown, go back to Uninitialized state
      if (init_state == internal::State::Initializing) {
        init_state = internal::State::Uninitialized;
      }
    });

    success &= internal::Caffe2InitializeRegistry::Registry()
                   ->RunRegisteredEarlyInitFunctions(pargc, pargv);
    CAFFE_ENFORCE(
        success, "Failed to run some early init functions for caffe2.");
    success &= c10::ParseCommandLineFlags(pargc, pargv);
    success &= InitCaffeLogging(pargc, *pargv);
    // Print out the current build version. Using cerr as LOG(INFO) might be off
    if (FLAGS_caffe2_version) {
      std::cerr << "Caffe2 build configuration: " << std::endl;
      for (const auto& it : GetBuildOptions()) {
        std::cerr << "  " << std::setw(25) << std::left << it.first << " : "
                  << it.second << std::endl;
      }
    }
    // All other initialization functions.
    success &= internal::Caffe2InitializeRegistry::Registry()
                   ->RunRegisteredInitFunctions(pargc, pargv);

    init_state =
        success ? internal::State::Initialized : internal::State::Uninitialized;
  }
  CAFFE_ENFORCE(success, "Failed to run some init functions for caffe2.");
  // TODO: if we fail GlobalInit(), should we continue?
  return success;
}

bool GlobalInit() {
  // This is a version of the GlobalInit where no argument is passed in.
  // On mobile devices, use this global init, since we cannot pass the
  // command line options to caffe2, no arguments are passed.
  int mobile_argc = 1;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  static char caffe2_name[] = "caffe2";
  char* mobile_name = &caffe2_name[0];
  char** mobile_argv = &mobile_name;
  return ::caffe2::GlobalInit(&mobile_argc, &mobile_argv);
}

bool unsafeRunCaffe2InitFunction(const char* name, int* pargc, char*** pargv) {
  return internal::Caffe2InitializeRegistry::Registry()->RunNamedFunction(
      name, pargc, pargv);
}
}  // namespace caffe2
