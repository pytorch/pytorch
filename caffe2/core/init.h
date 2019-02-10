#ifndef CAFFE2_CORE_INIT_H_
#define CAFFE2_CORE_INIT_H_

#include "caffe2/core/common.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

namespace internal {
class CAFFE2_API Caffe2InitializeRegistry {
 public:
  typedef bool (*InitFunction)(int*, char***);
  // Registry() is defined in .cpp file to make registration work across
  // multiple shared libraries loaded with RTLD_LOCAL
  static Caffe2InitializeRegistry* Registry();

  void
  Register(InitFunction function, bool run_early, const char* description) {
    if (run_early) {
      // Disallow registration after GlobalInit of early init functions
      CAFFE_ENFORCE(!early_init_functions_run_yet_);
      early_init_functions_.emplace_back(function, description);
    } else {
      if (init_functions_run_yet_) {
        // Run immediately, since GlobalInit already ran. This should be
        // rare but we want to allow it in some cases.
        LOG(WARNING) << "Running init function after GlobalInit: "
                     << description;
        // TODO(orionr): Consider removing argc and argv for non-early
        // registration. Unfortunately that would require a new InitFunction
        // typedef, so not making the change right now.
        //
        // Note that init doesn't receive argc and argv, so the function
        // might fail and we want to raise an error in that case.
        int argc = 0;
        char** argv = nullptr;
        bool success = (function)(&argc, &argv);
        CAFFE_ENFORCE(success);
      } else {
        // Wait until GlobalInit to run
        init_functions_.emplace_back(function, description);
      }
    }
  }

  bool RunRegisteredEarlyInitFunctions(int* pargc, char*** pargv) {
    CAFFE_ENFORCE(!early_init_functions_run_yet_);
    early_init_functions_run_yet_ = true;
    return RunRegisteredInitFunctionsInternal(
        early_init_functions_, pargc, pargv);
  }

  bool RunRegisteredInitFunctions(int* pargc, char*** pargv) {
    CAFFE_ENFORCE(!init_functions_run_yet_);
    init_functions_run_yet_ = true;
    return RunRegisteredInitFunctionsInternal(init_functions_, pargc, pargv);
  }

 private:
  // Run all registered initialization functions. This has to be called AFTER
  // all static initialization are finished and main() has started, since we are
  // using logging.
  bool RunRegisteredInitFunctionsInternal(
      vector<std::pair<InitFunction, const char*>>& functions,
      int* pargc, char*** pargv) {
    for (const auto& init_pair : functions) {
      VLOG(1) << "Running init function: " << init_pair.second;
      if (!(*init_pair.first)(pargc, pargv)) {
        LOG(ERROR) << "Initialization function failed.";
        return false;
      }
    }
    return true;
  }

  Caffe2InitializeRegistry() {}
  vector<std::pair<InitFunction, const char*> > early_init_functions_;
  vector<std::pair<InitFunction, const char*> > init_functions_;
  bool early_init_functions_run_yet_ = false;
  bool init_functions_run_yet_ = false;
};
}  // namespace internal

class CAFFE2_API InitRegisterer {
 public:
  InitRegisterer(internal::Caffe2InitializeRegistry::InitFunction function,
                 bool run_early, const char* description) {
    internal::Caffe2InitializeRegistry::Registry()
        ->Register(function, run_early, description);
  }
};

#define REGISTER_CAFFE2_INIT_FUNCTION(name, function, description)             \
  namespace {                                                                  \
  ::caffe2::InitRegisterer g_caffe2_initregisterer_##name(                     \
      function, false, description);                                           \
  }  // namespace

#define REGISTER_CAFFE2_EARLY_INIT_FUNCTION(name, function, description)       \
  namespace {                                                                  \
  ::caffe2::InitRegisterer g_caffe2_initregisterer_##name(                     \
      function, true, description);                                            \
  }  // namespace

/**
 * @brief Determine whether GlobalInit has already been run
 */
CAFFE2_API bool GlobalInitAlreadyRun();

class CAFFE2_API GlobalInitIsCalledGuard {
 public:
  GlobalInitIsCalledGuard() {
    if (!GlobalInitAlreadyRun()) {
      LOG(WARNING)
          << "Caffe2 GlobalInit should be run before any other API calls.";
    }
  }
};

/**
 * @brief Initialize the global environment of caffe2.
 *
 * Caffe2 uses a registration pattern for initialization functions. Custom
 * initialization functions should take the signature
 *     bool (*func)(int*, char***)
 * where the pointers to argc and argv are passed in. Caffe2 then runs the
 * initialization in three phases:
 * (1) Functions registered with REGISTER_CAFFE2_EARLY_INIT_FUNCTION. Note that
 *     since it is possible the logger is not initialized yet, any logging in
 *     such early init functions may not be printed correctly.
 * (2) Parses Caffe-specific commandline flags, and initializes caffe logging.
 * (3) Functions registered with REGISTER_CAFFE2_INIT_FUNCTION.
 * If there is something wrong at each stage, the function returns false. If
 * the global initialization has already been run, the function returns false
 * as well.
 *
 * GlobalInit is re-entrant safe; a re-entrant call will no-op and exit.
 *
 * GlobalInit is safe to call multiple times but not idempotent;
 * successive calls will parse flags and re-set caffe2 logging levels from
 * flags as needed, but NOT re-run early init and init functions.
 *
 * GlobalInit is also thread-safe and can be called concurrently.
 */
CAFFE2_API bool GlobalInit(int* pargc, char*** argv);

/**
 * @brief Initialize the global environment without command line arguments
 *
 * This is a version of the GlobalInit where no argument is passed in.
 * On mobile devices, use this global init, since we cannot pass the
 * command line options to caffe2, no arguments are passed.
 */
CAFFE2_API bool GlobalInit();
}  // namespace caffe2
#endif  // CAFFE2_CORE_INIT_H_
