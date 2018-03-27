#ifndef CAFFE2_CORE_INIT_H_
#define CAFFE2_CORE_INIT_H_

#include "caffe2/core/common.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

namespace internal {
class Caffe2InitializeRegistry {
 public:
  typedef bool (*InitFunction)(int*, char***);
  // Registry() is defined in .cpp file to make registration work across
  // multiple shared libraries loaded with RTLD_LOCAL
  static Caffe2InitializeRegistry* Registry();

  void Register(InitFunction function, bool run_early,
                const char* description) {
    if (run_early) {
      early_init_functions_.emplace_back(function, description);
    } else {
      init_functions_.emplace_back(function, description);
    }
  }

  bool RunRegisteredEarlyInitFunctions(int* pargc, char*** pargv) {
    return RunRegisteredInitFunctionsInternal(
        early_init_functions_, pargc, pargv);
  }

  bool RunRegisteredInitFunctions(int* pargc, char*** pargv) {
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
};
}  // namespace internal

class InitRegisterer {
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
 */
bool GlobalInit(int* pargc, char*** argv);

}  // namespace caffe2
#endif  // CAFFE2_CORE_INIT_H_
