#ifndef CAFFE2_CORE_INIT_H_
#define CAFFE2_CORE_INIT_H_

#include "caffe2/core/common.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

namespace internal {
class Caffe2InitializeRegistry {
 public:
  typedef bool (*InitFunction)(void);
  static Caffe2InitializeRegistry* Registry() {
    static Caffe2InitializeRegistry gRegistry;
    return &gRegistry;
  }

  void Register(InitFunction function, const char* description) {
    init_functions_.emplace_back(function, description);
  }

  // Run all registered initialization functions. This has to be called AFTER
  // all static initialization are finished and main() has started, since we are
  // using logging.
  bool RunRegisteredInitFunctions() {
    for (const auto& init_pair : init_functions_) {
      CAFFE_LOG_INFO << "Running init function: " << init_pair.second;
      if (!(*init_pair.first)()) {
        CAFFE_LOG_ERROR << "Initialization function failed.";
        return false;
      }
    }
    return true;
  }

 private:
  Caffe2InitializeRegistry() : init_functions_() {}
  vector<std::pair<InitFunction, const char*> > init_functions_;
};
}  // namespace internal

class InitRegisterer {
 public:
  InitRegisterer(internal::Caffe2InitializeRegistry::InitFunction function,
                 const char* description) {
    internal::Caffe2InitializeRegistry::Registry()
        ->Register(function, description);
  }
};

#define REGISTER_CAFFE2_INIT_FUNCTION(name, function, description)             \
  namespace {                                                                  \
  ::caffe2::InitRegisterer g_caffe2_initregisterer_name(                       \
      function, description);                                                  \
  }  // namespace

// A global initialization stream where one can write messages to. After the
// GlobalInit() function finishes, all the messages written into the global
// init stream is written to LOG_INFO, and in case of error, LOG_ERROR.
std::stringstream& GlobalInitStream();

// Initialize the global environment of caffe2.
bool GlobalInit(int* pargc, char** argv);


}  // namespace caffe2
#endif  // CAFFE2_CORE_INIT_H_
