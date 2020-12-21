#pragma once

#include "caffe2/core/common.h"

#if defined(__APPLE__)
#define CAFFE2_SUPPORTS_SIGNAL_HANDLER
#elif defined(__linux__) && !defined(CAFFE2_DISABLE_SIGNAL_HANDLERS)
#define CAFFE2_SUPPORTS_FATAL_SIGNAL_HANDLERS
#define CAFFE2_SUPPORTS_SIGNAL_HANDLER
#endif

namespace caffe2 {

class TORCH_API SignalHandler {
 public:
  enum class Action {
    NONE,
    STOP
  };

  // Constructor. Specify what action to take when a signal is received.
  SignalHandler(Action SIGINT_action,
                Action SIGHUP_action);
  ~SignalHandler();

  Action CheckForSignals();

 private:
  bool GotSIGINT();
  bool GotSIGHUP();
  Action SIGINT_action_;
  Action SIGHUP_action_;
  unsigned long my_sigint_count_;
  unsigned long my_sighup_count_;
};

#if defined(CAFFE2_SUPPORTS_FATAL_SIGNAL_HANDLERS)
// This works by setting up certain fatal signal handlers. Previous fatal
// signal handlers will still be called when the signal is raised. Defaults
// to being off.
TORCH_API void setPrintStackTracesOnFatalSignal(bool print);
TORCH_API bool printStackTracesOnFatalSignal();
#endif // defined(CAFFE2_SUPPORTS_SIGNAL_HANDLER)

}  // namespace caffe2
