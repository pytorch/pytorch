#pragma once

namespace caffe2 {

class SignalHandler {
 public:
  enum class Action {
    NONE,
    STOP
  };

  // Contructor. Specify what action to take when a signal is received.
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

#if defined(__linux__) && !defined(__ANDROID__)
#define CAFFE2_SUPPORTS_SIGNAL_HANDLER
#endif

#if defined(CAFFE2_SUPPORTS_SIGNAL_HANDLER)
namespace internal {     
bool Caffe2InitFatalSignalHandler(int* pargc, char*** pargv);      
} // namespace internal       
#endif // defined(CAFFE2_SUPPORTS_SIGNAL_HANDLER)

}  // namespace caffe2
