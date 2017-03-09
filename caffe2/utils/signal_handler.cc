#include "caffe2/utils/signal_handler.h"
#include "caffe2/core/logging.h"

#if defined(_MSC_VER)

// Currently we do not support signal handling in Windows yet - below is a
// minimal implementation that makes things compile.
namespace caffe2 {
SignalHandler::SignalHandler(
    SignalHandler::Action SIGINT_action,
    SignalHandler::Action SIGHUP_action) {}
SignalHandler::~SignalHandler() {}
bool SignalHandler::GotSIGINT() {
  return false;
}
bool SignalHandler::GotSIGHUP() {
  return false;
}
SignalHandler::Action SignalHandler::CheckForSignals() {
  return SignalHandler::Action::NONE;
}
} // namespace caffe2

#else // defined(_MSC_VER)

// Normal signal handler implementation.

#include <atomic>
#include <signal.h>
#include <unordered_set>

namespace {

  static struct sigaction previous_sighup;
  static struct sigaction previous_sigint;
  static volatile sig_atomic_t sigint_count = 0;
  static volatile sig_atomic_t sighup_count = 0;
  static std::atomic<int> hooked_up_count{0};

  void handle_signal(int signal) {
    switch (signal) {
    case SIGHUP:
      sighup_count += 1;
      if (previous_sighup.sa_handler) {
        previous_sighup.sa_handler(signal);
      }
      break;
    case SIGINT:
      sigint_count += 1;
      if (previous_sigint.sa_handler) {
        previous_sigint.sa_handler(signal);
      }
      break;
    }
  }

  void HookupHandler() {
    if (hooked_up_count++) {
      return;
    }
    struct sigaction sa;
    // Setup the handler
    sa.sa_handler = &handle_signal;
    // Restart the system call, if at all possible
    sa.sa_flags = SA_RESTART;
    // Block every signal during the handler
    sigfillset(&sa.sa_mask);
    // Intercept SIGHUP and SIGINT
    if (sigaction(SIGHUP, &sa, nullptr) == -1) {
      LOG(FATAL) << "Cannot install SIGHUP handler.";
    }
    if (sigaction(SIGINT, &sa, nullptr) == -1) {
      LOG(FATAL) << "Cannot install SIGINT handler.";
    }
  }

  // Set the signal handlers to the default.
  void UnhookHandler() {
    if (--hooked_up_count > 0) {
      return;
    }
    struct sigaction sa;
    // Setup the sighub handler
    sa.sa_handler = SIG_DFL;
    // Restart the system call, if at all possible
    sa.sa_flags = SA_RESTART;
    // Block every signal during the handler
    sigfillset(&sa.sa_mask);
    // Intercept SIGHUP and SIGINT
    if (sigaction(SIGHUP, &sa, &previous_sighup) == -1) {
      LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
    }
    if (sigaction(SIGINT, &sa, &previous_sigint) == -1) {
      LOG(FATAL) << "Cannot uninstall SIGINT handler.";
    }
  }

}  // namespace

namespace caffe2 {

SignalHandler::SignalHandler(SignalHandler::Action SIGINT_action,
                             SignalHandler::Action SIGHUP_action):
  SIGINT_action_(SIGINT_action),
  SIGHUP_action_(SIGHUP_action),
  my_sigint_count_(sigint_count),
  my_sighup_count_(sighup_count) {
  HookupHandler();
}

SignalHandler::~SignalHandler() {
  UnhookHandler();
}

// Return true iff a SIGINT has been received since the last time this
// function was called.
bool SignalHandler::GotSIGINT() {
  uint64_t count = sigint_count;
  bool result = (count != my_sigint_count_);
  my_sigint_count_ = count;
  return result;
}

// Return true iff a SIGHUP has been received since the last time this
// function was called.
bool SignalHandler::GotSIGHUP() {
  uint64_t count = sighup_count;
  bool result = (count != my_sighup_count_);
  my_sighup_count_ = count;
  return result;
}


SignalHandler::Action SignalHandler::CheckForSignals() {
  if (GotSIGHUP()) {
    return SIGHUP_action_;
  }
  if (GotSIGINT()) {
    return SIGINT_action_;
  }
  return SignalHandler::Action::NONE;
}

}  // namespace caffe2

#endif // defined(_MSC_VER)
