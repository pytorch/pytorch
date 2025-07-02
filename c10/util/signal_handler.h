#pragma once

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <mutex>

#include <c10/macros/Export.h>

#if defined(__APPLE__)
#define C10_SUPPORTS_SIGNAL_HANDLER
#elif defined(__linux__) && !defined(C10_DISABLE_SIGNAL_HANDLERS)
#define C10_SUPPORTS_FATAL_SIGNAL_HANDLERS
#define C10_SUPPORTS_SIGNAL_HANDLER
#endif

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
#include <pthread.h>
#endif

namespace c10 {

class C10_API SignalHandler {
 public:
  enum class Action { NONE, STOP };

  // Constructor. Specify what action to take when a signal is received.
  SignalHandler(Action SIGINT_action, Action SIGHUP_action);

  SignalHandler(const SignalHandler&) = delete;
  SignalHandler(SignalHandler&&) = delete;
  SignalHandler& operator=(const SignalHandler&) = delete;
  SignalHandler& operator=(SignalHandler&&) = delete;
  ~SignalHandler();

  Action CheckForSignals();

  bool GotSIGINT();
  bool GotSIGHUP();

  Action SIGINT_action_;
  Action SIGHUP_action_;
  std::atomic<uint64_t> my_sigint_count_;
  std::atomic<uint64_t> my_sighup_count_;
};

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
class C10_API FatalSignalHandler {
  // This works by setting up certain fatal signal handlers. Previous fatal
  // signal handlers will still be called when the signal is raised. Defaults
  // to being off.
 public:
  C10_API void setPrintStackTracesOnFatalSignal(bool print);
  C10_API bool printStackTracesOnFatalSignal();
  static FatalSignalHandler& getInstance();
  FatalSignalHandler(const FatalSignalHandler&) = delete;
  FatalSignalHandler(FatalSignalHandler&&) = delete;
  FatalSignalHandler& operator=(const FatalSignalHandler&) = delete;
  FatalSignalHandler& operator=(FatalSignalHandler&&) = delete;
  virtual ~FatalSignalHandler() = default;

 protected:
  explicit FatalSignalHandler();

 private:
  void installFatalSignalHandlers();
  void uninstallFatalSignalHandlers();
  static void fatalSignalHandlerStatic(int signum);
  void fatalSignalHandler(int signum);
  virtual void fatalSignalHandlerPostProcess();
  struct sigaction* getPreviousSigaction(int signum);
  const char* getSignalName(int signum);
  void callPreviousSignalHandler(
      struct sigaction* action,
      int signum,
      siginfo_t* info,
      void* ctx);
  void stacktraceSignalHandler(bool needsLock);
  static void stacktraceSignalHandlerStatic(
      int signum,
      siginfo_t* info,
      void* ctx);
  void stacktraceSignalHandler(int signum, siginfo_t* info, void* ctx);

  // The mutex protects the bool.
  std::mutex fatalSignalHandlersInstallationMutex;
  bool fatalSignalHandlersInstalled;
  // We need to hold a reference to call the previous SIGUSR2 handler in case
  // we didn't signal it
  struct sigaction previousSigusr2{};
  // Flag dictating whether the SIGUSR2 handler falls back to previous handlers
  // or is intercepted in order to print a stack trace.
  std::atomic<bool> fatalSignalReceived;
  // Global state set when a fatal signal is received so that backtracing
  // threads know why they're printing a stacktrace.
  const char* fatalSignalName;
  int fatalSignum = -1;
  // This wait condition is used to wait for other threads to finish writing
  // their stack trace when in fatal sig handler (we can't use pthread_join
  // because there's no way to convert from a tid to a pthread_t).
  std::condition_variable writingCond;
  std::mutex writingMutex;
  // used to indicate if the other thread responded to the signal
  bool signalReceived;

  struct signal_handler {
    const char* name;
    int signum;
    struct sigaction previous;
  };

  // NOLINTNEXTLINE(*c-arrays*)
  static signal_handler kSignalHandlers[];
};

#endif // defined(C10_SUPPORTS_SIGNAL_HANDLER)

} // namespace c10
