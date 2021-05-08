#include <c10/util/Backtrace.h>
#include <c10/util/signal_handler.h>
#include <fmt/format.h>

#if defined(C10_SUPPORTS_SIGNAL_HANDLER)

// Normal signal handler implementation.
#include <cxxabi.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

#ifdef C10_ANDROID
#ifndef SYS_gettid
#define SYS_gettid __NR_gettid
#endif
#ifndef SYS_tgkill
#define SYS_tgkill __NR_tgkill
#endif
#endif

namespace {

struct sigaction previousSighup;
struct sigaction previousSigint;
std::atomic<int> sigintCount(0);
std::atomic<int> sighupCount(0);
std::atomic<int> hookedUpCount(0);

void handleSignal(int signal) {
  switch (signal) {
    // TODO: what if the previous handler uses sa_sigaction?
    case SIGHUP:
      sighupCount += 1;
      if (previousSighup.sa_handler) {
        previousSighup.sa_handler(signal);
      }
      break;
    case SIGINT:
      sigintCount += 1;
      if (previousSigint.sa_handler) {
        previousSigint.sa_handler(signal);
      }
      break;
  }
}

void hookupHandler() {
  if (hookedUpCount++) {
    return;
  }
  struct sigaction sa;
  // Setup the handler
  sa.sa_handler = &handleSignal;
  // Restart the system call, if at all possible
  sa.sa_flags = SA_RESTART;
  // Block every signal during the handler
  sigfillset(&sa.sa_mask);
  // Intercept SIGHUP and SIGINT
  if (sigaction(SIGHUP, &sa, &previousSighup) == -1) {
    LOG(FATAL) << "Cannot install SIGHUP handler.";
  }
  if (sigaction(SIGINT, &sa, &previousSigint) == -1) {
    LOG(FATAL) << "Cannot install SIGINT handler.";
  }
}

// Set the signal handlers to the default.
void unhookHandler() {
  if (--hookedUpCount > 0) {
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
  if (sigaction(SIGHUP, &previousSighup, nullptr) == -1) {
    LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
  }
  if (sigaction(SIGINT, &previousSigint, nullptr) == -1) {
    LOG(FATAL) << "Cannot uninstall SIGINT handler.";
  }
}

} // namespace

namespace c10 {

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)

FatalSignalHandler& FatalSignalHandler::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static FatalSignalHandler* handler = new FatalSignalHandler();
  return *handler;
}

FatalSignalHandler::~FatalSignalHandler() {}

FatalSignalHandler::FatalSignalHandler()
    : fatalSignalHandlersInstalled(false),
      fatalSignalReceived(false),
      fatalSignalName("<UNKNOWN>"),
      writingCond(PTHREAD_COND_INITIALIZER),
      writingMutex(PTHREAD_MUTEX_INITIALIZER) {}

FatalSignalHandler::signal_handler FatalSignalHandler::kSignalHandlers[] = {
    {"SIGABRT", SIGABRT, {}},
    {"SIGINT", SIGINT, {}},
    {"SIGILL", SIGILL, {}},
    {"SIGFPE", SIGFPE, {}},
    {"SIGBUS", SIGBUS, {}},
    {"SIGSEGV", SIGSEGV, {}},
    {nullptr, 0, {}}};

struct sigaction* FatalSignalHandler::getPreviousSigaction(int signum) {
  for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return &handler->previous;
    }
  }
  return nullptr;
}

const char* FatalSignalHandler::getSignalName(int signum) {
  for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return handler->name;
    }
  }
  return nullptr;
}

void FatalSignalHandler::callPreviousSignalHandler(
    struct sigaction* action,
    int signum,
    siginfo_t* info,
    void* ctx) {
  if (!action->sa_handler) {
    return;
  }
  if ((action->sa_flags & SA_SIGINFO) == SA_SIGINFO) {
    action->sa_sigaction(signum, info, ctx);
  } else {
    action->sa_handler(signum);
  }
}

// needsLock signals whether we need to lock our writing mutex.
void FatalSignalHandler::stacktraceSignalHandler(bool needsLock) {
  if (needsLock) {
    pthread_mutex_lock(&writingMutex);
  }
  pid_t tid = syscall(SYS_gettid);
  std::string backtrace = fmt::format(
      "{}({}), PID: {}, Thread {}: \n {}",
      fatalSignalName,
      fatalSignum,
      ::getpid(),
      tid,
      c10::get_backtrace());
  std::cerr << backtrace << std::endl;
  if (needsLock) {
    pthread_mutex_unlock(&writingMutex);
    pthread_cond_signal(&writingCond);
  }
}

void FatalSignalHandler::fatalSignalHandlerPostProcess() {}

void FatalSignalHandler::fatalSignalHandlerStatic(int signum) {
  getInstance().fatalSignalHandler(signum);
}

// Our fatal signal entry point
void FatalSignalHandler::fatalSignalHandler(int signum) {
  // Check if this is a proper signal that we declared above.
  const char* name = getSignalName(signum);
  if (!name) {
    return;
  }
  if (fatalSignalReceived) {
    return;
  }
  // Set the flag so that our SIGUSR2 handler knows that we're aborting and
  // that it should intercept any SIGUSR2 signal.
  fatalSignalReceived = true;
  // Set state for other threads.
  fatalSignum = signum;
  fatalSignalName = name;
  // Linux doesn't have a nice userland API for enumerating threads so we
  // need to use the proc pseudo-filesystem.
  DIR* procDir = opendir("/proc/self/task");
  if (procDir) {
    pid_t pid = getpid();
    pid_t currentTid = syscall(SYS_gettid);
    struct dirent* entry;
    pthread_mutex_lock(&writingMutex);
    while ((entry = readdir(procDir)) != nullptr) {
      if (entry->d_name[0] == '.') {
        continue;
      }
      pid_t tid = atoi(entry->d_name);
      // If we've found the current thread then we'll jump into the SIGUSR2
      // handler before calling pthread_cond_wait thus deadlocking, so branch
      // our directly to the backtrace handler instead of signaling it.
      if (tid != currentTid) {
        syscall(SYS_tgkill, pid, tid, SIGUSR2);
        pthread_cond_wait(&writingCond, &writingMutex);
      } else {
        stacktraceSignalHandler(false);
      }
    }
    pthread_mutex_unlock(&writingMutex);
  } else {
    perror("Failed to open /proc/self/task");
  }
  fatalSignalHandlerPostProcess();
  sigaction(signum, getPreviousSigaction(signum), nullptr);
  raise(signum);
}

// Our SIGUSR2 entry point
void FatalSignalHandler::stacktraceSignalHandlerStatic(
    int signum,
    siginfo_t* info,
    void* ctx) {
  getInstance().stacktraceSignalHandler(signum, info, ctx);
}

void FatalSignalHandler::stacktraceSignalHandler(
    int signum,
    siginfo_t* info,
    void* ctx) {
  if (fatalSignalReceived) {
    stacktraceSignalHandler(true);
  } else {
    // We don't want to actually change the signal handler as we want to
    // remain the signal handler so that we may get the usr2 signal later.
    callPreviousSignalHandler(&previousSigusr2, signum, info, ctx);
  }
}

// Installs SIGABRT signal handler so that we get stack traces
// from every thread on SIGABRT caused exit. Also installs SIGUSR2 handler
// so that threads can communicate with each other (be sure if you use SIGUSR2)
// to install your handler before initing caffe2 (we properly fall back to
// the previous handler if we didn't initiate the SIGUSR2).
void FatalSignalHandler::installFatalSignalHandlers() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  if (fatalSignalHandlersInstalled) {
    return;
  }
  fatalSignalHandlersInstalled = true;
  struct sigaction sa;
  sigemptyset(&sa.sa_mask);
  // Since we'll be in an exiting situation it's possible there's memory
  // corruption, so make our own stack just in case.
  sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
  sa.sa_handler = FatalSignalHandler::fatalSignalHandlerStatic;
  for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (sigaction(handler->signum, &sa, &handler->previous)) {
      std::string str("Failed to add ");
      str += handler->name;
      str += " handler!";
      perror(str.c_str());
    }
  }
  sa.sa_sigaction = FatalSignalHandler::stacktraceSignalHandlerStatic;
  if (sigaction(SIGUSR2, &sa, &previousSigusr2)) {
    perror("Failed to add SIGUSR2 handler!");
  }
}

void FatalSignalHandler::uninstallFatalSignalHandlers() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  if (!fatalSignalHandlersInstalled) {
    return;
  }
  fatalSignalHandlersInstalled = false;
  for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (sigaction(handler->signum, &handler->previous, nullptr)) {
      std::string str("Failed to remove ");
      str += handler->name;
      str += " handler!";
      perror(str.c_str());
    } else {
      handler->previous = {};
    }
  }
  if (sigaction(SIGUSR2, &previousSigusr2, nullptr)) {
    perror("Failed to add SIGUSR2 handler!");
  } else {
    previousSigusr2 = {};
  }
}
#endif // defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)

SignalHandler::SignalHandler(
    SignalHandler::Action SIGINT_action,
    SignalHandler::Action SIGHUP_action)
    : SIGINT_action_(SIGINT_action),
      SIGHUP_action_(SIGHUP_action),
      my_sigint_count_(sigintCount),
      my_sighup_count_(sighupCount) {
  hookupHandler();
}

SignalHandler::~SignalHandler() {
  unhookHandler();
}

// Return true iff a SIGINT has been received since the last time this
// function was called.
bool SignalHandler::GotSIGINT() {
  uint64_t count = sigintCount;
  bool result = (count != my_sigint_count_);
  my_sigint_count_ = count;
  return result;
}

// Return true iff a SIGHUP has been received since the last time this
// function was called.
bool SignalHandler::GotSIGHUP() {
  uint64_t count = sighupCount;
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

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
void FatalSignalHandler::setPrintStackTracesOnFatalSignal(bool print) {
  if (print) {
    installFatalSignalHandlers();
  } else {
    uninstallFatalSignalHandlers();
  }
}
bool FatalSignalHandler::printStackTracesOnFatalSignal() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  return fatalSignalHandlersInstalled;
}

#endif // defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
} // namespace c10

#else // defined(C10_SUPPORTS_SIGNAL_HANDLER)

// TODO: Currently we do not support signal handling in non-Linux yet - below is
// a minimal implementation that makes things compile.
namespace c10 {
SignalHandler::SignalHandler(
    SignalHandler::Action SIGINT_action,
    SignalHandler::Action SIGHUP_action) {
  SIGINT_action_ = SIGINT_action;
  SIGHUP_action_ = SIGHUP_action;
  my_sigint_count_ = 0;
  my_sighup_count_ = 0;
}
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
} // namespace c10

#endif // defined(C10_SUPPORTS_SIGNAL_HANDLER)
