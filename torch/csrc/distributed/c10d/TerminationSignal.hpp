#pragma once

#include <signal.h>
#include <atomic>

namespace c10d::detail {

inline std::atomic<bool> terminationSignalReceived{false};
inline struct sigaction previousSigint;
inline struct sigaction previousSigterm;

inline void handleSignal(int signum) {
  terminationSignalReceived = true;
  switch (signum) {
    case SIGINT:
      if (previousSigint.sa_handler) {
        previousSigint.sa_handler(signum);
      }
      break;
    case SIGTERM:
      if (previousSigterm.sa_handler) {
        previousSigterm.sa_handler(signum);
      }
      break;
  }
}

inline bool isTerminationSignalReceived() {
  return terminationSignalReceived.load(std::memory_order_acquire);
}

inline void installTerminationHandlers() {
  struct sigaction sa{};
  sa.sa_handler = &handleSignal;
  sigfillset(&sa.sa_mask);

  struct sigaction old;
  sigaction(SIGINT, &sa, &old);
  if (old.sa_handler != &handleSignal) {
    previousSigint = old;
  }
  sigaction(SIGTERM, &sa, &old);
  if (old.sa_handler != &handleSignal) {
    previousSigterm = old;
  }
}
} // namespace c10d::detail
