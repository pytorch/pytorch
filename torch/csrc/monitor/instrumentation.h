#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include <c10/macros/Macros.h>
#include <c10/util/ScopeExit.h>

namespace torch {
namespace monitor {

// A handle to a wait counter.
class WaitCounterHandle {
 public:
  explicit WaitCounterHandle(std::string_view key);
  ~WaitCounterHandle();

  // Starts a waiter
  void start(
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());
  // Stops the waiter. Each start() call should be matched by exactly one stop()
  // call.
  void stop(
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());

  struct State;

 private:
  const std::string key_;
  std::shared_ptr<State> state_;
};
} // namespace monitor
} // namespace torch

#define STATIC_WAIT_COUNTER(_key)                           \
  []() {                                                    \
    static torch::monitor::WaitCounterHandle handle(#_key); \
    return handle;                                          \
  }()

#define STATIC_SCOPED_WAIT_COUNTER(_name)    \
  STATIC_WAIT_COUNTER(_name).start();        \
  auto C10_ANONYMOUS_VARIABLE(SCOPE_GUARD) = \
      c10::make_scope_exit([&]() { STATIC_WAIT_COUNTER(_name).stop(); });
