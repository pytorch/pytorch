#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include <c10/macros/Macros.h>

namespace torch {
namespace monitor {


class WaitCounterImpl {

};

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

 private:
  const std::string key_;
  std::shared_ptr<WaitCounterImpl> impl_;
};
} // namespace monitor
} // namespace torch

#define WAIT_COUNTER(_key)                                                  \
  []() {                                                                    \
    static torch::monitor::WaitCounterHandle handle(#_key);                 \
    return handle;                                                          \
  }()

#define SCOPED_WAIT_COUNTER(_name)                                          \
  auto C10_ANONYMOUS_VARIABLE(SCOPED_WAIT_COUNTER) =                        \
      WAIT_COUNTER(_name)                                                   \
  WAIT_COUNTER_US(_name).start();                                           \
  auto guard = c10::make_scope_exit(                                        \
    [&]() {                                                                 \
      WAIT_COUNTER_US(_name).stop();                                        \
    }                                                                       \
  );
