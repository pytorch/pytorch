#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include <c10/macros/Macros.h>
#include <c10/util/ScopeExit.h>

namespace torch {
namespace monitor {
namespace detail {
class WaitCounterImpl;
}

// A handle to a wait counter.
class WaitCounterHandle {
 public:
  explicit WaitCounterHandle(std::string_view key);

  class WaitGuard {
   public:
    ~WaitGuard() {
      stop();
    }

    void stop() {
      if (auto handle = std::exchange(handle_, nullptr)) {
        handle->stop(ctx_);
      }
    }

   private:
    WaitGuard(WaitCounterHandle& handle, intptr_t ctx)
        : handle_{&handle}, ctx_{ctx} {}

    friend class WaitCounterHandle;

    WaitCounterHandle* handle_;
    intptr_t ctx_;
  };

  // Starts a waiter
  WaitGuard start();

 private:
  // Stops the waiter. Each start() call should be matched by exactly one stop()
  // call.
  void stop(intptr_t ctx);

  detail::WaitCounterImpl& impl_;
};
} // namespace monitor
} // namespace torch

#define STATIC_WAIT_COUNTER(_key)                           \
  []() -> torch::monitor::WaitCounterHandle& {              \
    static torch::monitor::WaitCounterHandle handle(#_key); \
    return handle;                                          \
  }()

#define STATIC_SCOPED_WAIT_COUNTER(_name) \
  auto C10_ANONYMOUS_VARIABLE(SCOPE_GUARD) = STATIC_WAIT_COUNTER(_name).start();
