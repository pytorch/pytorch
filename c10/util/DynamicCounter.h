#pragma once

#include <functional>
#include <memory>
#include <string_view>

#include <c10/macros/Macros.h>

namespace c10::monitor {

class C10_API DynamicCounter {
 public:
  using Callback = std::function<int64_t()>;

  // Creates a dynamic counter that can be queried at any point in time by
  // multiple backends. Only one counter with a given key can exist at any point
  // in time.
  //
  // The callback is invoked every time the counter is queried.
  // The callback must be thread-safe.
  // The callback must not throw.
  // The callback must not block.
  DynamicCounter(std::string_view key, Callback getCounterCallback);

  // Unregisters the callback.
  // Waits for all ongoing callback invocations to finish.
  ~DynamicCounter();

 private:
  struct Guard;
  std::unique_ptr<Guard> guard_;
};

namespace detail {
class DynamicCounterBackendIf {
 public:
  virtual ~DynamicCounterBackendIf() = default;

  virtual void registerCounter(
      std::string_view key,
      DynamicCounter::Callback getCounterCallback) = 0;
  // MUST wait for all ongoing callback invocations to finish
  virtual void unregisterCounter(std::string_view key) = 0;
};

void C10_API
    registerDynamicCounterBackend(std::unique_ptr<DynamicCounterBackendIf>);
} // namespace detail
} // namespace c10::monitor
