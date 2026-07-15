#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/macros/Export.h>

// The watchdog is only built with the libuv timer backend. Without it the types
// below are not declared, so any use is a compile error (and the Python
// bindings are not registered, so importing them raises ImportError).
#ifdef TORCH_USE_LIBUV

namespace c10d::watchdog {

// Callback invoked from the watchdog's timer loop thread. Callbacks must not
// block, since all timeouts are serviced on that single thread.
using Callback = std::function<void()>;

class Watchdog;

// Handle to a registered monitor. cancel() removes it (idempotent). For an
// op_timeout, completed() records the operation's end event on the handle's
// stream, marking the enqueue done; it is a no-op for other monitor kinds.
class TORCH_API Handle {
 public:
  Handle() = default;
  void cancel() const;
  void completed() const;

 private:
  friend class Watchdog;
  Handle(
      std::weak_ptr<Watchdog> watchdog,
      uint64_t id,
      std::optional<c10::Stream> stream = std::nullopt)
      : watchdog_(std::move(watchdog)), id_(id), stream_(stream) {}

  std::weak_ptr<Watchdog> watchdog_;
  uint64_t id_{0};
  // Set for op_timeout handles: the stream the end event is recorded on.
  std::optional<c10::Stream> stream_;
};

// A process-wide timer/timeout service backed by a libuv event loop running on
// a dedicated background thread.
//
// This is the interface; the implementation (which owns the libuv loop) derives
// from it and is created via makeWatchdog(). A single global instance is
// available via singleton(). Instances must be owned by a std::shared_ptr (which
// makeWatchdog and singleton return) so the handles they hand out can refer back
// to them; keep that shared_ptr alive across calls (do not race a call against
// the last reference being dropped).
class TORCH_API Watchdog : public std::enable_shared_from_this<Watchdog> {
 public:
  virtual ~Watchdog() = default;
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;
  Watchdog(Watchdog&&) = delete;
  Watchdog& operator=(Watchdog&&) = delete;

  // Process-wide instance. Intentionally leaked so the background thread is
  // never joined during interpreter shutdown.
  static const std::shared_ptr<Watchdog>& singleton();

  // Fire callback after timeout elapses, unless the returned handle is
  // cancelled first.
  virtual Handle cpu_timeout(
      std::chrono::milliseconds timeout,
      Callback callback) = 0;

  // Record an event on stream now and fire callback if the work enqueued up to
  // this point has not completed within timeout. The monitor removes itself once
  // the event completes (success) or the callback fires. cancel() stops it.
  virtual Handle stream_timeout(
      c10::Stream stream,
      std::chrono::milliseconds timeout,
      Callback callback) = 0;

  // Record an event on stream now and fire callback once the work enqueued up
  // to this point has completed. The monitor removes itself after firing.
  // cancel() stops it.
  virtual Handle stream_completed(c10::Stream stream, Callback callback) = 0;

  // Monitor a device operation in two phases. A start event is recorded on
  // stream now: if it does not complete within timeout (the operation never
  // starts executing) callback fires; once it completes, a timeout clock begins
  // and, if the end event recorded by Handle::completed() has not completed
  // within timeout, callback fires. callback fires at most once and the monitor
  // always removes itself (no leak if the operation never starts). cancel()
  // stops it.
  virtual Handle op_timeout(
      c10::Stream stream,
      std::chrono::milliseconds timeout,
      Callback callback) = 0;

  // Number of monitors currently active. Primarily for tests.
  virtual size_t numActiveStreamTimeouts() const = 0;

 protected:
  Watchdog() = default;
  // Constructs a Handle bound to this watchdog (Handle's constructor is private
  // to Watchdog, so implementations mint handles through here).
  Handle makeHandle(uint64_t id, std::optional<c10::Stream> stream = std::nullopt);

 private:
  friend class Handle;
  virtual void cancel(uint64_t id) = 0;
  virtual void markCompleted(uint64_t id, const c10::Stream& stream) = 0;
};

// Creates a new watchdog instance. pollInterval controls how often device
// events are polled for stream/op timeouts. Primarily for tests that want an
// isolated instance (and a short interval); production code should use
// singleton().
TORCH_API std::shared_ptr<Watchdog> makeWatchdog(
    std::chrono::milliseconds pollInterval = std::chrono::seconds(1));

} // namespace c10d::watchdog

#endif // TORCH_USE_LIBUV
