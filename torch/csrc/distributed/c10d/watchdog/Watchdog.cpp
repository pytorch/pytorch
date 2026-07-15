#include <torch/csrc/distributed/c10d/watchdog/Watchdog.hpp>

#ifdef TORCH_USE_LIBUV

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>
#include <c10/util/thread_name.h>
#include <uv.h>

namespace c10d::watchdog {

namespace {

// Queries an event without throwing. c10::Event::query() (cudaEventQuery) throws
// while another thread holds a CUDA-graph capture in the default (global) mode;
// treat that (and any transient query error) as "not completed" so the poll
// thread neither aborts nor interferes with capture. The deadline check still
// runs, so a genuinely stuck operation past its deadline still fires.
bool queryCompleted(const c10::Event& event) {
  try {
    return event.query();
  } catch (const std::exception&) {
    return false;
  }
}

class WatchdogImpl : public Watchdog {
 public:
  explicit WatchdogImpl(std::chrono::milliseconds pollInterval);
  ~WatchdogImpl() override;
  WatchdogImpl(const WatchdogImpl&) = delete;
  WatchdogImpl& operator=(const WatchdogImpl&) = delete;
  WatchdogImpl(WatchdogImpl&&) = delete;
  WatchdogImpl& operator=(WatchdogImpl&&) = delete;

  Handle cpu_timeout(std::chrono::milliseconds timeout, Callback callback)
      override;
  Handle stream_timeout(
      c10::Stream stream,
      std::chrono::milliseconds timeout,
      Callback callback) override;
  Handle stream_completed(c10::Stream stream, Callback callback) override;
  Handle op_timeout(
      c10::Stream stream,
      std::chrono::milliseconds timeout,
      Callback callback) override;
  size_t numActiveStreamTimeouts() const override {
    return activeCount_.load();
  }

 private:
  void cancel(uint64_t id) override;
  void markCompleted(uint64_t id, const c10::Stream& stream) override;

  // A scheduled one-shot CPU timer.
  struct CpuTimer {
    uv_timer_t handle{};
    WatchdogImpl* impl{nullptr};
    uint64_t id{0};
    Callback onTimeout;
  };

  // A single-event device monitor that fires if the event has not completed by
  // the deadline.
  struct StreamTimeout {
    std::shared_ptr<c10::Event> event;
    std::chrono::steady_clock::time_point deadline;
    Callback onTimeout;
  };

  // A single-event device monitor that fires once the event has completed.
  struct StreamCompleted {
    std::shared_ptr<c10::Event> event;
    Callback onCompleted;
  };

  // A two-phase device-operation monitor. A start event bounds the launch (the
  // op must begin executing within timeout); once it completes, an end event
  // (attached by markCompleted) bounds the completion (timeout from start).
  struct OpTimeout {
    std::shared_ptr<c10::Event> startEvent;
    std::shared_ptr<c10::Event> endEvent; // null until markCompleted
    std::chrono::milliseconds timeout{};
    std::chrono::steady_clock::time_point registeredAt;
    bool started{false};
    std::chrono::steady_clock::time_point startedAt;
    Callback onTimeout;
  };

  // Recycles c10::Event objects (per device) instead of destroying them.
  // Destroying an event (e.g. cudaEventDestroy) can block, so it must never run
  // on the loop thread; releasing an event back here only moves a pointer, and
  // events are destroyed only when the cache itself is, after the loop has been
  // joined.
  class EventCache {
   public:
    std::shared_ptr<c10::Event> acquire(const c10::Stream& stream) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& free = free_[stream.device()];
        if (!free.empty()) {
          auto event = std::move(free.back());
          free.pop_back();
          return event;
        }
      }
      return std::make_shared<c10::Event>(stream.device_type());
    }

    void release(std::shared_ptr<c10::Event> event) {
      if (!event) {
        return;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      free_[event->device()].push_back(std::move(event));
    }

   private:
    std::mutex mutex_;
    std::unordered_map<c10::Device, std::vector<std::shared_ptr<c10::Event>>>
        free_;
  };

  uint64_t nextId() {
    return nextId_.fetch_add(1);
  }
  void enqueue(std::function<void()> request);

  // Loop thread.
  void run();
  void onRequests();
  void onStop();
  void pollMonitors();
  void armCpuTimer(
      uint64_t id,
      std::chrono::milliseconds timeout,
      Callback onTimeout);
  void closeCpuTimer(uint64_t id);
  void eraseStreamTimeout(uint64_t id);
  void eraseStreamCompleted(uint64_t id);
  void eraseOpTimeout(uint64_t id);
  void maybeStartPollTimer();
  void maybeStopPollTimer();
  void updateActiveCount() {
    activeCount_.store(
        streamTimeouts_.size() + streamCompleted_.size() + opTimeouts_.size());
  }

  static WatchdogImpl& fromHandle(uv_handle_t* handle) {
    return *static_cast<WatchdogImpl*>(uv_handle_get_data(handle));
  }

  // libuv loop and the handles that drive it. All uv_* state below is only
  // touched from the loop thread, except requestAsync_/stopAsync_ which are
  // signalled cross-thread via the thread-safe uv_async_send.
  uv_loop_t loop_{};
  uv_async_t requestAsync_{};
  uv_async_t stopAsync_{};
  uv_timer_t pollTimer_{};
  std::chrono::milliseconds pollInterval_;
  bool pollActive_{false};
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};

  std::mutex requestMutex_;
  std::vector<std::function<void()>> requests_;

  // Loop-thread-only state, keyed by monitor id.
  std::unordered_map<uint64_t, std::unique_ptr<CpuTimer>> cpuTimers_;
  std::unordered_map<uint64_t, StreamTimeout> streamTimeouts_;
  std::unordered_map<uint64_t, StreamCompleted> streamCompleted_;
  std::unordered_map<uint64_t, OpTimeout> opTimeouts_;

  std::atomic<uint64_t> nextId_{1};
  std::atomic<size_t> activeCount_{0};

  EventCache eventCache_;
};

WatchdogImpl::WatchdogImpl(std::chrono::milliseconds pollInterval)
    : pollInterval_(pollInterval) {
  TORCH_CHECK(uv_loop_init(&loop_) == 0, "Failed to init watchdog uv loop");

  TORCH_CHECK(
      uv_async_init(
          &loop_,
          &requestAsync_,
          [](uv_async_t* h) {
            fromHandle(reinterpret_cast<uv_handle_t*>(h)).onRequests();
          }) == 0,
      "Failed to init watchdog request handle");
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&requestAsync_), this);

  TORCH_CHECK(
      uv_async_init(
          &loop_,
          &stopAsync_,
          [](uv_async_t* h) {
            fromHandle(reinterpret_cast<uv_handle_t*>(h)).onStop();
          }) == 0,
      "Failed to init watchdog stop handle");
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&stopAsync_), this);

  TORCH_CHECK(
      uv_timer_init(&loop_, &pollTimer_) == 0,
      "Failed to init watchdog poll timer");
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&pollTimer_), this);

  thread_ = std::thread([this] {
    c10::setThreadName("pt_watchdog");
    run();
  });
  running_.store(true);
}

WatchdogImpl::~WatchdogImpl() {
  if (running_.load()) {
    uv_async_send(&stopAsync_);
    thread_.join();
    running_.store(false);
  }
  // eventCache_ (and thus its events) is destroyed with this object, now that
  // the loop thread has been joined.
}

void WatchdogImpl::run() {
  // Blocks servicing timers and async wakeups. onStop closes every handle (and
  // does not call uv_stop), so once their close callbacks have fired there are
  // no active handles left and uv_run returns; the loop can then be closed.
  uv_run(&loop_, UV_RUN_DEFAULT);
  if (uv_loop_close(&loop_) != 0) {
    // Should not happen: every handle is closed before uv_run returns. Warn
    // rather than throw, since this runs on the loop thread during teardown.
    TORCH_WARN("watchdog uv loop close failed; a handle may have leaked");
  }
}

void WatchdogImpl::enqueue(std::function<void()> request) {
  {
    std::lock_guard<std::mutex> lock(requestMutex_);
    requests_.push_back(std::move(request));
  }
  uv_async_send(&requestAsync_);
}

void WatchdogImpl::onRequests() {
  // Once shutdown has started, don't arm new work; pending requests are
  // discarded (and destroyed) by the joining thread.
  if (stopping_.load()) {
    return;
  }
  std::vector<std::function<void()>> requests;
  {
    std::lock_guard<std::mutex> lock(requestMutex_);
    requests.swap(requests_);
  }
  for (auto& request : requests) {
    request();
  }
}

void WatchdogImpl::armCpuTimer(
    uint64_t id,
    std::chrono::milliseconds timeout,
    Callback onTimeout) {
  auto timer = std::make_unique<CpuTimer>();
  timer->impl = this;
  timer->id = id;
  timer->onTimeout = std::move(onTimeout);
  uv_timer_init(&loop_, &timer->handle);
  uv_handle_set_data(
      reinterpret_cast<uv_handle_t*>(&timer->handle), timer.get());

  CpuTimer* raw = timer.get();
  cpuTimers_[id] = std::move(timer);
  uv_timer_start(
      &raw->handle,
      [](uv_timer_t* h) {
        auto* t = static_cast<CpuTimer*>(
            uv_handle_get_data(reinterpret_cast<uv_handle_t*>(h)));
        if (t->onTimeout) {
          t->onTimeout();
        }
        t->impl->closeCpuTimer(t->id);
      },
      static_cast<uint64_t>(timeout.count()),
      /*repeat=*/0);
}

void WatchdogImpl::closeCpuTimer(uint64_t id) {
  auto it = cpuTimers_.find(id);
  if (it == cpuTimers_.end()) {
    return;
  }
  // Transfer ownership to the close callback, which deletes the timer once
  // libuv has finished closing the handle.
  CpuTimer* timer = it->second.release();
  cpuTimers_.erase(it);
  uv_timer_stop(&timer->handle);
  uv_close(reinterpret_cast<uv_handle_t*>(&timer->handle), [](uv_handle_t* h) {
    delete static_cast<CpuTimer*>(uv_handle_get_data(h));
  });
}

void WatchdogImpl::eraseStreamTimeout(uint64_t id) {
  auto it = streamTimeouts_.find(id);
  if (it == streamTimeouts_.end()) {
    return;
  }
  eventCache_.release(std::move(it->second.event));
  streamTimeouts_.erase(it);
  updateActiveCount();
  maybeStopPollTimer();
}

void WatchdogImpl::eraseStreamCompleted(uint64_t id) {
  auto it = streamCompleted_.find(id);
  if (it == streamCompleted_.end()) {
    return;
  }
  eventCache_.release(std::move(it->second.event));
  streamCompleted_.erase(it);
  updateActiveCount();
  maybeStopPollTimer();
}

void WatchdogImpl::eraseOpTimeout(uint64_t id) {
  auto it = opTimeouts_.find(id);
  if (it == opTimeouts_.end()) {
    return;
  }
  eventCache_.release(std::move(it->second.startEvent));
  eventCache_.release(std::move(it->second.endEvent));
  opTimeouts_.erase(it);
  updateActiveCount();
  maybeStopPollTimer();
}

void WatchdogImpl::maybeStartPollTimer() {
  if (pollActive_) {
    return;
  }
  auto intervalMs = static_cast<uint64_t>(pollInterval_.count());
  uv_timer_start(
      &pollTimer_,
      [](uv_timer_t* h) {
        fromHandle(reinterpret_cast<uv_handle_t*>(h)).pollMonitors();
      },
      intervalMs,
      intervalMs);
  pollActive_ = true;
}

void WatchdogImpl::maybeStopPollTimer() {
  if (streamTimeouts_.empty() && streamCompleted_.empty() &&
      opTimeouts_.empty() && pollActive_) {
    uv_timer_stop(&pollTimer_);
    pollActive_ = false;
  }
}

void WatchdogImpl::pollMonitors() {
  auto now = std::chrono::steady_clock::now();

  for (auto it = streamTimeouts_.begin(); it != streamTimeouts_.end();) {
    StreamTimeout& st = it->second;
    bool done = false;
    if (queryCompleted(*st.event)) {
      done = true; // completed in time
    } else if (now >= st.deadline) {
      if (st.onTimeout) {
        st.onTimeout();
      }
      done = true;
    }
    if (done) {
      eventCache_.release(std::move(st.event));
      it = streamTimeouts_.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = streamCompleted_.begin(); it != streamCompleted_.end();) {
    StreamCompleted& sc = it->second;
    if (queryCompleted(*sc.event)) {
      if (sc.onCompleted) {
        sc.onCompleted();
      }
      eventCache_.release(std::move(sc.event));
      it = streamCompleted_.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = opTimeouts_.begin(); it != opTimeouts_.end();) {
    OpTimeout& op = it->second;
    bool done = false;

    if (!op.started) {
      if (queryCompleted(*op.startEvent)) {
        op.started = true;
        op.startedAt = now;
      } else if (now - op.registeredAt > op.timeout) {
        if (op.onTimeout) {
          op.onTimeout();
        }
        done = true;
      }
    }

    if (op.started && !done) {
      if (op.endEvent && queryCompleted(*op.endEvent)) {
        done = true;
      } else if (now - op.startedAt > op.timeout) {
        if (op.onTimeout) {
          op.onTimeout();
        }
        done = true;
      }
    }

    if (done) {
      eventCache_.release(std::move(op.startEvent));
      eventCache_.release(std::move(op.endEvent));
      it = opTimeouts_.erase(it);
    } else {
      ++it;
    }
  }

  updateActiveCount();
  maybeStopPollTimer();
}

void WatchdogImpl::onStop() {
  stopping_.store(true);

  for (auto& [id, timer] : cpuTimers_) {
    CpuTimer* raw = timer.release();
    uv_timer_stop(&raw->handle);
    uv_close(reinterpret_cast<uv_handle_t*>(&raw->handle), [](uv_handle_t* h) {
      delete static_cast<CpuTimer*>(uv_handle_get_data(h));
    });
  }
  cpuTimers_.clear();

  for (auto& [id, st] : streamTimeouts_) {
    eventCache_.release(std::move(st.event));
  }
  streamTimeouts_.clear();

  for (auto& [id, sc] : streamCompleted_) {
    eventCache_.release(std::move(sc.event));
  }
  streamCompleted_.clear();

  for (auto& [id, op] : opTimeouts_) {
    eventCache_.release(std::move(op.startEvent));
    eventCache_.release(std::move(op.endEvent));
  }
  opTimeouts_.clear();
  activeCount_.store(0);

  if (pollActive_) {
    uv_timer_stop(&pollTimer_);
    pollActive_ = false;
  }
  // Close every remaining handle (including this one) but don't uv_stop: once
  // the close callbacks have fired the loop has no active handles and uv_run
  // returns on its own.
  uv_close(reinterpret_cast<uv_handle_t*>(&pollTimer_), nullptr);
  uv_close(reinterpret_cast<uv_handle_t*>(&requestAsync_), nullptr);
  uv_close(reinterpret_cast<uv_handle_t*>(&stopAsync_), nullptr);
}

Handle WatchdogImpl::cpu_timeout(
    std::chrono::milliseconds timeout,
    Callback callback) {
  uint64_t id = nextId();
  enqueue([this, id, timeout, cb = std::move(callback)]() mutable {
    armCpuTimer(id, timeout, std::move(cb));
  });
  return makeHandle(id);
}

Handle WatchdogImpl::stream_timeout(
    c10::Stream stream,
    std::chrono::milliseconds timeout,
    Callback callback) {
  uint64_t id = nextId();
  auto event = eventCache_.acquire(stream);
  event->record(stream);
  auto deadline = std::chrono::steady_clock::now() + timeout;

  enqueue([this, id, event, deadline, cb = std::move(callback)]() mutable {
    streamTimeouts_[id] = StreamTimeout{event, deadline, std::move(cb)};
    updateActiveCount();
    maybeStartPollTimer();
  });
  return makeHandle(id);
}

Handle WatchdogImpl::stream_completed(c10::Stream stream, Callback callback) {
  uint64_t id = nextId();
  auto event = eventCache_.acquire(stream);
  event->record(stream);

  enqueue([this, id, event, cb = std::move(callback)]() mutable {
    streamCompleted_[id] = StreamCompleted{event, std::move(cb)};
    updateActiveCount();
    maybeStartPollTimer();
  });
  return makeHandle(id);
}

Handle WatchdogImpl::op_timeout(
    c10::Stream stream,
    std::chrono::milliseconds timeout,
    Callback callback) {
  uint64_t id = nextId();
  auto startEvent = eventCache_.acquire(stream);
  startEvent->record(stream);
  auto registeredAt = std::chrono::steady_clock::now();

  enqueue([this, id, startEvent, timeout, registeredAt,
           cb = std::move(callback)]() mutable {
    opTimeouts_[id] = OpTimeout{
        startEvent, nullptr, timeout, registeredAt, false, {}, std::move(cb)};
    updateActiveCount();
    maybeStartPollTimer();
  });
  return makeHandle(id, stream);
}

void WatchdogImpl::markCompleted(uint64_t id, const c10::Stream& stream) {
  auto endEvent = eventCache_.acquire(stream);
  endEvent->record(stream);
  enqueue([this, id, endEvent]() mutable {
    auto it = opTimeouts_.find(id);
    if (it != opTimeouts_.end()) {
      it->second.endEvent = endEvent;
    } else {
      // The op already completed or was cancelled.
      eventCache_.release(std::move(endEvent));
    }
  });
}

void WatchdogImpl::cancel(uint64_t id) {
  enqueue([this, id]() {
    closeCpuTimer(id);
    eraseStreamTimeout(id);
    eraseStreamCompleted(id);
    eraseOpTimeout(id);
  });
}

} // namespace

Handle Watchdog::makeHandle(uint64_t id, std::optional<c10::Stream> stream) {
  return Handle(weak_from_this(), id, stream);
}

std::shared_ptr<Watchdog> makeWatchdog(std::chrono::milliseconds pollInterval) {
  return std::make_shared<WatchdogImpl>(pollInterval);
}

const std::shared_ptr<Watchdog>& Watchdog::singleton() {
  // Intentionally leaked: the global watchdog owns a background thread that we
  // never want to join during interpreter/static destruction.
  static auto* instance = new std::shared_ptr<Watchdog>(makeWatchdog());
  return *instance;
}

void Handle::cancel() const {
  if (auto watchdog = watchdog_.lock()) {
    watchdog->cancel(id_);
  }
}

void Handle::completed() const {
  if (!stream_.has_value()) {
    return;
  }
  if (auto watchdog = watchdog_.lock()) {
    watchdog->markCompleted(id_, *stream_);
  }
}

} // namespace c10d::watchdog

#endif // TORCH_USE_LIBUV
