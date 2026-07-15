#include <gtest/gtest.h>

#include <torch/csrc/distributed/c10d/watchdog/Watchdog.hpp>

// The watchdog is only built with the libuv backend; without it there is
// nothing to test.
#ifdef TORCH_USE_LIBUV

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

using namespace std::chrono_literals;

namespace {

// Counts callback invocations and lets a test block until a target count is
// reached (or a timeout elapses).
class Latch {
 public:
  void notify() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++count_;
    }
    cv_.notify_all();
  }

  bool waitFor(int target, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return count_ >= target; });
  }

  int count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_{0};
};

} // namespace

TEST(WatchdogTest, singletonNotNull) {
  const auto& watchdog = c10d::watchdog::Watchdog::singleton();
  EXPECT_NE(watchdog, nullptr);
  // The singleton is stable across calls.
  EXPECT_EQ(watchdog, c10d::watchdog::Watchdog::singleton());
}

TEST(WatchdogTest, cpuTimeoutFires) {
  std::shared_ptr<c10d::watchdog::Watchdog> watchdog =
      c10d::watchdog::makeWatchdog();
  auto latch = std::make_shared<Latch>();

  watchdog->cpu_timeout(20ms, [latch] { latch->notify(); });

  EXPECT_TRUE(latch->waitFor(1, 2000ms));
  EXPECT_EQ(latch->count(), 1);
}

TEST(WatchdogTest, cpuTimeoutCancelPreventsFire) {
  std::shared_ptr<c10d::watchdog::Watchdog> watchdog =
      c10d::watchdog::makeWatchdog();
  auto latch = std::make_shared<Latch>();

  auto handle = watchdog->cpu_timeout(200ms, [latch] { latch->notify(); });
  handle.cancel();

  // Wait well past the timeout; the callback must not have fired.
  EXPECT_FALSE(latch->waitFor(1, 500ms));
  EXPECT_EQ(latch->count(), 0);
}

TEST(WatchdogTest, cancelDefaultHandleIsNoop) {
  // A default-constructed handle has nothing to cancel.
  c10d::watchdog::Handle handle;
  handle.cancel();
}

TEST(WatchdogTest, multipleCpuTimeoutsFire) {
  std::shared_ptr<c10d::watchdog::Watchdog> watchdog =
      c10d::watchdog::makeWatchdog();
  auto latch = std::make_shared<Latch>();

  constexpr int kNumTimers = 8;
  for (int i = 0; i < kNumTimers; ++i) {
    watchdog->cpu_timeout(20ms, [latch] { latch->notify(); });
  }

  EXPECT_TRUE(latch->waitFor(kNumTimers, 2000ms));
  EXPECT_EQ(latch->count(), kNumTimers);
}

TEST(WatchdogTest, destroyWithPendingTimeoutDoesNotFire) {
  // A timeout that outlives its watchdog must be cancelled cleanly on shutdown
  // and must not invoke its callback.
  auto latch = std::make_shared<Latch>();
  {
    std::shared_ptr<c10d::watchdog::Watchdog> watchdog =
        c10d::watchdog::makeWatchdog();
    watchdog->cpu_timeout(60s, [latch] { latch->notify(); });
    // Give the loop a moment to actually arm the timer before teardown.
    std::this_thread::sleep_for(50ms);
  }
  EXPECT_EQ(latch->count(), 0);
}

TEST(WatchdogTest, noActiveStreamTimeouts) {
  std::shared_ptr<c10d::watchdog::Watchdog> watchdog =
      c10d::watchdog::makeWatchdog();
  EXPECT_EQ(watchdog->numActiveStreamTimeouts(), 0u);
}

#endif // TORCH_USE_LIBUV
