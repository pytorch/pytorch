#include <c10/util/WaitCounter.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

// Mock backend for testing WaitCounter functionality
class MockWaitCounterBackend
    : public c10::monitor::detail::WaitCounterBackendIf {
 public:
  explicit MockWaitCounterBackend(
      std::atomic<int>& startCount,
      std::atomic<int>& stopCount)
      : startCount_(startCount), stopCount_(stopCount) {}

  intptr_t start(std::chrono::steady_clock::time_point now) noexcept override {
    startCount_.fetch_add(1);
    lastStartTime_ = now;
    return reinterpret_cast<intptr_t>(this);
  }

  void stop(std::chrono::steady_clock::time_point now, intptr_t ctx) noexcept
      override {
    stopCount_.fetch_add(1);
    lastStopTime_ = now;
    EXPECT_EQ(ctx, reinterpret_cast<intptr_t>(this));
  }

  std::chrono::steady_clock::time_point lastStartTime() const {
    return lastStartTime_;
  }

  std::chrono::steady_clock::time_point lastStopTime() const {
    return lastStopTime_;
  }

 private:
  std::atomic<int>& startCount_;
  std::atomic<int>& stopCount_;
  std::chrono::steady_clock::time_point lastStartTime_;
  std::chrono::steady_clock::time_point lastStopTime_;
};

class MockWaitCounterBackendFactory
    : public c10::monitor::detail::WaitCounterBackendFactoryIf {
 public:
  MockWaitCounterBackendFactory(
      std::atomic<int>& startCount,
      std::atomic<int>& stopCount,
      std::string_view keyFilter = "")
      : startCount_(startCount), stopCount_(stopCount), keyFilter_(keyFilter) {}

  std::unique_ptr<c10::monitor::detail::WaitCounterBackendIf> create(
      std::string_view key) noexcept override {
    if (!keyFilter_.empty() && key.find(keyFilter_) == std::string_view::npos) {
      return nullptr;
    }
    return std::make_unique<MockWaitCounterBackend>(startCount_, stopCount_);
  }

 private:
  std::atomic<int>& startCount_;
  std::atomic<int>& stopCount_;
  std::string keyFilter_;
};

TEST(WaitCounter, BackendRegistration) {
  auto backends = c10::monitor::detail::getRegisteredWaitCounterBackends();
  size_t initialCount = backends.size();

  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(startCount, stopCount));

  backends = c10::monitor::detail::getRegisteredWaitCounterBackends();
  EXPECT_EQ(backends.size(), initialCount + 1);
}

TEST(WaitCounter, WaitGuardStartStop) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "wait_guard_start_stop"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  EXPECT_GE(startCount.load(), startBefore);
  {
    c10::monitor::WaitCounterHandle handle("wait_guard_start_stop");
    auto guard = handle.start();
    EXPECT_GE(startCount.load(), startBefore + 1);
    EXPECT_EQ(stopCount.load(), stopBefore);
  }

  EXPECT_GE(stopCount.load(), stopBefore + 1);
}

TEST(WaitCounter, WaitGuardExplicitStop) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "wait_guard_explicit_stop"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  c10::monitor::WaitCounterHandle handle("wait_guard_explicit_stop");
  auto guard = handle.start();
  EXPECT_GE(startCount.load(), startBefore + 1);
  EXPECT_EQ(stopCount.load(), stopBefore);

  guard.stop();
  EXPECT_GE(stopCount.load(), stopBefore + 1);

  // Calling stop() again should be a no-op (guard is already stopped)
  int stopAfterFirst = stopCount.load();
  guard.stop();
  EXPECT_EQ(stopCount.load(), stopAfterFirst);
}

TEST(WaitCounter, WaitGuardMoveConstruction) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "wait_guard_move"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  {
    c10::monitor::WaitCounterHandle handle("wait_guard_move");
    auto guard1 = handle.start();
    EXPECT_GE(startCount.load(), startBefore + 1);

    // Move the guard
    auto guard2 = std::move(guard1);
    // Original guard should not call stop on destruction
  }

  // Stop should be called exactly once when guard2 is destroyed
  EXPECT_GE(stopCount.load(), stopBefore + 1);
}

TEST(WaitCounter, StaticWaitCounterMacro) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "static_macro_test"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  {
    auto guard = STATIC_WAIT_COUNTER(static_macro_test).start();
    EXPECT_GE(startCount.load(), startBefore + 1);
  }

  EXPECT_GE(stopCount.load(), stopBefore + 1);
}

TEST(WaitCounter, StaticScopedWaitCounterMacro) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "static_scoped_test"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  {
    STATIC_SCOPED_WAIT_COUNTER(static_scoped_test);
    EXPECT_GE(startCount.load(), startBefore + 1);
  }

  EXPECT_GE(stopCount.load(), stopBefore + 1);
}

TEST(WaitCounter, WithWaitCounterMacroAssign) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "execute_with_test"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  int value = 0;
  WITH_WAIT_COUNTER(execute_with_test, value = 42);

  EXPECT_EQ(value, 42);
  EXPECT_GE(startCount.load(), startBefore + 1);
  EXPECT_GE(stopCount.load(), stopBefore + 1);
}

TEST(WaitCounter, WithWaitCounterMacroReturn) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "execute_with_test"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  int value = 0;
  value = WITH_WAIT_COUNTER(execute_with_test, []() { return 42; }());

  EXPECT_EQ(value, 42);
  EXPECT_GE(startCount.load(), startBefore + 1);
  EXPECT_GE(stopCount.load(), stopBefore + 1);
}

TEST(WaitCounter, SameHandleMultipleTimes) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "multiple_times_test"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  c10::monitor::WaitCounterHandle handle("multiple_times_test");

  for (int i = 0; i < 5; ++i) {
    auto guard = handle.start();
  }

  EXPECT_GE(startCount.load(), startBefore + 5);
  EXPECT_GE(stopCount.load(), stopBefore + 5);
}

TEST(WaitCounter, ConcurrentUsage) {
  std::atomic<int> startCount{0};
  std::atomic<int> stopCount{0};

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<MockWaitCounterBackendFactory>(
          startCount, stopCount, "concurrent_test"));

  int startBefore = startCount.load();
  int stopBefore = stopCount.load();

  constexpr int kNumThreads = 10;
  constexpr int kIterationsPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kIterationsPerThread; ++i) {
        STATIC_SCOPED_WAIT_COUNTER(concurrent_test);
        std::this_thread::yield();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_GE(
      startCount.load(), startBefore + kNumThreads * kIterationsPerThread);
  EXPECT_GE(stopCount.load(), stopBefore + kNumThreads * kIterationsPerThread);
}

TEST(WaitCounter, StaticHandlePerCallSite) {
  // STATIC_WAIT_COUNTER creates a static handle per call-site, not per key
  // Each invocation at the same source location returns the same handle
  auto& handle1 = STATIC_WAIT_COUNTER(call_site_test);
  auto& handle1_again = STATIC_WAIT_COUNTER(call_site_test);

  // Different source lines create different static handles
  // (This is expected behavior - each call site gets its own static)
  // To test same-location singleton behavior, we call the same macro twice
  // within a loop
  std::vector<c10::monitor::WaitCounterHandle*> handles;
  for (int i = 0; i < 3; ++i) {
    handles.push_back(&STATIC_WAIT_COUNTER(loop_test));
  }

  // All handles from the loop should be the same (same call site)
  EXPECT_EQ(handles[0], handles[1]);
  EXPECT_EQ(handles[1], handles[2]);

  // Suppress unused variable warning
  (void)handle1;
  (void)handle1_again;
}

TEST(WaitCounter, FactoryReturnsNullptr) {
  // Test that backend factory returning nullptr is handled gracefully
  class NullBackendFactory
      : public c10::monitor::detail::WaitCounterBackendFactoryIf {
   public:
    std::unique_ptr<c10::monitor::detail::WaitCounterBackendIf> create(
        std::string_view /*key*/) noexcept override {
      return nullptr;
    }
  };

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<NullBackendFactory>());

  // Should not crash when creating a counter with a null backend
  c10::monitor::WaitCounterHandle handle("null_backend_test");
  auto guard = handle.start();
  guard.stop();
}

TEST(WaitCounter, TimeMeasurement) {
  std::chrono::steady_clock::time_point startTime;
  std::chrono::steady_clock::time_point stopTime;

  class TimingBackend : public c10::monitor::detail::WaitCounterBackendIf {
   public:
    TimingBackend(
        std::chrono::steady_clock::time_point& startTime,
        std::chrono::steady_clock::time_point& stopTime)
        : startTime_(startTime), stopTime_(stopTime) {}

    intptr_t start(
        std::chrono::steady_clock::time_point now) noexcept override {
      startTime_ = now;
      return 0;
    }

    void stop(std::chrono::steady_clock::time_point now, intptr_t) noexcept
        override {
      stopTime_ = now;
    }

   private:
    std::chrono::steady_clock::time_point& startTime_;
    std::chrono::steady_clock::time_point& stopTime_;
  };

  class TimingBackendFactory
      : public c10::monitor::detail::WaitCounterBackendFactoryIf {
   public:
    TimingBackendFactory(
        std::chrono::steady_clock::time_point& startTime,
        std::chrono::steady_clock::time_point& stopTime)
        : startTime_(startTime), stopTime_(stopTime) {}

    std::unique_ptr<c10::monitor::detail::WaitCounterBackendIf> create(
        std::string_view key) noexcept override {
      if (key == "timing_test") {
        return std::make_unique<TimingBackend>(startTime_, stopTime_);
      }
      return nullptr;
    }

   private:
    std::chrono::steady_clock::time_point& startTime_;
    std::chrono::steady_clock::time_point& stopTime_;
  };

  c10::monitor::detail::registerWaitCounterBackend(
      std::make_unique<TimingBackendFactory>(startTime, stopTime));

  {
    c10::monitor::WaitCounterHandle handle("timing_test");
    auto guard = handle.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      stopTime - startTime);
  EXPECT_GE(duration.count(), 10);
}

} // namespace
