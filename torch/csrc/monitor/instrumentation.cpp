#include <torch/csrc/monitor/instrumentation.h>
#include <torch/csrc/monitor/stats.h>

#include <c10/util/Logging.h>
#include <c10/util/Synchronized.h>
#include <fmt/format.h>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace torch {
namespace monitor {

namespace {
ssize_t toTimestampUs(std::chrono::steady_clock::time_point now) {
  static auto start{std::chrono::steady_clock::now()};
  return std::chrono::duration_cast<std::chrono::microseconds>(now - start)
      .count();
}
} // namespace

struct WaitCounterHandle::State {
  explicit State(const std::string& key)
      : waitStat_{key},
        waiters_{
            key + "_active_waiters",
            [&]() -> ssize_t {
              auto lastPublishUsAndRefCountSnapshot =
                  lastPublishUsAndRefCount_.load(std::memory_order_relaxed);
              if (!lastPublishUsAndRefCountSnapshot.refCount()) {
                return 0;
              }
              return lastPublishUsAndRefCountSnapshot.refCount();
            }},
        key_{key} {}

  ~State() {
    DCHECK(
        !lastPublishUsAndRefCount_.load(std::memory_order_relaxed).refCount())
        << "WaitCounterHandle must not be destoyed while waiting.";
  }

  void flush(std::chrono::steady_clock::time_point now) {
    auto lastPublishUsAndRefCountSnapshot =
        lastPublishUsAndRefCount_.load(std::memory_order_relaxed);
    LastPublishUsAndRefCount lastPublishUsAndRefCountNew{};
    do {
      if (lastPublishUsAndRefCountSnapshot.refCount() == 0) {
        return;
      }

      lastPublishUsAndRefCountNew = LastPublishUsAndRefCount(
          now, lastPublishUsAndRefCountSnapshot.refCount());
    } while (!lastPublishUsAndRefCount_.compare_exchange_weak(
        lastPublishUsAndRefCountSnapshot,
        lastPublishUsAndRefCountNew,
        std::memory_order_relaxed));

    DCHECK(
        lastPublishUsAndRefCountNew.refCount() ==
        lastPublishUsAndRefCountSnapshot.refCount());
    waitStat_.addValue(double(
        lastPublishUsAndRefCountNew.timestampUs() -
        lastPublishUsAndRefCountSnapshot.timestampUs()));
  }

  void start(std::chrono::steady_clock::time_point now) {
    waiters_.flushIntegral(now);
    auto lastPublishUsAndRefCountSnapshot =
        lastPublishUsAndRefCount_.load(std::memory_order_relaxed);
    LastPublishUsAndRefCount lastPublishUsAndRefCountNew{};
    do {
      if (!lastPublishUsAndRefCountSnapshot.refCount()) {
        lastPublishUsAndRefCountNew = LastPublishUsAndRefCount(now, 1);
      } else {
        CHECK(
            lastPublishUsAndRefCountSnapshot.refCount() <
            LastPublishUsAndRefCount::kMaxRefCount)
            << fmt::format(
                   "More than kMaxRefCount waiters are not supported. Key: {}, refCount: {}",
                   key_,
                   lastPublishUsAndRefCountSnapshot.refCount());
        lastPublishUsAndRefCountNew = LastPublishUsAndRefCount(
            lastPublishUsAndRefCountSnapshot.timestampUs(),
            lastPublishUsAndRefCountSnapshot.refCount() + 1);
      }
    } while (!lastPublishUsAndRefCount_.compare_exchange_weak(
        lastPublishUsAndRefCountSnapshot,
        lastPublishUsAndRefCountNew,
        std::memory_order_relaxed));
  }

  void stop(std::chrono::steady_clock::time_point now) {
    waiters_.flushIntegral(now);
    auto lastPublishUsAndRefCountSnapshot =
        lastPublishUsAndRefCount_.load(std::memory_order_relaxed);
    LastPublishUsAndRefCount lastPublishUsAndRefCountNew{};
    do {
      DCHECK(lastPublishUsAndRefCountSnapshot.refCount())
          << "stop() called more times than start()";

      if (lastPublishUsAndRefCountSnapshot.refCount() == 1) {
        lastPublishUsAndRefCountNew = {};
      } else {
        lastPublishUsAndRefCountNew = LastPublishUsAndRefCount(
            lastPublishUsAndRefCountSnapshot.timestampUs(),
            lastPublishUsAndRefCountSnapshot.refCount() - 1);
      }
    } while (!lastPublishUsAndRefCount_.compare_exchange_weak(
        lastPublishUsAndRefCountSnapshot,
        lastPublishUsAndRefCountNew,
        std::memory_order_relaxed));
    if (lastPublishUsAndRefCountNew.refCount()) {
      return;
    }
    waitStat_.addValue(double(
        toTimestampUs(now) - lastPublishUsAndRefCountSnapshot.timestampUs()));
  }

 private:
  class LastPublishUsAndRefCount {
   public:
    // kRefCountBits must be <= 14 for storage_ to fit in int64_t
    static constexpr ssize_t kRefCountBits = 14;
    static constexpr ssize_t kMaxRefCount = 1 << kRefCountBits;

    LastPublishUsAndRefCount() = default;
    LastPublishUsAndRefCount(ssize_t timestampUs, ssize_t refCount)
        : storage_{timestampUs * kMaxRefCount + refCount - 1} {
      DCHECK(refCount > 0);
    }
    LastPublishUsAndRefCount(
        std::chrono::steady_clock::time_point time,
        ssize_t refCount)
        : LastPublishUsAndRefCount{toTimestampUs(time), refCount} {}

    ssize_t timestampUs() const {
      return storage_ / kMaxRefCount;
    }

    ssize_t refCount() const {
      return storage_ % kMaxRefCount + 1;
    }

   private:
    // wait_start_timestamp * kMaxRefCount_ + (num_waiters - 1). -1 if wait is
    // not active
    ssize_t storage_{-1};
  };

  PeriodicSumStat waitStat_;
  // wait_start_timestamp * kMaxRefCount_ + (num_waiters - 1). -1 if wait is not
  // active
  std::atomic<LastPublishUsAndRefCount> lastPublishUsAndRefCount_{};
  DynamicCounterHandle waiters_;
  const std::string key_;
};

struct DynamicCounterHandle::State {
 public:
  State(std::string_view key, std::function<int64_t()> callback)
      : periodicStats_(key),
        integralStats_(std::string(key) + "_integral_us"),
        callback_(std::move(callback)) {}

  int64_t getValue() const {
    return callback_();
  }

  void flushIntegral(
      std::optional<int64_t> value,
      std::chrono::steady_clock::time_point now) {
    auto nowUs = toTimestampUs(now);
    auto lastPublishUs =
        lastPublishUs_.exchange(nowUs, std::memory_order_relaxed);
    if (!value) {
      value = getValue();
    }
    integralStats_.addValue(double(*value * (nowUs - lastPublishUs)));
  }

  void publishPeriodic(std::chrono::steady_clock::time_point now) {
    auto value = getValue();
    flushIntegral(value, now);
    periodicStats_.addValue(double(value), now);
  }

 private:
  std::atomic<ssize_t> lastPublishUs_{
      toTimestampUs(std::chrono::steady_clock::now())};
  PeriodicAvgStat periodicStats_;
  PeriodicSumStat integralStats_;
  std::function<int64_t()> callback_;
};

namespace {
struct Counters {
  using DynamicCounters = c10::Synchronized<std::unordered_map<
      std::string,
      std::unique_ptr<DynamicCounterHandle::State>>>;
  using WaitCounters = c10::Synchronized<
      std::unordered_map<std::string, std::weak_ptr<WaitCounterHandle::State>>>;

  DynamicCounters dynamicCountersMap;
  WaitCounters waitCountersMap;
};
// Leaky Meyer's Singleton
static Counters* countersSingleton = new Counters();

// Normal Singleton Implementation
class CountersPublisher {
 public:
  CountersPublisher(CountersPublisher& other) = delete;
  void operator=(const CountersPublisher&) = delete;

  static CountersPublisher& getInstance() {
    // If the instance doesn't exist, create it.
    if (!instance_) {
      terminated_ = false;
      instance_ = new CountersPublisher();
    }
    return *instance_;
  }

 private:
  CountersPublisher() {
    thread_ = std::thread([&]() {
      constexpr auto kCollectionPeriod = std::chrono::milliseconds{10};
      while (!terminated_) {
        auto now = std::chrono::steady_clock::now();
        {
          countersSingleton->dynamicCountersMap.withLock(
              [&](auto& rDynamicCountersMap) {
                for (const auto& keyToStats : rDynamicCountersMap) {
                  keyToStats.second->publishPeriodic(now);
                }
              });
        }
        {
          countersSingleton->waitCountersMap.withLock([&](auto&
                                                              rWaitCountersMap) {
            for (const auto& [_, statePtrWeak] : rWaitCountersMap) {
              auto statePtr = statePtrWeak.lock();
              CHECK(statePtr)
                  << "State weak_ptr should be removed from the map once expired";
              statePtr->flush(now);
            }
          });
        }
        std::this_thread::sleep_for(kCollectionPeriod);
      }
    });
  }

  ~CountersPublisher() {
    terminated_ = true;
    thread_.join();
  }

  static CountersPublisher* instance_;
  static bool terminated_;
  std::thread thread_ = std::thread();
};

CountersPublisher* CountersPublisher::instance_ = nullptr;
bool CountersPublisher::terminated_ = false;

auto& publisherSingleton = CountersPublisher::getInstance();
} // namespace

DynamicCounterHandle::DynamicCounterHandle(
    std::string_view key,
    std::function<int64_t()> callback)
    : key_(key) {
  countersSingleton->dynamicCountersMap.withLock([&](auto& wCounters) {
    statePtr_ = wCounters.emplace(key_, std::make_unique<State>(key, callback))
                    .first->second.get();
  });
  registerCallback(
      key_, [statePtr = statePtr_]() { return statePtr->getValue(); });
}

DynamicCounterHandle::DynamicCounterHandle(DynamicCounterHandle&& src) noexcept
    : key_{src.key_}, statePtr_{std::exchange(src.statePtr_, nullptr)} {}

DynamicCounterHandle::~DynamicCounterHandle() {
  if (statePtr_ == nullptr) {
    return;
  }
  unregisterCallback(key_);
  countersSingleton->dynamicCountersMap.withLock(
      [&](auto& wCounters) { wCounters.erase(key_); });
}

void DynamicCounterHandle::flushIntegral(
    std::chrono::steady_clock::time_point now) const {
  statePtr_->flushIntegral({}, now);
}

WaitCounterHandle::WaitCounterHandle(std::string_view key) : key_(key) {
  {
    countersSingleton->waitCountersMap.withLock(
        [&](auto& wCounters) {
          if (wCounters.find(key_) == wCounters.end()) {
            state_ = std::make_shared<State>(key_);
            wCounters.emplace(key_, state_);
          } else {
            auto statePtr = wCounters.find(key_)->second.lock();
            CHECK(statePtr)
                << "State weak_ptr should be removed from the map once expired";
            state_ = statePtr;
          }
        });
  }
}

WaitCounterHandle::~WaitCounterHandle() {
  countersSingleton->waitCountersMap.withLock([&](auto& wCounters) {
    std::weak_ptr<State> stateWeak = std::exchange(state_, {});
    if (stateWeak.expired()) {
      wCounters.erase(key_);
    }
  });
}

void WaitCounterHandle::start(std::chrono::steady_clock::time_point now) {
  state_->start(now);
}

void WaitCounterHandle::stop(std::chrono::steady_clock::time_point now) {
  state_->stop(now);
}

} // namespace monitor
} // namespace torch
