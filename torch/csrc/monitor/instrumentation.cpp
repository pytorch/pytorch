#include <torch/csrc/monitor/instrumentation.h>
#include <torch/csrc/monitor/stats.h>

#include <atomic>
#include <utility>
#include <optional>
#include <c10/util/Synchronized.h>
#include <memory>
#include <unordered_map>
#include <thread>
#include <fmt/format.h>
#include <c10/util/Logging.h>

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
      : stats_{key},
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
    stats_.addValue(
        lastPublishUsAndRefCountNew.timestampUs() -
        lastPublishUsAndRefCountSnapshot.timestampUs());
  }

  void start(std::chrono::steady_clock::time_point now) {
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
    stats_.addValue(
        toTimestampUs(now) - lastPublishUsAndRefCountSnapshot.timestampUs());
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

  IntegralStat stats_;
  // wait_start_timestamp * kMaxRefCount_ + (num_waiters - 1). -1 if wait is not
  // active
  std::atomic<LastPublishUsAndRefCount> lastPublishUsAndRefCount_{};
  const std::string key_;
};

namespace {
struct Counters {
  using WaitCounters = c10::Synchronized<
      std::unordered_map<std::string, std::weak_ptr<WaitCounterHandle::State>>>;

  WaitCounters waitCountersMap;
};
// Leaky Meyer's Singleton
static Counters* countersSingleton = new Counters();

// Normal Singleton Implementation
class CountersPublisher {
 public:
  CountersPublisher(CountersPublisher &other) = delete;
  void operator=(const CountersPublisher &) = delete;

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
          countersSingleton->waitCountersMap.withLock([&](auto& rWaitCountersMap){
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

WaitCounterHandle::WaitCounterHandle(std::string_view key)
    : key_(key) {
  {
    countersSingleton->waitCountersMap.withLock([&](auto& wCounters){
      if (wCounters.find(key_) == wCounters.end()){
        state_ = std::make_shared<State>(key_);
        wCounters.emplace(key_, state_);
      } else {
        auto statePtr = wCounters.find(key_)->second.lock();
        CHECK(statePtr) << "State weak_ptr should be removed from the map once expired";
        state_ = statePtr;
      }
    });
  }
}

WaitCounterHandle::~WaitCounterHandle() {
  countersSingleton->waitCountersMap.withLock([&](auto& wCounters){
    std::weak_ptr<State> stateWeak = std::exchange(state_, {});;
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
