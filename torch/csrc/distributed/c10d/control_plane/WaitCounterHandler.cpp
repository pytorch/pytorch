#include <torch/csrc/distributed/c10d/control_plane/WaitCounterHandler.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

#include <c10/util/CallOnce.h>
#include <c10/util/Synchronized.h>
#include <c10/util/WaitCounter.h>

#include <nlohmann/json.hpp>

namespace c10d::control_plane {

namespace {

// Data structure to hold counter metrics
struct CounterData {
  std::atomic<int64_t> active_count{0};
  std::atomic<int64_t> total_calls{0};
  std::atomic<int64_t> total_time_us{0};
  std::atomic<int64_t> max_time_us{0};
};

// Holder struct for the counter data map
struct CounterDataMapHolder {
  c10::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<CounterData>>>
      map;
};

// Leaky singleton to avoid static destruction order issues
CounterDataMapHolder* getCounterDataMapHolder() {
  static CounterDataMapHolder* holder = new CounterDataMapHolder();
  return holder;
}

// Backend implementation that tracks counter metrics
class TrackingBackend : public c10::monitor::detail::WaitCounterBackendIf {
 public:
  explicit TrackingBackend(std::string key) : key_(std::move(key)) {
    // Get or create counter data for this key
    getCounterDataMapHolder()->map.withLock([&](auto& map) {
      auto it = map.find(key_);
      if (it == map.end()) {
        data_ = std::make_shared<CounterData>();
        map[key_] = data_;
      } else {
        data_ = it->second;
      }
    });
  }

  intptr_t start(std::chrono::steady_clock::time_point now) noexcept override {
    data_->active_count.fetch_add(1, std::memory_order_relaxed);
    data_->total_calls.fetch_add(1, std::memory_order_relaxed);
    // Return the start time as the context
    return static_cast<intptr_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch())
            .count());
  }

  void stop(std::chrono::steady_clock::time_point now, intptr_t ctx) noexcept
      override {
    // Calculate duration from the stored start time
    auto start_ns = std::chrono::nanoseconds(ctx);
    auto start_time = std::chrono::steady_clock::time_point(start_ns);
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(now - start_time)
            .count();

    data_->active_count.fetch_sub(1, std::memory_order_relaxed);
    data_->total_time_us.fetch_add(duration_us, std::memory_order_relaxed);

    // Update max_time_us using compare-and-swap
    int64_t current_max = data_->max_time_us.load(std::memory_order_relaxed);
    while (duration_us > current_max) {
      if (data_->max_time_us.compare_exchange_weak(
              current_max, duration_us, std::memory_order_relaxed)) {
        break;
      }
    }
  }

 private:
  std::string key_;
  std::shared_ptr<CounterData> data_;
};

// Factory for creating tracking backends
class TrackingBackendFactory
    : public c10::monitor::detail::WaitCounterBackendFactoryIf {
 public:
  std::unique_ptr<c10::monitor::detail::WaitCounterBackendIf> create(
      std::string_view key) noexcept override {
    return std::make_unique<TrackingBackend>(std::string(key));
  }
};

} // namespace

// Ensures the wait counter backend is registered
// NOTE: This function is in the c10d::control_plane namespace, NOT anonymous
void ensureWaitCounterBackendRegistered() {
  static c10::once_flag once;
  c10::call_once(once, []() {
    c10::monitor::detail::registerWaitCounterBackend(
        std::make_unique<TrackingBackendFactory>());
  });
}

// Returns all wait counter values as a JSON string
// NOTE: This function is in the c10d::control_plane namespace, NOT anonymous
std::string getWaitCounterValuesJson() {
  nlohmann::json j = nlohmann::json::object();

  getCounterDataMapHolder()->map.withLock([&](const auto& map) {
    for (const auto& [name, data] : map) {
      nlohmann::json counter_obj = nlohmann::json::object();
      counter_obj["active_count"] =
          data->active_count.load(std::memory_order_relaxed);
      counter_obj["total_calls"] =
          data->total_calls.load(std::memory_order_relaxed);
      counter_obj["total_time_us"] =
          data->total_time_us.load(std::memory_order_relaxed);
      counter_obj["max_time_us"] =
          data->max_time_us.load(std::memory_order_relaxed);
      j[name] = std::move(counter_obj);
    }
  });

  return j.dump();
}

} // namespace c10d::control_plane
