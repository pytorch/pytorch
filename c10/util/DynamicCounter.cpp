#include <c10/util/DynamicCounter.h>
#include <c10/util/Synchronized.h>

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10::monitor {

namespace {
using DynamicCounterBackendFactories =
    std::vector<std::shared_ptr<detail::DynamicCounterBackendFactoryIf>>;

Synchronized<DynamicCounterBackendFactories>& dynamicCounterBackendFactories() {
  static auto instance = new Synchronized<DynamicCounterBackendFactories>();
  return *instance;
}

Synchronized<std::unordered_set<std::string>>& registeredCounters() {
  static auto instance = new Synchronized<std::unordered_set<std::string>>();
  return *instance;
}
} // namespace

namespace detail {
void registerDynamicCounterBackend(
    std::unique_ptr<DynamicCounterBackendFactoryIf> factory) {
  dynamicCounterBackendFactories().withLock(
      [&](auto& factories) { factories.push_back(std::move(factory)); });
}
} // namespace detail

struct DynamicCounter::Guard {
  Guard(std::string_view key, Callback&& getCounterCallback)
      : key_{key}, getCounterCallback_(std::move(getCounterCallback)) {
    // Ensure that the counter with this key is not already registered
    registeredCounters().withLock([&](auto& registeredCounters) {
      if (!registeredCounters.insert(std::string(key)).second) {
        throw std::logic_error(
            "Counter " + std::string(key) + " already registered");
      }
    });

    // Create backends for this dynamic counter using the factory
    dynamicCounterBackendFactories().withLock([&](auto& factories) {
      for (const auto& factory : factories) {
        auto backend = factory->create(key);
        if (backend) {
          // Avoid copying the user-provided callback to avoid unexpected
          // behavior changes when more than one backend is registered.
          backend->registerCounter(
              key_, [&]() { return getCounterCallback_(); });
          backends_.push_back(std::move(backend));
        }
      }
    });
  }

  Guard(Guard&& other) = delete;
  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;
  Guard& operator=(Guard&&) = delete;

  ~Guard() {
    // Unregister the counter from all backends
    for (const auto& backend : backends_) {
      backend->unregisterCounter(key_);
    }

    // Remove the counter from the registered counters set
    registeredCounters().withLock(
        [&](auto& registeredCounters) { registeredCounters.erase(key_); });
  }

 private:
  std::string key_;
  Callback getCounterCallback_;
  std::vector<std::unique_ptr<detail::DynamicCounterBackendIf>>
      backends_; // Store backends created by the factory
};

DynamicCounter::DynamicCounter(
    std::string_view key,
    Callback getCounterCallback)
    : guard_{std::make_unique<Guard>(key, std::move(getCounterCallback))} {}

DynamicCounter::~DynamicCounter() = default;

} // namespace c10::monitor
