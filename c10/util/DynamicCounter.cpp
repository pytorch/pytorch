#include <c10/util/DynamicCounter.h>

#include <c10/util/Synchronized.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace c10::monitor {

namespace {
using DynamicCounterBackends =
    std::vector<std::shared_ptr<detail::DynamicCounterBackendIf>>;

Synchronized<DynamicCounterBackends>& dynamicCounterBackends() {
  static auto instance = new Synchronized<DynamicCounterBackends>();
  return *instance;
}

Synchronized<std::unordered_set<std::string>>& registeredCounters() {
  static auto instance = new Synchronized<std::unordered_set<std::string>>();
  return *instance;
}
} // namespace

namespace detail {
void registerDynamicCounterBackend(
    std::unique_ptr<DynamicCounterBackendIf> backend) {
  dynamicCounterBackends().withLock(
      [&](auto& backends) { backends.push_back(std::move(backend)); });
}
} // namespace detail

struct DynamicCounter::Guard {
  Guard(std::string_view key, Callback getCounterCallback)
      : key_{key},
        backends_{dynamicCounterBackends().withLock(
            [](auto& backends) { return backends; })} {
    registeredCounters().withLock([&](auto& registeredCounters) {
      if (!registeredCounters.insert(std::string(key)).second) {
        throw std::logic_error(
            "Counter " + std::string(key) + " already registered");
      }
    });

    for (const auto& backend : backends_) {
      backend->registerCounter(key, getCounterCallback);
    }
  }

  ~Guard() {
    for (const auto& backend : backends_) {
      backend->unregisterCounter(key_);
    }

    registeredCounters().withLock(
        [&](auto& registeredCounters) { registeredCounters.erase(key_); });
  }

 private:
  std::string key_;
  DynamicCounterBackends backends_;
};

DynamicCounter::DynamicCounter(
    std::string_view key,
    Callback getCounterCallback)
    : guard_{std::make_unique<Guard>(key, getCounterCallback)} {}
DynamicCounter::~DynamicCounter() {}

} // namespace c10::monitor
