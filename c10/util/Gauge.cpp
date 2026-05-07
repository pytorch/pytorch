#include <c10/util/Gauge.h>

#include <c10/util/Synchronized.h>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace c10::monitor {

namespace detail {
namespace {
using GaugeBackendFactories =
    std::vector<std::shared_ptr<GaugeBackendFactoryIf>>;

Synchronized<GaugeBackendFactories>& gaugeBackendFactories() {
  static auto instance = new Synchronized<GaugeBackendFactories>();
  return *instance;
}
} // namespace

class GaugeImpl {
 public:
  static GaugeImpl& getInstance(std::string_view key) {
    static auto& implMapSynchronized = *new Synchronized<
        std::unordered_map<std::string, std::unique_ptr<GaugeImpl>>>();

    return *implMapSynchronized.withLock([&](auto& implMap) {
      if (auto implIt = implMap.find(std::string(key));
          implIt != implMap.end()) {
        return implIt->second.get();
      }

      auto [implIt, emplaceSuccess] = implMap.emplace(
          std::string{key}, std::unique_ptr<GaugeImpl>(new GaugeImpl(key)));

      assert(emplaceSuccess);

      return implIt->second.get();
    });
  }

  void record(int64_t value) {
    for (auto& backend : backends_) {
      backend->record(value);
    }
  }

 private:
  explicit GaugeImpl(std::string_view key) {
    auto factoriesCopy = gaugeBackendFactories().withLock(
        [](auto& factories) { return factories; });
    for (const auto& factory : factoriesCopy) {
      if (auto backend = factory->create(key)) {
        backends_.push_back(std::move(backend));
      }
    }
  }

  SmallVector<std::unique_ptr<GaugeBackendIf>> backends_;
};

void registerGaugeBackend(std::unique_ptr<GaugeBackendFactoryIf> backend) {
  gaugeBackendFactories().withLock(
      [&](auto& backends) { backends.push_back(std::move(backend)); });
}

} // namespace detail

GaugeHandle::GaugeHandle(std::string_view key)
    : impl_(detail::GaugeImpl::getInstance(key)) {}

void GaugeHandle::record(int64_t value) {
  impl_.record(value);
}

} // namespace c10::monitor
