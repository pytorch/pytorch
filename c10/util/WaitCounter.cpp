#include <c10/util/WaitCounter.h>

#include <c10/util/Synchronized.h>

#include <chrono>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace c10::monitor {

namespace detail {
namespace {
using WaitCounterBackendFactories =
    std::vector<std::shared_ptr<WaitCounterBackendFactoryIf>>;

Synchronized<WaitCounterBackendFactories>& waitCounterBackendFactories() {
  static auto instance = new Synchronized<WaitCounterBackendFactories>();
  return *instance;
}
} // namespace

class WaitCounterImpl {
 public:
  static WaitCounterImpl& getInstance(std::string_view key) {
    static auto& implMapSynchronized = *new Synchronized<
        std::unordered_map<std::string, std::unique_ptr<WaitCounterImpl>>>();

    return *implMapSynchronized.withLock([&](auto& implMap) {
      if (auto implIt = implMap.find(std::string(key));
          implIt != implMap.end()) {
        return implIt->second.get();
      }

      auto [implIt, emplaceSuccess] = implMap.emplace(
          std::string{key},
          std::unique_ptr<WaitCounterImpl>(new WaitCounterImpl(key)));

      assert(emplaceSuccess);

      return implIt->second.get();
    });
  }

  SmallVector<intptr_t> start() noexcept {
    auto now = std::chrono::steady_clock::now();
    SmallVector<intptr_t> ctxs;
    ctxs.reserve(backends_.size());
    for (const auto& backend : backends_) {
      ctxs.push_back(backend->start(now));
    }
    return ctxs;
  }

  void stop(SmallVector<intptr_t>&& ctxs) noexcept {
    auto now = std::chrono::steady_clock::now();
    assert(ctxs.size() == backends_.size());
    for (size_t i = 0; i < ctxs.size(); ++i) {
      backends_[i]->stop(now, ctxs[i]);
    }
  }

 private:
  explicit WaitCounterImpl(std::string_view key) {
    auto factoriesCopy = waitCounterBackendFactories().withLock(
        [](auto& factories) { return factories; });
    for (const auto& factory : factoriesCopy) {
      if (auto backend = factory->create(key)) {
        backends_.push_back(std::move(backend));
      }
    }
  }

  SmallVector<std::unique_ptr<WaitCounterBackendIf>> backends_;
};

void registerWaitCounterBackend(
    std::unique_ptr<WaitCounterBackendFactoryIf> factory) {
  waitCounterBackendFactories().withLock(
      [&](auto& factories) { factories.push_back(std::move(factory)); });
}
} // namespace detail

WaitCounterHandle::WaitCounterHandle(std::string_view key)
    : impl_(detail::WaitCounterImpl::getInstance(key)) {}

WaitCounterHandle::WaitGuard WaitCounterHandle::start() {
  return WaitCounterHandle::WaitGuard(*this, impl_.start());
}

void WaitCounterHandle::stop(SmallVector<intptr_t>&& ctxs) {
  return impl_.stop(std::move(ctxs));
}
} // namespace c10::monitor
