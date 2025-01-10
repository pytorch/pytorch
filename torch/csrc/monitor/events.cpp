#include <torch/csrc/monitor/events.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace torch::monitor {

namespace {
class EventHandlers {
 public:
  void registerEventHandler(std::shared_ptr<EventHandler> handler) noexcept {
    std::unique_lock<std::mutex> lock(mu_);

    handlers_.emplace_back(std::move(handler));
  }

  void unregisterEventHandler(
      const std::shared_ptr<EventHandler>& handler) noexcept {
    std::unique_lock<std::mutex> lock(mu_);

    auto it = std::find(handlers_.begin(), handlers_.end(), handler);
    handlers_.erase(it);
  }

  void logEvent(const Event& e) {
    std::unique_lock<std::mutex> lock(mu_);

    for (auto& handler : handlers_) {
      handler->handle(e);
    }
  }

  static EventHandlers& get() noexcept {
    static auto ehsPtr = new EventHandlers();
    return *ehsPtr;
  }

 private:
  std::mutex mu_{};
  std::vector<std::shared_ptr<EventHandler>> handlers_{};
};
} // namespace

void logEvent(const Event& e) {
  EventHandlers::get().logEvent(e);
}

void registerEventHandler(std::shared_ptr<EventHandler> p) {
  EventHandlers::get().registerEventHandler(std::move(p));
}

void unregisterEventHandler(const std::shared_ptr<EventHandler>& p) {
  EventHandlers::get().unregisterEventHandler(p);
}

} // namespace torch::monitor
