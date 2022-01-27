#include <torch/csrc/monitor/events.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <ATen/Functions.h>

namespace torch {
namespace monitor {

namespace {
bool eventEqual(const data_value_t& lhs, const data_value_t& rhs) {
  // check if same type
  if (lhs.index() != rhs.index()) {
    return false;
  }
  if (c10::holds_alternative<double>(lhs)) {
    return c10::get<double>(lhs) == c10::get<double>(rhs);
  } else if (c10::holds_alternative<int64_t>(lhs)) {
    return c10::get<int64_t>(lhs) == c10::get<int64_t>(rhs);
  } else if (c10::holds_alternative<bool>(lhs)) {
    return c10::get<bool>(lhs) == c10::get<bool>(rhs);
  } else if (c10::holds_alternative<std::string>(lhs)) {
    return c10::get<std::string>(lhs) == c10::get<std::string>(rhs);
  } else if (c10::holds_alternative<at::Tensor>(lhs)) {
    return at::equal(c10::get<at::Tensor>(lhs), c10::get<at::Tensor>(rhs));
  } else {
    throw std::runtime_error("unknown data_value_t type");
  }
}
bool eventEqual(
    const std::unordered_map<std::string, data_value_t>& lhs,
    const std::unordered_map<std::string, data_value_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto& kv : lhs) {
    auto it = rhs.find(kv.first);
    if (it == rhs.end()) {
      return false;
    }
    if (!eventEqual(kv.second, it->second)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool operator==(const Event& lhs, const Event& rhs) {
  return lhs.name == rhs.name && lhs.timestamp == rhs.timestamp &&
      eventEqual(lhs.data, rhs.data);
}

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
    static EventHandlers ehs;
    return ehs;
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

} // namespace monitor
} // namespace torch
