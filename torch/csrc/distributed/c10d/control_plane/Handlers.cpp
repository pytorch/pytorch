#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

#include <fmt/format.h>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

namespace c10d {
namespace control_plane {

namespace {

class HandlerRegistry {
 public:
  void registerHandler(const std::string& name, HandlerFunc f) {
    std::unique_lock<std::shared_mutex> lock(handlersMutex_);

    if (handlers_.find(name) != handlers_.end()) {
      throw std::runtime_error(
          fmt::format("Handler {} already registered", name));
    }

    handlers_[name] = f;
  }

  HandlerFunc getHandler(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(handlersMutex_);

    auto it = handlers_.find(name);
    if (it == handlers_.end()) {
      throw std::runtime_error(fmt::format("Failed to find handler {}", name));
    }
    return handlers_[name];
  }

  std::vector<std::string> getHandlerNames() {
    std::shared_lock<std::shared_mutex> lock(handlersMutex_);

    std::vector<std::string> names;
    for (const auto& [name, _] : handlers_) {
      names.push_back(name);
    }
    return names;
  }

 private:
  std::shared_mutex handlersMutex_{};
  std::unordered_map<std::string, HandlerFunc> handlers_{};
};

HandlerRegistry& getHandlerRegistry() {
  static HandlerRegistry registry;
  return registry;
}

RegisterHandler pingHandler{"ping", [](const Request&, Response& res) {
                              res.setContent("pong", "text/plain");
                            }};

} // namespace

void registerHandler(const std::string& name, HandlerFunc f) {
  return getHandlerRegistry().registerHandler(name, f);
}

HandlerFunc getHandler(const std::string& name) {
  return getHandlerRegistry().getHandler(name);
}

std::vector<std::string> getHandlerNames() {
  return getHandlerRegistry().getHandlerNames();
}

} // namespace control_plane
} // namespace c10d
