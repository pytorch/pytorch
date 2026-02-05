#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>

#include <fmt/format.h>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/csrc/distributed/c10d/control_plane/WaitCounterHandler.hpp>

namespace c10d::control_plane {

namespace {

class HandlerRegistry {
 public:
  void registerHandler(const std::string& name, HandlerFunc f) {
    std::unique_lock<std::shared_mutex> lock(handlersMutex_);

    if (handlers_.find(name) != handlers_.end()) {
      throw std::invalid_argument(
          fmt::format("Handler {} already registered", name));
    }

    handlers_[name] = std::move(f);
  }

  HandlerFunc getHandler(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(handlersMutex_);

    auto it = handlers_.find(name);
    if (it == handlers_.end()) {
      throw std::invalid_argument(
          fmt::format("Failed to find handler {}", name));
    }
    return handlers_[name];
  }

  std::vector<std::string> getHandlerNames() {
    std::shared_lock<std::shared_mutex> lock(handlersMutex_);

    std::vector<std::string> names;
    names.reserve(handlers_.size());
    for (const auto& [name, _] : handlers_) {
      names.push_back(name);
    }
    return names;
  }

 private:
  std::shared_mutex handlersMutex_;
  std::unordered_map<std::string, HandlerFunc> handlers_;
};

HandlerRegistry& getHandlerRegistry() {
  static HandlerRegistry registry;
  return registry;
}

RegisterHandler pingHandler{"ping", [](const Request&, Response& res) {
                              res.setContent("pong", "text/plain");
                              res.setStatus(200);
                            }};

RegisterHandler frTracehandler(
    "fr_trace_json",
    [](const Request&, Response& res) {
      auto trace = ::c10d::dump_fr_trace_json(true, true);
      res.setContent(std::move(trace), "application/json");
      res.setStatus(200);
    });

RegisterHandler waitCounterHandler{
    "wait_counter_values",
    [](const Request&, Response& res) {
      // Get all wait counter values from our tracking backend
      res.setContent(getWaitCounterValuesJson(), "application/json");
      res.setStatus(200);
    }};

#if !defined(FBCODE_CAFFE2)
// Initialize the wait counter backend
[[maybe_unused]] static bool init_backend = []() {
  ensureWaitCounterBackendRegistered();
  return true;
}();
#endif

#ifndef _WIN32
RegisterHandler pyspyHandler{
    "pyspy_dump",
    [](const Request& req, Response& res) {
      pid_t target = getpid();
      std::string cmd = "py-spy dump";
      cmd += " --pid " + std::to_string(target);
      if (!req.getParam("native").empty()) {
        cmd += " --native";
      }
      if (!req.getParam("subprocesses").empty()) {
        cmd += " --subprocesses";
      }
      if (!req.getParam("nonblocking").empty()) {
        cmd += " --nonblocking";
      }
      cmd += " 2>&1";
      std::array<char, 4096> buf{};
      std::string output;
      FILE* pipe = popen(cmd.c_str(), "r");
      if (!pipe) {
        throw std::runtime_error("Failed to start py-spy, not installed?");
      }
      while (fgets(buf.data(), buf.size(), pipe)) {
        output.append(buf.data());
      }
      int rc = pclose(pipe);

      // Get all wait counter values from our tracking backend
      res.setContent(std::move(output), "text/plain");
      if (rc != 0) {
        res.setStatus(500);
      } else {
        res.setStatus(200);
      }
    }};
#endif

} // namespace

void registerHandler(const std::string& name, HandlerFunc f) {
  return getHandlerRegistry().registerHandler(name, std::move(f));
}

HandlerFunc getHandler(const std::string& name) {
  return getHandlerRegistry().getHandler(name);
}

std::vector<std::string> getHandlerNames() {
  return getHandlerRegistry().getHandlerNames();
}

} // namespace c10d::control_plane
