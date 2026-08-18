#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

#include <c10/core/Event.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>

#include <fmt/format.h>
#include <chrono>
#include <future>
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

    if (handlers_.contains(name)) {
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

// Each hooked backend records into its own FlightRecorder instance, so the
// caller says which one to read. Defaulting to the instance ProcessGroupGloo
// records into keeps existing callers -- the debug server's "FlightRecorder
// CPU" page among them -- seeing what they always did.
std::string frBackendParam(const Request& req) {
  auto backend = req.getParam("backend");
  return backend.empty() ? ::c10d::kDefaultFRBackend : backend;
}

RegisterHandler frTracehandler(
    "fr_trace_json",
    [](const Request& req, Response& res) {
      auto trace = ::c10d::dump_fr_trace_json(true, true, frBackendParam(req));
      res.setContent(std::move(trace), "application/json");
      res.setStatus(200);
    });

RegisterHandler frDumpFileHandler(
    "fr_dump_file",
    [](const Request& req, Response& res) {
      auto backend = frBackendParam(req);
      int rank = ::c10d::getFlightRecorder(backend)->getRank();
      // Nothing has told the recorder which rank this process is, so we would
      // write to <prefix>-1 and clobber every other rank's guess.
      if (rank < 0) {
        res.setStatus(503);
        res.setContent(
            "Flight Recorder rank is unset; no process group has registered with it yet",
            "text/plain");
        return;
      }

      // Single-flight guard: a polling health check must not spawn one worker
      // per request, all writing the same file. The previous future's state is
      // the signal - once wait_for(0) reports ready the worker has exited, so
      // reassigning cannot block on a join; otherwise a dump is still running
      // and we coalesce into it.
      static std::mutex dumpFutureMutex;
      static std::future<void> dumpFuture;

      {
        std::lock_guard<std::mutex> lock(dumpFutureMutex);
        if (dumpFuture.valid() &&
            dumpFuture.wait_for(std::chrono::seconds(0)) !=
                std::future_status::ready) {
          res.setStatus(200);
          res.setContent(
              fmt::format(
                  "Flight Recorder dump already in progress for rank {}", rank),
              "text/plain");
          return;
        }
        // std::launch::async so the future's destructor joins the worker
        // rather than leaking a detached thread.
        dumpFuture = std::async(std::launch::async, [rank, backend]() {
          ::c10d::dump_fr_trace_file(
              rank,
              /*includeCollectives=*/true,
              /*includeStackTraces=*/false,
              /*onlyActive=*/false,
              backend);
        });
      }

      res.setStatus(200);
      res.setContent(
          fmt::format("Flight Recorder dump initiated for rank {}", rank),
          "text/plain");
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
