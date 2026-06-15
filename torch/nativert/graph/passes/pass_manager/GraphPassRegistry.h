#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <shared_mutex>

#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/nativert/graph/Graph.h>
namespace torch::nativert {

using PassSignature = std::function<bool(Graph*)>;
using GraphPassIdentifier = std::string;

class GraphPass {
 public:
  GraphPass(GraphPassIdentifier&& name, PassSignature&& pass)
      : name_(std::move(name)), pass_(std::move(pass)) {}

  const GraphPassIdentifier& name() const {
    return name_;
  }

  const PassSignature& get() const {
    return pass_;
  }

 private:
  GraphPassIdentifier name_;
  PassSignature pass_;
};

class GraphPassRegistry {
 public:
  static GraphPassRegistry& get() {
    static GraphPassRegistry instance;
    return instance;
  }

  static void add_pass(GraphPassIdentifier&& name, PassSignature&& pass) {
    GraphPassRegistry::get().add_pass(
        GraphPass(std::move(name), std::move(pass)));
  }

  void add_pass(GraphPass&& pass) {
    GraphPassIdentifier name = pass.name();
    bool already_registered = false;
    {
      std::unique_lock lock(mutex_);
      if (registry_.find(name) != registry_.end()) {
        already_registered = true;
      } else {
        registry_.insert({name, std::move(pass)});
      }
    }
    if (already_registered) {
      LOG(WARNING) << "Pass " << name << " already registered";
    } else {
      LOG(INFO) << "Pass " << name << " registered";
    }
  }

  void remove_pass(const GraphPassIdentifier& name) {
    bool removed = false;
    {
      std::unique_lock lock(mutex_);
      removed = registry_.erase(name) > 0;
    }
    if (removed) {
      LOG(INFO) << "Pass " << name << " unregistered";
    } else {
      LOG(WARNING) << "Pass " << name << " not registered but tried to remove";
    }
  }

  // Returns by value: a reference into registry_ would be invalidated by a
  // concurrent remove_pass once the shared_lock is released.
  GraphPass get_pass(const GraphPassIdentifier& name) {
    std::shared_lock lock(mutex_);
    auto it = registry_.find(name);
    TORCH_CHECK(it != registry_.end(), "Pass ", name, " not registered to get");
    return it->second;
  }

 private:
  GraphPassRegistry() {
    LOG(INFO) << "Creating GraphPassRegistry";
  }

  std::map<std::string, GraphPass> registry_;
  mutable std::shared_mutex mutex_;

 public:
  GraphPassRegistry(GraphPassRegistry const&) = delete;
  void operator=(GraphPassRegistry const&) = delete;
};

} // namespace torch::nativert
