#pragma once

#include <functional>
#include <map>

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
    if (auto it = registry_.find(pass.name()); it != registry_.end()) {
      LOG(WARNING) << "Pass " << pass.name() << " already registered";
      return;
    }

    GraphPassIdentifier name = pass.name();

    LOG(INFO) << "Pass " << name << " registered";
    registry_.insert({std::move(name), std::move(pass)});
  }

  void remove_pass(const GraphPassIdentifier& name) {
    if (!registry_.erase(name)) {
      LOG(WARNING) << "Pass " << name << " not registered but tried to remove";
      return;
    }
    LOG(INFO) << "Pass " << name << " unregistered";
  }

  const GraphPass& get_pass(const GraphPassIdentifier& name) {
    auto it = registry_.find(name);
    TORCH_CHECK(it != registry_.end(), "Pass ", name, " not registered to get");
    return it->second;
  }

 private:
  GraphPassRegistry() {
    LOG(INFO) << "Creating GraphPassRegistry";
  }

  std::map<std::string, GraphPass> registry_;

 public:
  GraphPassRegistry(GraphPassRegistry const&) = delete;
  void operator=(GraphPassRegistry const&) = delete;
};

} // namespace torch::nativert
